from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .tokenizer import SimpleTokenizer
from .torch_model import TorchTransformerLM
from .configs import get_preset


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def seq_logprob(model: TorchTransformerLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute log-prob of target sequence y given prompt x (per example).
    x: (B, Tx) prompt ids; y: (B, Ty) target ids (no BOS).
    Returns: (B,) log-prob sums over y tokens.
    """
    device = next(model.parameters()).device
    B, Ty = y.shape
    # concatenate x and y[:-1] to predict y tokens
    xy = torch.cat([x, y[:, :-1]], dim=1)
    logits = model(xy)  # (B, T, V)
    # take last Ty-1 positions that correspond to next tokens in y
    pred = logits[:, - (Ty - 1):, :]  # (B, Ty-1, V)
    target = y[:, 1:]
    logp = F.log_softmax(pred, dim=-1)
    # gather logp at target indices
    lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, Ty-1)
    # sum; prepend first token prob by conditioning on last token of x
    # To keep it simple, we ignore the very first y token's prob (approximation).
    return lp.sum(dim=1)


def dpo_step(model: TorchTransformerLM, x: torch.Tensor, y_pos: torch.Tensor, y_neg: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """Pairwise preference loss: encourage chosen > rejected.
    L = -log(sigmoid(beta * (logp_pos - logp_neg)))
    """
    lp_pos = seq_logprob(model, x, y_pos)
    lp_neg = seq_logprob(model, x, y_neg)
    diff = lp_pos - lp_neg
    loss = F.softplus(-beta * diff)  # = -log(sigmoid(beta*diff))
    return loss.mean()


def collate_batch(batch: List[Tuple[List[int], List[int], List[int]]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def pad(seqs: List[List[int]]) -> torch.Tensor:
        m = max(len(s) for s in seqs)
        out = torch.full((len(seqs), m), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return out
    X = pad([b[0] for b in batch])
    Yp = pad([b[1] for b in batch])
    Yn = pad([b[2] for b in batch])
    return X, Yp, Yn


def main():
    ap = argparse.ArgumentParser(description="Simple preference fine-tuning (DPO-style) for TorchTransformerLM")
    ap.add_argument('--data', type=str, required=True, help='Path to preference JSONL with {prompt, chosen, rejected}')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--beta', type=float, default=0.1)
    ap.add_argument('--preset', type=str, default='small')
    ap.add_argument('--ckpt-out', type=str, default='EXE_Output/kroxai_dpo.pt')
    args = ap.parse_args()

    path = Path(args.data)
    rows = load_jsonl(path)
    if not rows:
        raise SystemExit(f"No data in {path}")

    tk = SimpleTokenizer()
    cfg = get_preset(args.preset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TorchTransformerLM(vocab_size=tk.vocab_size, dim=cfg.dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads, ff_hidden=cfg.ff_hidden, max_len=cfg.max_len).to(device)

    opt = AdamW(model.parameters(), lr=args.lr)
    model.train()

    # Build dataset tuples (x, y_pos, y_neg)
    ds: List[Tuple[List[int], List[int], List[int]]] = []
    for r in rows:
        p = str(r.get('prompt') or r.get('question') or '')
        pos = str(r.get('chosen') or r.get('answer') or '')
        neg = str(r.get('rejected') or '')
        if not p or not pos or not neg:
            continue
        x = tk.encode(p, add_bos=True)
        yp = tk.encode(pos, add_bos=False, add_eos=True)
        yn = tk.encode(neg, add_bos=False, add_eos=True)
        ds.append((x, yp, yn))
    if not ds:
        raise SystemExit("No valid preference pairs found")

    import random
    for epoch in range(max(1, args.epochs)):
        random.shuffle(ds)
        total = 0.0
        nsteps = 0
        for i in range(0, len(ds), args.batch_size):
            batch = ds[i:i+args.batch_size]
            x, yp, yn = collate_batch(batch, pad_id=tk.PAD)
            x = x.to(device)
            yp = yp.to(device)
            yn = yn.to(device)
            opt.zero_grad(set_to_none=True)
            loss = dpo_step(model, x, yp, yn, beta=args.beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            nsteps += 1
        avg = total / max(1, nsteps)
        print(f"epoch {epoch+1} avg_loss={avg:.4f}")

    # Save checkpoint
    out = Path(args.ckpt_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sd = {
        'model': model.state_dict(),
        'config': {
            'dim': cfg.dim,
            'n_layers': cfg.n_layers,
            'n_heads': cfg.n_heads,
            'ff_hidden': cfg.ff_hidden,
            'max_len': cfg.max_len,
            'tokenizer': {'type': 'simple', 'pad_id': tk.PAD, 'bos_id': tk.BOS, 'eos_id': tk.EOS},
        }
    }
    torch.save(sd, str(out))
    print(f"saved checkpoint to {out}")


if __name__ == '__main__':
    main()
