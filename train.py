from __future__ import annotations

"""
Placeholder training scaffold. Full gradient descent is non-trivial in pure numpy; to keep dependencies minimal and
maintain tests, we provide data preparation, batching, and a stub for future optimization.

Future improvement: Port model to PyTorch/JAX for proper training while keeping inference-compatible weights export.
"""

import argparse
import numpy as np

from .tokenizer import SimpleTokenizer
from .transformer import TransformerLM
from .data import load_qa_json, build_sequences, iter_minibatches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", help="Path to training JSON file")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--max-len", type=int, default=128)
    args = p.parse_args()

    tk = SimpleTokenizer()
    items = load_qa_json(args.data)
    seqs = build_sequences(items, tk)

    model = TransformerLM(vocab_size=tk.vocab_size, dim=args.dim, n_layers=args.layers, n_heads=args.heads, ff_hidden=args.ff, max_len=args.max_len)

    # Smoke: iterate a couple mini-batches and compute logits to validate wiring
    n_tokens = 0
    for i, batch in enumerate(iter_minibatches(seqs, args.batch_size)):
        logits = model.forward(batch)
        n_tokens += int(batch.shape[0] * batch.shape[1])
        if i >= 2:
            break
    print(f"Prepared {len(seqs)} sequences, seen ~{n_tokens} tokens across a few batches. Training stub complete.")


if __name__ == "__main__":
    main()
