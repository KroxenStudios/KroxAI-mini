from __future__ import annotations

import math
from typing import Optional

import numpy as np


def xavier_init(shape, gain=1.0, rng=None):
    rng = rng or np.random.default_rng()
    fan_in, fan_out = shape[0], shape[1]
    limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = np.ones((dim,), dtype=np.float32)
        self.beta = np.zeros((dim,), dtype=np.float32)
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (B, T, C)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        xhat = (x - mean) / np.sqrt(var + self.eps)
        return xhat * self.gamma + self.beta

    def state_dict(self, prefix: str):
        return {
            f"{prefix}.gamma": self.gamma,
            f"{prefix}.beta": self.beta,
        }

    def load_state_dict(self, sd, prefix: str):
        if f"{prefix}.gamma" in sd:
            self.gamma = sd[f"{prefix}.gamma"].astype(np.float32)
        if f"{prefix}.beta" in sd:
            self.beta = sd[f"{prefix}.beta"].astype(np.float32)


class MultiHeadSelfAttention:
    def __init__(self, dim: int, n_heads: int, rng=None):
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        rng = rng or np.random.default_rng()

        self.W_q = xavier_init((dim, dim), rng=rng)
        self.W_k = xavier_init((dim, dim), rng=rng)
        self.W_v = xavier_init((dim, dim), rng=rng)
        self.W_o = xavier_init((dim, dim), rng=rng)

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # x: (B, T, C)
        B, T, C = x.shape
        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v

        # reshape to heads
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)  # (B, H, T, Hd)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # scaled dot-product attention with causal mask
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)  # (B,H,T,T)
        # causal mask to prevent attending to future positions
        causal = np.triu(np.ones((T, T), dtype=np.float32), k=1)
        attn_scores = attn_scores - 1e9 * causal  # broadcast over B,H
        if mask is not None:
            attn_scores = attn_scores + mask  # assume mask already has -inf where to mask

        attn_weights = np.exp(attn_scores - attn_scores.max(axis=-1, keepdims=True))
        attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)

        context = attn_weights @ v  # (B,H,T,Hd)
        context = context.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = context @ self.W_o
        return out

    def state_dict(self, prefix: str):
        return {
            f"{prefix}.W_q": self.W_q,
            f"{prefix}.W_k": self.W_k,
            f"{prefix}.W_v": self.W_v,
            f"{prefix}.W_o": self.W_o,
        }

    def load_state_dict(self, sd, prefix: str):
        for name in ["W_q", "W_k", "W_v", "W_o"]:
            k = f"{prefix}.{name}"
            if k in sd:
                setattr(self, name, sd[k].astype(np.float32))


class FeedForward:
    def __init__(self, dim: int, hidden: int, rng=None):
        rng = rng or np.random.default_rng()
        self.W1 = xavier_init((dim, hidden), rng=rng)
        self.b1 = np.zeros((hidden,), dtype=np.float32)
        self.W2 = xavier_init((hidden, dim), rng=rng)
        self.b2 = np.zeros((dim,), dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x @ self.W1 + self.b1
        h = np.maximum(h, 0)  # ReLU
        out = h @ self.W2 + self.b2
        return out

    def state_dict(self, prefix: str):
        return {
            f"{prefix}.W1": self.W1,
            f"{prefix}.b1": self.b1,
            f"{prefix}.W2": self.W2,
            f"{prefix}.b2": self.b2,
        }

    def load_state_dict(self, sd, prefix: str):
        for name in ["W1", "b1", "W2", "b2"]:
            k = f"{prefix}.{name}"
            if k in sd:
                setattr(self, name, sd[k].astype(np.float32))


class PositionalEncoding:
    def __init__(self, dim: int, max_len: int = 2048):
        pe = np.zeros((max_len, dim), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, dim, 2, dtype=np.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe  # (T, C)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (B, T, C)
        T = x.shape[1]
        return x + self.pe[None, :T, :]


class TransformerBlock:
    def __init__(self, dim: int, n_heads: int, ff_hidden: int, rng=None):
        rng = rng or np.random.default_rng()
        self.ln1 = LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads, rng=rng)
        self.ln2 = LayerNorm(dim)
        self.ff = FeedForward(dim, ff_hidden, rng=rng)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

    def state_dict(self, prefix: str):
        sd = {}
        sd.update(self.ln1.state_dict(f"{prefix}.ln1"))
        sd.update(self.attn.state_dict(f"{prefix}.attn"))
        sd.update(self.ln2.state_dict(f"{prefix}.ln2"))
        sd.update(self.ff.state_dict(f"{prefix}.ff"))
        return sd

    def load_state_dict(self, sd, prefix: str):
        self.ln1.load_state_dict(sd, f"{prefix}.ln1")
        self.attn.load_state_dict(sd, f"{prefix}.attn")
        self.ln2.load_state_dict(sd, f"{prefix}.ln2")
        self.ff.load_state_dict(sd, f"{prefix}.ff")


class TransformerLM:
    """
    Minimal numpy Transformer Language Model for inference and simple generation.
    This is not optimized and intended for small tests and educational purposes.
    """

    def __init__(self, vocab_size: int, dim: int = 128, n_layers: int = 4, n_heads: int = 4, ff_hidden: int = 256, max_len: int = 256, rng=None):
        self.vocab_size = int(vocab_size)
        self.dim = dim
        self.max_len = max_len
        rng = rng or np.random.default_rng(42)

        # token embeddings and output head
        self.token_emb = xavier_init((self.vocab_size, dim), rng=rng)
        self.pos_enc = PositionalEncoding(dim, max_len=max_len)
        self.blocks = [TransformerBlock(dim, n_heads, ff_hidden, rng=rng) for _ in range(n_layers)]
        self.ln_f = LayerNorm(dim)
        self.head = xavier_init((dim, self.vocab_size), rng=rng)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Compute logits for each position.
        token_ids: numpy array (B, T) of ints
        returns: logits (B, T, vocab_size)
        """
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be 2D (B, T)")
        B, T = token_ids.shape
        if T > self.max_len:
            token_ids = token_ids[:, -self.max_len:]
            T = token_ids.shape[1]

        # Embedding lookup
        x = self.token_emb[token_ids]  # (B, T, C)
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = x @ self.head  # (B, T, vocab_size)
        return logits

    def generate(self, token_ids: np.ndarray, max_new_tokens: int = 20, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None) -> np.ndarray:
        """Autoregressive generation in numpy (greedy/top-k/top-p sampling).
        token_ids: (B, T)
        returns: (B, T + max_new_tokens)
        """
        B, T = token_ids.shape
        out = token_ids.copy()
        for _ in range(max_new_tokens):
            window = out[:, -self.max_len:]
            logits = self.forward(window)
            next_logits = logits[:, -1, :] / max(1e-6, temperature)
            if top_k is not None and top_k > 0 and top_k < next_logits.shape[-1]:
                # top-k filter
                kth = np.partition(next_logits, -top_k, axis=-1)[:, -top_k][:, None]
                mask = next_logits < kth
                next_logits = np.where(mask, -1e9, next_logits)

            # softmax
            probs = np.exp(next_logits - next_logits.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            # top-p (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                next_tokens = []
                for i in range(B):
                    p = probs[i]
                    order = np.argsort(-p)
                    p_sorted = p[order]
                    cum = np.cumsum(p_sorted)
                    # mask tokens that push cumulative prob over top_p, but keep at least first token
                    mask = cum > top_p
                    if mask.size > 0:
                        mask[0] = False
                    p_sorted = np.where(mask, 0.0, p_sorted)
                    s = p_sorted.sum()
                    if s <= 0:
                        p_sorted = np.ones_like(p_sorted) / p_sorted.size
                    else:
                        p_sorted = p_sorted / s
                    choice_sorted = np.random.choice(p_sorted.size, p=p_sorted)
                    next_tokens.append(order[choice_sorted])
                next_tokens = np.array(next_tokens, dtype=np.int64)
                out = np.concatenate([out, next_tokens[:, None]], axis=1)
                continue
            # sample
            next_tokens = np.array([np.random.choice(self.vocab_size, p=probs[i]) for i in range(B)], dtype=np.int64)
            out = np.concatenate([out, next_tokens[:, None]], axis=1)
        return out

    def state_dict(self):
        sd = {
            "token_emb": self.token_emb,
            "head": self.head,
        }
        # blocks
        for i, blk in enumerate(self.blocks):
            sd.update(blk.state_dict(f"blocks.{i}"))
        # final layer norm
        sd.update(self.ln_f.state_dict("ln_f"))
        # positional enc is deterministic; omit
        return sd

    def load_state_dict(self, sd):
        if "token_emb" in sd:
            self.token_emb = sd["token_emb"].astype(np.float32)
        if "head" in sd:
            self.head = sd["head"].astype(np.float32)
        for i, blk in enumerate(self.blocks):
            blk.load_state_dict(sd, f"blocks.{i}")
        self.ln_f.load_state_dict(sd, "ln_f")
