"""Deterministic local embeddings.

No API calls: a hashed bag-of-words + bigram + character-trigram model
projected to a fixed dimension. It is weak compared to a real sentence
encoder, but it is *identical across every experimental condition*, so it
never advantages one arm of the ablation over another. Swap `embed()` for a
real encoder in production; nothing else in the package depends on how
vectors are made.

The hash is 32-bit FNV-1a so the browser visualisation (viz/engine.js) can
reproduce every vector bit-for-bit. Vectors are float64 for the same reason.
"""

from __future__ import annotations

import re

import numpy as np

DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _features(text: str):
    words = _TOKEN_RE.findall(text.lower())
    for w in words:
        yield w, 1.0
    for a, b in zip(words, words[1:]):
        yield f"{a}_{b}", 0.7
    packed = " ".join(words)
    for i in range(len(packed) - 2):
        yield f"#{packed[i:i + 3]}", 0.25


def embed(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float64)
    for feat, weight in _features(text):
        h = fnv1a(feat)
        v[h % DIM] += (1.0 if (h >> 17) & 1 else -1.0) * weight
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def embed_many(texts) -> np.ndarray:
    if not texts:
        return np.zeros((0, DIM), dtype=np.float64)
    return np.stack([embed(t) for t in texts])


def cosine(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    if M.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return M @ q
