"""Turning sequences into numbers.

These are unrelated proteins of different lengths, so nothing position-wise applies. What is
left is counting: which residues a protein is made of, and which pairs follow each other.
"""

from __future__ import annotations

import numpy as np


def composition(seqs: list[str], alphabet: str) -> np.ndarray:
    """(n, len(alphabet) + 1): how often each residue occurs, plus log length."""
    index = {a: i for i, a in enumerate(alphabet)}
    x = np.zeros((len(seqs), len(alphabet) + 1), dtype=np.float32)
    for n, s in enumerate(seqs):
        if not s:
            raise ValueError(f"sequence {n} is empty")
        for a in s:
            j = index.get(a)
            if j is not None:
                x[n, j] += 1.0
        x[n, : len(alphabet)] /= len(s)
        x[n, -1] = np.log(len(s))
    return x


def dipeptide(seqs: list[str], alphabet: str) -> np.ndarray:
    """(n, 20 + 400 + 1): residue composition, adjacent-pair composition, and log length.

    Pair counts carry a little local order, which plain composition throws away entirely.
    """
    index = {a: i for i, a in enumerate(alphabet)}
    k = len(alphabet)
    x = np.zeros((len(seqs), k + k * k + 1), dtype=np.float32)
    for n, s in enumerate(seqs):
        if not s:
            raise ValueError(f"sequence {n} is empty")
        for a in s:
            j = index.get(a)
            if j is not None:
                x[n, j] += 1.0
        for a, b in zip(s, s[1:]):
            i, j = index.get(a), index.get(b)
            if i is not None and j is not None:
                x[n, k + i * k + j] += 1.0
        x[n, :k] /= len(s)
        if len(s) > 1:
            x[n, k : k + k * k] /= len(s) - 1
        x[n, -1] = np.log(len(s))
    return x


def build(seqs: list[str], bundle: dict) -> np.ndarray:
    """Featurise using whatever a saved bundle says it was fitted on."""
    kind = bundle["features"]
    if kind == "composition":
        return composition(seqs, bundle["alphabet"])
    if kind == "dipeptide":
        return dipeptide(seqs, bundle["alphabet"])
    raise ValueError(f"unknown feature kind {kind!r}")
