"""Turning sequences into numbers. A starting point; change it however you like."""

from __future__ import annotations

import numpy as np


def one_hot(seqs: list[str], alphabet: str, length: int) -> np.ndarray:
    """(n, length * len(alphabet)). Unknown residues encode as all zeros."""
    index = {a: i for i, a in enumerate(alphabet)}
    x = np.zeros((len(seqs), length, len(alphabet)), dtype=np.float32)
    for n, s in enumerate(seqs):
        if len(s) != length:
            raise ValueError(f"sequence {n} has length {len(s)}, expected {length}")
        for p, a in enumerate(s):
            j = index.get(a)
            if j is not None:
                x[n, p, j] = 1.0
    return x.reshape(len(seqs), -1)


def build(seqs: list[str], bundle: dict) -> np.ndarray:
    """Featurise using whatever a saved bundle says it was fitted on."""
    if bundle["features"] == "onehot":
        return one_hot(seqs, bundle["alphabet"], bundle["length"])
    raise ValueError(f"unknown feature kind {bundle['features']!r}")
