"""Turning sequences into numbers. A starting point; change it however you like."""

from __future__ import annotations

import numpy as np


def one_hot(seqs: list[str], alphabet: str, length: int) -> np.ndarray:
    """(n, length * len(alphabet)). Needs every sequence to be the same length.

    Good when the sequences are variants of one protein, so that position 42 means the same
    thing in every row. Useless when they are unrelated proteins.
    """
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


def composition(seqs: list[str], alphabet: str) -> np.ndarray:
    """(n, len(alphabet) + 1): how often each residue occurs, plus log length.

    Works for sequences of any length, and for unrelated proteins. It throws away all
    order, so it is a floor to beat rather than a good model.
    """
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


def build(seqs: list[str], bundle: dict) -> np.ndarray:
    """Featurise using whatever a saved bundle says it was fitted on."""
    kind = bundle["features"]
    if kind == "onehot":
        return one_hot(seqs, bundle["alphabet"], bundle["length"])
    if kind == "composition":
        return composition(seqs, bundle["alphabet"])
    raise ValueError(f"unknown feature kind {kind!r}")
