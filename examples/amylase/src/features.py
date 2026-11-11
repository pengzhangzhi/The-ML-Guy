"""Ways of turning a variant sequence into numbers."""

from __future__ import annotations

import numpy as np


def one_hot(seqs: list[str], alphabet: str, length: int) -> np.ndarray:
    """(n, length * len(alphabet)). Every position of the sequence, unknown residues all zero."""
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


def mutation_vocab(seqs: list[str], reference: str) -> list[tuple[int, str]]:
    """Every (position, residue) substitution seen in these sequences, in a stable order."""
    seen = {(p, a) for s in seqs for p, (r, a) in enumerate(zip(reference, s)) if r != a}
    return sorted(seen)


def mutation_matrix(seqs: list[str], reference: str, vocab: list[tuple[int, str]]) -> np.ndarray:
    """(n, len(vocab) + 1): one column per known substitution, plus the number of mutations.

    Substitutions outside the vocabulary are not represented, but still raise the count,
    so a variant carrying an unseen mutation is not silently identical to the reference.
    """
    index = {key: i for i, key in enumerate(vocab)}
    x = np.zeros((len(seqs), len(vocab) + 1), dtype=np.float32)
    for n, s in enumerate(seqs):
        if len(s) != len(reference):
            raise ValueError(f"sequence {n} has length {len(s)}, expected {len(reference)}")
        count = 0
        for p, (r, a) in enumerate(zip(reference, s)):
            if r != a:
                count += 1
                j = index.get((p, a))
                if j is not None:
                    x[n, j] = 1.0
        x[n, -1] = count
    return x


def build(seqs: list[str], bundle: dict) -> np.ndarray:
    """Featurise using whatever a trained bundle says it was fitted on."""
    kind = bundle["features"]
    if kind == "onehot":
        return one_hot(seqs, bundle["alphabet"], bundle["length"])
    if kind == "mutations":
        return mutation_matrix(seqs, bundle["reference"], [tuple(v) for v in bundle["vocab"]])
    raise ValueError(f"unknown feature kind {kind!r}")
