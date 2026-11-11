"""Turning a table of input columns into a matrix of numbers.

This is a starting point that works on most tabular data without being told anything: it
looks at each input column and picks an encoding. Replace it the moment you have a better
idea, which will usually be quickly.

Column kinds it recognises:
  numeric     used as is
  category    one-hot over the values seen in training
  text        character composition and log length (order is thrown away)
  text_fixed  one-hot per position, when every value has the same length
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_CATEGORIES = 50


def describe(df: pd.DataFrame, columns: list[str], declared: dict | None = None) -> list[dict]:
    """Decide how to encode each input column. `declared` is the task's [input] table."""
    declared = declared or {}
    spec = []
    for name in columns:
        values = df[name]
        if pd.api.types.is_numeric_dtype(values):
            spec.append({"name": name, "kind": "numeric"})
            continue
        text = values.astype(str)
        lengths = text.str.len()
        alphabet = declared.get("alphabet") or "".join(sorted(set("".join(text.head(2000)))))
        if lengths.median() <= 25 and text.nunique() <= MAX_CATEGORIES:
            spec.append({"name": name, "kind": "category", "categories": sorted(text.unique())})
        elif lengths.nunique() == 1:
            spec.append({"name": name, "kind": "text_fixed", "alphabet": alphabet,
                         "length": int(lengths.iloc[0])})
        else:
            spec.append({"name": name, "kind": "text", "alphabet": alphabet})
    return spec


def _category(values: pd.Series, categories: list[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(categories)}
    x = np.zeros((len(values), len(categories)), dtype=np.float32)
    for n, v in enumerate(values.astype(str)):
        j = index.get(v)
        if j is not None:  # a value never seen in training stays all zeros
            x[n, j] = 1.0
    return x


def _text(values: pd.Series, alphabet: str) -> np.ndarray:
    """How often each character occurs, plus log length. Works for any length."""
    index = {a: i for i, a in enumerate(alphabet)}
    x = np.zeros((len(values), len(alphabet) + 1), dtype=np.float32)
    for n, s in enumerate(values.astype(str)):
        if not s:
            raise ValueError(f"row {n} has an empty value")
        for a in s:
            j = index.get(a)
            if j is not None:
                x[n, j] += 1.0
        x[n, : len(alphabet)] /= len(s)
        x[n, -1] = np.log(len(s))
    return x


def _text_fixed(values: pd.Series, alphabet: str, length: int) -> np.ndarray:
    """One column per (position, character). Only meaningful when position means something."""
    index = {a: i for i, a in enumerate(alphabet)}
    x = np.zeros((len(values), length, len(alphabet)), dtype=np.float32)
    for n, s in enumerate(values.astype(str)):
        if len(s) != length:
            raise ValueError(f"row {n} has length {len(s)}, expected {length}")
        for p, a in enumerate(s):
            j = index.get(a)
            if j is not None:
                x[n, p, j] = 1.0
    return x.reshape(len(values), -1)


def build(df: pd.DataFrame, spec: list[dict]) -> np.ndarray:
    parts = []
    for column in spec:
        values = df[column["name"]]
        if column["kind"] == "numeric":
            parts.append(values.to_numpy(dtype=np.float32).reshape(-1, 1))
        elif column["kind"] == "category":
            parts.append(_category(values, column["categories"]))
        elif column["kind"] == "text":
            parts.append(_text(values, column["alphabet"]))
        elif column["kind"] == "text_fixed":
            parts.append(_text_fixed(values, column["alphabet"], column["length"]))
        else:
            raise ValueError(f"unknown column kind {column['kind']!r}")
    return np.hstack(parts)


def summarise(spec: list[dict]) -> str:
    return ", ".join(f"{c['name']} as {c['kind']}" for c in spec)
