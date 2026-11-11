"""Optional helpers for projects whose input is a biological sequence.

Nothing in the core imports this except the prediction page, and only when a task asks for
it. A project in another domain never touches it. Project code is free to import it:

    from mlguy.sequence import read_sequences, check_sequence
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FASTA_SUFFIXES = {".fasta", ".fa", ".faa", ".fas"}


def parse_fasta(text: str) -> tuple[list[str], list[str]]:
    ids, seqs, cur = [], [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
            elif ids:
                seqs.append("")  # header with no sequence
            ids.append(line[1:].split()[0] if len(line) > 1 else f"seq{len(ids) + 1}")
        else:
            cur.append(line)
    if ids:
        seqs.append("".join(cur))
    return ids, [s.upper() for s in seqs]


def parse_text(text: str) -> tuple[list[str], list[str]]:
    """Pasted input: FASTA if it has '>' headers, otherwise one sequence per line."""
    if text.lstrip().startswith(">"):
        return parse_fasta(text)
    seqs = [ln.strip().upper() for ln in text.splitlines() if ln.strip()]
    return [f"seq{i + 1}" for i in range(len(seqs))], seqs


def read_sequences(path: str | Path, sequence_column: str = "sequence") -> tuple[list[str], list[str]]:
    """Return (ids, sequences) from a CSV with a sequence column, or a FASTA file."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in FASTA_SUFFIXES or text.lstrip().startswith(">"):
        return parse_fasta(text)
    df = pd.read_csv(path)
    if sequence_column not in df.columns:
        raise ValueError(f"{path.name} needs a '{sequence_column}' column; found {list(df.columns)}")
    seqs = df[sequence_column].astype(str).str.strip().str.upper().tolist()
    ids = df["id"].astype(str).tolist() if "id" in df.columns else [f"seq{i + 1}" for i in range(len(seqs))]
    return ids, seqs


def check_sequence(seq: str, spec: dict) -> str | None:
    """Return None if the sequence fits the spec, else a plain-language reason."""
    if not seq:
        return "empty sequence"
    alphabet = spec.get("alphabet")
    if alphabet:
        bad = sorted(set(seq) - set(alphabet))
        if bad:
            return f"contains characters the model was not trained on: {' '.join(bad)}"
    length = spec.get("length")
    if length and len(seq) != length:
        return f"length is {len(seq)}, but this model only accepts length {length}"
    return None


def describe_mutations(seq: str, reference: str | None) -> str:
    """'A12V, G45D' relative to the reference; '' if no reference or lengths differ."""
    if not reference or len(seq) != len(reference):
        return ""
    muts = [f"{r}{i + 1}{s}" for i, (r, s) in enumerate(zip(reference, seq)) if r != s]
    return ", ".join(muts) if muts else "reference"
