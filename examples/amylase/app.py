"""The page for this project: rank amylase variants by predicted stain-removal activity.

    python app.py [--port N] [--share]      (or: mlguy serve)

Written for this problem, not supplied by the framework. It takes what an enzyme engineer
has to hand (a FASTA file, a CSV with a `sequence` column, or pasted sequences), names each
variant by its mutations against the wild type, and ranks them best first. Predictions come
from predict.py, the same code path the scores in REPORT.md were measured on.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
with open(HERE / "task" / "task.toml", "rb") as f:
    SPEC = tomllib.load(f)["input"]

TITLE = "Amylase stain-removal predictor"
ABOUT = ("Predicts the stain-removal activity of *Bacillus subtilis* alpha-amylase variants from "
         "their amino-acid sequence, and ranks them best first.")
TESTED = ("**How it was tested.** Trained on variants with 2 to 8 mutations and tested on single "
          "mutants it had never seen. It has not been shown to work on insertions, deletions, or "
          "other enzymes.")


def read_variants(text: str) -> pd.DataFrame:
    """FASTA, a CSV with a `sequence` column, or one sequence per line."""
    text = text.strip()
    if text.startswith(">"):
        names, seqs = [], []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(">"):
                names.append(line[1:].split()[0] if len(line) > 1 else f"variant{len(names) + 1}")
                seqs.append("")
            elif line and names:
                seqs[-1] += line
        return pd.DataFrame({"name": names, "sequence": seqs})
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and "sequence" in [c.strip() for c in lines[0].split(",")]:
        from io import StringIO

        df = pd.read_csv(StringIO(text))
        names = df["id"].astype(str) if "id" in df.columns else [f"variant{i + 1}" for i in range(len(df))]
        return pd.DataFrame({"name": list(names), "sequence": df["sequence"].astype(str)})
    return pd.DataFrame({"name": [f"variant{i + 1}" for i in range(len(lines))], "sequence": lines})


def problem_with(seq: str) -> str | None:
    if not seq:
        return "empty sequence"
    bad = sorted(set(seq) - set(SPEC["alphabet"]))
    if bad:
        return f"contains characters the model was not trained on: {' '.join(bad)}"
    if len(seq) != SPEC["length"]:
        return f"length is {len(seq)}; this model only accepts {SPEC['length']} (substitutions only)"
    return None


def mutations(seq: str) -> str:
    found = [f"{r}{i + 1}{s}" for i, (r, s) in enumerate(zip(SPEC["reference"], seq)) if r != s]
    return ", ".join(found) or "wild type"


def rank_variants(text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ranked predictions, rejected variants)."""
    variants = read_variants(text)
    variants["sequence"] = variants["sequence"].str.strip().str.upper()
    variants["reason"] = variants["sequence"].map(problem_with)
    rejected = variants.loc[variants["reason"].notna(), ["name", "reason"]].reset_index(drop=True)
    kept = variants[variants["reason"].isna()].reset_index(drop=True)
    columns = ["rank", "name", "predicted activity", "mutations"]
    if kept.empty:
        return pd.DataFrame(columns=columns), rejected

    with tempfile.TemporaryDirectory() as tmp:
        inputs, outputs = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
        kept[["sequence"]].to_csv(inputs, index=False)
        proc = subprocess.run([sys.executable, "predict.py", "--model", "model", "--input", str(inputs),
                               "--output", str(outputs)], cwd=HERE, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"the model failed to run:\n{proc.stderr.strip()[-1000:]}")
        kept["predicted activity"] = pd.read_csv(outputs)["prediction"].to_numpy()

    kept["mutations"] = kept["sequence"].map(mutations)
    kept = kept.sort_values("predicted activity", ascending=False, kind="stable").reset_index(drop=True)
    kept.insert(0, "rank", range(1, len(kept) + 1))
    return kept[columns], rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    import gradio as gr

    def run(file, text):
        source = Path(file).read_text() if file else (text or "")
        if not source.strip():
            raise gr.Error("Upload a FASTA or CSV file, or paste sequences.")
        try:
            ranked, rejected = rank_variants(source)
        except RuntimeError as e:
            raise gr.Error(str(e))
        summary = f"**{len(ranked)}** variants ranked" + (f", **{len(rejected)}** rejected" if len(rejected) else "") + "."
        download = None
        if len(ranked):
            download = str(Path(tempfile.mkdtemp()) / "ranked_variants.csv")
            ranked.to_csv(download, index=False)
        return summary, ranked.head(500), download, rejected

    with gr.Blocks(title=TITLE) as page:
        gr.Markdown(f"# {TITLE}\n{ABOUT}\n\n{TESTED}")
        with gr.Row():
            file = gr.File(label="FASTA, or CSV with a 'sequence' column", type="filepath")
            text = gr.Textbox(label="Or paste sequences (FASTA, or one per line)", lines=6)
        button = gr.Button("Rank variants", variant="primary")
        summary = gr.Markdown()
        table = gr.Dataframe(label="Ranked variants (higher is better)", wrap=False)
        download = gr.File(label="Download the full ranking (CSV)")
        rejected = gr.Dataframe(label="Rejected variants")
        button.click(run, [file, text], [summary, table, download, rejected])
    page.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
