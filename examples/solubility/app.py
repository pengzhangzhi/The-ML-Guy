"""The page for this project: will a protein be soluble when expressed in E. coli?

    python app.py [--port N] [--share]

Written for this problem, not supplied by the framework. The user has a list of proteins and
wants to know which to try first, so the page gives a call and the probability behind it,
most likely soluble first. Predictions come from infer.py, the same code path the scores in
REPORT.md were measured on.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FINAL = HERE / "final"   # the submission that was scored, put there by `mlguy finish`
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

TITLE = "Protein solubility predictor"
ABOUT = ("Predicts whether a protein will be soluble when expressed in *E. coli*, from its "
         "amino-acid sequence alone.")
TESTED = ("**How it was tested.** On 2,001 proteins that are not close relatives of anything it was "
          "trained on, it made the right call for about two in three. Treat it as a way to order "
          "candidates, not as a verdict on any one protein.")


def read_proteins(text: str) -> pd.DataFrame:
    """FASTA, a CSV with a `sequence` column, or one sequence per line."""
    text = text.strip()
    if text.startswith(">"):
        names, seqs = [], []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(">"):
                names.append(line[1:].split()[0] if len(line) > 1 else f"protein{len(names) + 1}")
                seqs.append("")
            elif line and names:
                seqs[-1] += line
        return pd.DataFrame({"name": names, "sequence": seqs})
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and "sequence" in [c.strip() for c in lines[0].split(",")]:
        df = pd.read_csv(StringIO(text))
        names = df["id"].astype(str) if "id" in df.columns else [f"protein{i + 1}" for i in range(len(df))]
        return pd.DataFrame({"name": list(names), "sequence": df["sequence"].astype(str)})
    return pd.DataFrame({"name": [f"protein{i + 1}" for i in range(len(lines))], "sequence": lines})


def problem_with(seq: str) -> str | None:
    if not seq:
        return "empty sequence"
    bad = sorted(set(seq) - AMINO_ACIDS)
    if bad:
        return f"contains characters the model was not trained on: {' '.join(bad)}"
    return None


def call_proteins(text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (calls ordered by probability of being soluble, rejected proteins)."""
    proteins = read_proteins(text)
    proteins["sequence"] = proteins["sequence"].str.strip().str.upper()
    proteins["reason"] = proteins["sequence"].map(problem_with)
    rejected = proteins.loc[proteins["reason"].notna(), ["name", "reason"]].reset_index(drop=True)
    kept = proteins[proteins["reason"].isna()].reset_index(drop=True)
    columns = ["rank", "name", "call", "probability soluble", "length"]
    if kept.empty:
        return pd.DataFrame(columns=columns), rejected

    with tempfile.TemporaryDirectory() as tmp:
        inputs, outputs = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
        kept.assign(id=kept.index)[["id", "sequence"]].to_csv(inputs, index=False)
        proc = subprocess.run([sys.executable, "infer.py", "--model", "model", "--input", str(inputs),
                               "--output", str(outputs)], cwd=FINAL, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"the model failed to run:\n{proc.stderr.strip()[-1000:]}")
        out = pd.read_csv(outputs).set_index("id").loc[kept.index]

    kept["call"] = out["prediction"].map({1: "soluble", 0: "insoluble"}).to_numpy()
    kept["probability soluble"] = out["score"].round(3).to_numpy()
    kept["length"] = kept["sequence"].str.len()
    kept = kept.sort_values("probability soluble", ascending=False, kind="stable").reset_index(drop=True)
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
            calls, rejected = call_proteins(source)
        except RuntimeError as e:
            raise gr.Error(str(e))
        summary = f"**{len(calls)}** proteins scored" + (f", **{len(rejected)}** rejected" if len(rejected) else "") + "."
        download = None
        if len(calls):
            download = str(Path(tempfile.mkdtemp()) / "solubility_calls.csv")
            calls.to_csv(download, index=False)
        return summary, calls.head(500), download, rejected

    with gr.Blocks(title=TITLE) as page:
        gr.Markdown(f"# {TITLE}\n{ABOUT}\n\n{TESTED}")
        with gr.Row():
            file = gr.File(label="FASTA, or CSV with a 'sequence' column", type="filepath")
            text = gr.Textbox(label="Or paste sequences (FASTA, or one per line)", lines=6)
        button = gr.Button("Predict", variant="primary")
        summary = gr.Markdown()
        table = gr.Dataframe(label="Most likely soluble first", wrap=False)
        download = gr.File(label="Download all calls (CSV)")
        rejected = gr.Dataframe(label="Rejected proteins")
        button.click(run, [file, text], [summary, table, download, rejected])
    page.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
