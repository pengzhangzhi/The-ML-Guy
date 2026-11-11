"""The prediction page. Generic: everything project-specific comes from task.toml and predict.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from mlguy import seqio
from mlguy.judge import SubmissionError, check_predictions
from mlguy.project import load_task, run_predict

MAX_ROWS_SHOWN = 500


def predict_table(project_dir: str | Path, ids: list[str], seqs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ranked predictions, rejected rows) for the given sequences."""
    task = load_task(project_dir)
    spec = task.get("input", {})
    seq_col = task["data"]["sequence_column"]

    reasons = [seqio.check_sequence(s, spec) for s in seqs]
    rejected = pd.DataFrame(
        [{"id": i, "reason": r} for i, r in zip(ids, reasons) if r], columns=["id", "reason"]
    )
    keep = [(i, s) for i, s, r in zip(ids, seqs, reasons) if r is None]
    columns = ["rank", "id", "prediction", "mutations", "sequence"]
    if not keep:
        return pd.DataFrame(columns=columns), rejected

    with tempfile.TemporaryDirectory() as tmp:
        input_csv, output_csv = Path(tmp) / "inputs.csv", Path(tmp) / "predictions.csv"
        pd.DataFrame({seq_col: [s for _, s in keep]}).to_csv(input_csv, index=False)
        run_predict(task["_project_dir"], input_csv, output_csv)
        values = check_predictions(pd.read_csv(output_csv), len(keep))

    reference = spec.get("reference")
    table = pd.DataFrame(
        {
            "id": [i for i, _ in keep],
            "prediction": values,
            "mutations": [seqio.describe_mutations(s, reference) for _, s in keep],
            "sequence": [s for _, s in keep],
        }
    )
    best_first = task["task"].get("higher_is_better", True)
    table = table.sort_values("prediction", ascending=not best_first, kind="stable").reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table[columns], rejected


def make_app(project_dir: str | Path):
    import gradio as gr
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task = load_task(project_dir)
    page = task.get("page", {})
    title = page.get("title", task["name"])
    output_name = task["task"]["output"].replace("_", " ")
    direction = "higher is better" if task["task"].get("higher_is_better", True) else "lower is better"

    def run(file, text):
        try:
            if file is not None:
                ids, seqs = seqio.read_sequences(file, task["data"]["sequence_column"])
            elif text and text.strip():
                ids, seqs = seqio.parse_text(text)
            else:
                raise gr.Error("Upload a CSV or FASTA file, or paste sequences.")
            if not seqs:
                raise gr.Error("No sequences found in the input.")
            table, rejected = predict_table(project_dir, ids, seqs)
        except (SubmissionError, ValueError) as e:
            raise gr.Error(str(e))

        summary = f"**{len(table)}** sequences predicted"
        if len(rejected):
            summary += f", **{len(rejected)}** rejected (see below)"
        if len(table) > MAX_ROWS_SHOWN:
            summary += f". Showing the top {MAX_ROWS_SHOWN}; the download has all of them"
        summary += "."

        fig = None
        download = None
        if len(table):
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.hist(table["prediction"], bins=min(40, max(5, len(table) // 5)), color="#4c72b0")
            ax.set_xlabel(f"predicted {output_name}")
            ax.set_ylabel("number of sequences")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            out = Path(tempfile.mkdtemp()) / "predictions.csv"
            table.to_csv(out, index=False)
            download = str(out)
        return summary, table.head(MAX_ROWS_SHOWN), fig, download, rejected

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        if page.get("description"):
            gr.Markdown(page["description"])
        if page.get("tested"):
            gr.Markdown(f"**How it was tested.** {page['tested']}")
        with gr.Row():
            file = gr.File(label="Upload CSV (with a 'sequence' column) or FASTA", type="filepath")
            text = gr.Textbox(label="Or paste sequences (one per line, or FASTA)", lines=6)
        button = gr.Button("Predict", variant="primary")
        summary = gr.Markdown()
        table = gr.Dataframe(label=f"Ranked predictions ({direction})", wrap=False)
        with gr.Row():
            plot = gr.Plot(label="Distribution of predictions")
            download = gr.File(label="Download all predictions (CSV)")
        rejected = gr.Dataframe(label="Rejected rows")
        button.click(run, [file, text], [summary, table, plot, download, rejected], api_name="predict")
    return demo


def launch(project_dir: str | Path, port: int | None = None, share: bool = False) -> None:
    make_app(project_dir).launch(server_port=port, share=share)
