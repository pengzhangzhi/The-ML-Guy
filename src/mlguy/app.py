"""The prediction page.

Generic: it knows only that a project takes a table of input columns and returns one
prediction per row. Everything domain-specific comes from task.toml and predict.py.

A project whose input is a biological sequence can switch on two conveniences in task.toml:

    [page]
    fasta_column = "sequence"        # also accept FASTA uploads and pasted sequences
    mutations_from = "sequence"      # name mutations against [input].reference
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from mlguy.judge import SubmissionError, check_predictions
from mlguy.project import input_columns, load_task, run_predict

MAX_ROWS_SHOWN = 500


def read_table(path: str | Path, task: dict) -> pd.DataFrame:
    """Read an upload into a table holding this task's input columns."""
    path = Path(path)
    columns = input_columns(task)
    fasta_column = task.get("page", {}).get("fasta_column")
    if fasta_column:
        from mlguy.sequence import FASTA_SUFFIXES, parse_fasta

        text = path.read_text()
        if path.suffix.lower() in FASTA_SUFFIXES or text.lstrip().startswith(">"):
            ids, seqs = parse_fasta(text)
            return pd.DataFrame({"id": ids, fasta_column: seqs})

    df = pd.read_csv(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"the file needs {'a column' if len(missing) == 1 else 'columns'} "
                         f"called {', '.join(repr(c) for c in missing)}; it has {list(df.columns)}")
    return df


def parse_pasted(text: str, task: dict) -> pd.DataFrame:
    """Pasted text. One input column only: one value per line, or FASTA when enabled."""
    columns = input_columns(task)
    fasta_column = task.get("page", {}).get("fasta_column")
    if fasta_column:
        from mlguy.sequence import parse_text

        ids, seqs = parse_text(text)
        return pd.DataFrame({"id": ids, fasta_column: seqs})
    if len(columns) != 1:
        raise ValueError(f"this model takes {len(columns)} columns ({', '.join(columns)}), "
                         "so please upload a CSV instead of pasting")
    values = [line.strip() for line in text.splitlines() if line.strip()]
    return pd.DataFrame({columns[0]: values})


def predict_table(project_dir: str | Path, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ranked predictions, rejected rows) for a table of inputs."""
    task = load_task(project_dir)
    columns = input_columns(task)
    page = task.get("page", {})

    ids = table["id"].astype(str).tolist() if "id" in table.columns else [f"row{i + 1}" for i in range(len(table))]
    inputs = table[columns]

    reasons: list[str | None] = [None] * len(table)
    empty = inputs.isna().any(axis=1) | (inputs.astype(str).apply(lambda c: c.str.strip() == "").any(axis=1))
    for i, is_empty in enumerate(empty.tolist()):
        if is_empty:
            reasons[i] = "a required value is missing"
    if page.get("fasta_column"):
        from mlguy.sequence import check_sequence

        spec = task.get("input", {})
        for i, value in enumerate(inputs[page["fasta_column"]].astype(str).tolist()):
            if reasons[i] is None:
                reasons[i] = check_sequence(value.strip().upper(), spec)

    rejected = pd.DataFrame([{"id": i, "reason": r} for i, r in zip(ids, reasons) if r],
                            columns=["id", "reason"])
    keep = [i for i, r in enumerate(reasons) if r is None]
    out_columns = ["rank", "id", "prediction", *columns]
    if not keep:
        return pd.DataFrame(columns=out_columns), rejected

    kept = inputs.iloc[keep].reset_index(drop=True)
    with tempfile.TemporaryDirectory() as tmp:
        input_csv, output_csv = Path(tmp) / "inputs.csv", Path(tmp) / "predictions.csv"
        kept.to_csv(input_csv, index=False)
        run_predict(task["_project_dir"], input_csv, output_csv)
        values, scores = check_predictions(pd.read_csv(output_csv), len(kept))

    result = kept.copy()
    result.insert(0, "id", [ids[i] for i in keep])
    result.insert(1, "prediction", values)
    sort_by = "prediction"
    if scores is not None:
        result.insert(2, "score", scores)
        out_columns.insert(3, "score")
        sort_by = "score"

    mutations_from = page.get("mutations_from")
    if mutations_from:
        from mlguy.sequence import describe_mutations

        reference = task.get("input", {}).get("reference")
        result["mutations"] = [describe_mutations(str(s), reference) for s in result[mutations_from]]
        out_columns.insert(out_columns.index(sort_by) + 1, "mutations")

    best_first = task["task"].get("higher_is_better", True)
    result = result.sort_values(sort_by, ascending=not best_first, kind="stable").reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result[out_columns], rejected


def make_app(project_dir: str | Path):
    import gradio as gr
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task = load_task(project_dir)
    page = task.get("page", {})
    columns = input_columns(task)
    title = page.get("title", task["name"])
    output_name = task["task"]["output"].replace("_", " ")
    direction = "higher is better" if task["task"].get("higher_is_better", True) else "lower is better"
    accepts_fasta = bool(page.get("fasta_column"))
    item = page.get("row_name", "row")

    upload_label = "Upload a CSV with " + (
        f"a '{columns[0]}' column" if len(columns) == 1 else f"columns: {', '.join(columns)}"
    ) + (", or a FASTA file" if accepts_fasta else "")
    paste_label = ("Or paste sequences (one per line, or FASTA)" if accepts_fasta
                   else f"Or paste one {columns[0]} per line" if len(columns) == 1
                   else "Paste is unavailable for multi-column input; please upload a CSV")

    def run(file, text):
        try:
            if file is not None:
                table = read_table(file, task)
            elif text and text.strip():
                table = parse_pasted(text, task)
            else:
                raise gr.Error(f"Upload a CSV{' or FASTA' if accepts_fasta else ''} file, or paste your data.")
            if not len(table):
                raise gr.Error("No rows found in the input.")
            result, rejected = predict_table(project_dir, table)
        except (SubmissionError, ValueError, KeyError) as e:
            raise gr.Error(str(e))

        summary = f"**{len(result)}** {item}s predicted"
        if len(rejected):
            summary += f", **{len(rejected)}** rejected (see below)"
        if len(result) > MAX_ROWS_SHOWN:
            summary += f". Showing the top {MAX_ROWS_SHOWN}; the download has all of them"
        summary += "."

        fig, download = None, None
        if len(result):
            column = "score" if "score" in result.columns else "prediction"
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.hist(result[column], bins=min(40, max(5, len(result) // 5)), color="#4c72b0")
            ax.set_xlabel(f"predicted {output_name}" if column == "prediction"
                          else f"probability of {output_name}")
            ax.set_ylabel(f"number of {item}s")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            out = Path(tempfile.mkdtemp()) / "predictions.csv"
            result.to_csv(out, index=False)
            download = str(out)
        return summary, result.head(MAX_ROWS_SHOWN), fig, download, rejected

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        if page.get("description"):
            gr.Markdown(page["description"])
        if page.get("tested"):
            gr.Markdown(f"**How it was tested.** {page['tested']}")
        with gr.Row():
            file = gr.File(label=upload_label, type="filepath")
            text = gr.Textbox(label=paste_label, lines=6, interactive=len(columns) == 1 or accepts_fasta)
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
