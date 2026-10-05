# python app/app.py                 # http://127.0.0.1:7860
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common
import gradio as gr
from decode import SQLGenerator

EXAMPLES = [
    ["What is Terrence Ross' nationality?",
     "Player, No., Nationality, Position, Years in Toronto, School/Club Team", "beam"],
    ["How many players played for Butler CC (KS)?",
     "Player, No., Nationality, Position, Years in Toronto, School/Club Team", "beam"],
    ["Which country has the highest gold count when silver is more than 2?",
     "Rank, Nation, Gold, Silver, Bronze, Total", "beam"],
    ["What is the average attendance for games against the Lions?",
     "Week, Date, Opponent, Result, Attendance", "greedy"],
]

_generator = None


def generator():
    global _generator
    if _generator is None:
        _generator = SQLGenerator(common.BEST_CKPT)
    return _generator


def translate(question: str, columns: str, method: str):
    question = (question or "").strip()
    header = [c.strip() for c in (columns or "").split(",") if c.strip()]
    if not question:
        raise gr.Error("Type a question.")
    if not header:
        raise gr.Error("Give at least one column name (comma-separated).")
    if len(header) > common.MAX_COLS:
        raise gr.Error(f"At most {common.MAX_COLS} columns are supported.")
    out = generator()(question, header, method)
    cols = "\n".join(f"<c{i}>  {h}" for i, h in enumerate(header))
    if out["sql"] is None:
        return "-- the model's output could not be parsed into a WikiSQL query", out["raw"], cols
    return out["sql"], out["raw"], cols


def build():
    with gr.Blocks(title="Text-to-SQL Transformer") as demo:
        gr.Markdown(
            "## Text-to-SQL with a from-scratch Transformer\n"
            "Encoder–decoder Transformer (Vaswani et al., 2017) trained from random initialisation on "
            "WikiSQL. Describe a table by its column names, ask a question, get the SQL.")
        with gr.Row():
            with gr.Column():
                q = gr.Textbox(label="Question", placeholder="What is Terrence Ross' nationality?")
                c = gr.Textbox(label="Column names (comma-separated)",
                               placeholder="Player, No., Nationality, Position")
                m = gr.Radio(["beam", "greedy"], value="beam", label="Decoding")
                btn = gr.Button("Generate SQL", variant="primary")
            with gr.Column():
                sql = gr.Code(label="Generated SQL", language="sql")
                raw = gr.Textbox(label="Raw model output (column tokens)")
                cols = gr.Textbox(label="Column token mapping", lines=4)
        btn.click(translate, [q, c, m], [sql, raw, cols])
        q.submit(translate, [q, c, m], [sql, raw, cols])
        gr.Examples(EXAMPLES, [q, c, m])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    generator()  # load the model before the first request [first] 
    build().launch(share=args.share, server_port=args.port)
