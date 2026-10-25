"""Task 5: prediction files, official metrics, component accuracy, attention map,
qualitative samples and result tables.

    python evaluate_model.py dev            # everything on dev (greedy + beam), picks final decoding
    python evaluate_model.py test           # ONE run on test with the decoding chosen on dev
    python evaluate_model.py report         # rebuild results/tables.md from results/metrics.json

Individual steps:
    python evaluate_model.py predict --split dev --decoding greedy
    python evaluate_model.py official --split dev --decoding greedy
    python evaluate_model.py components --decoding beam
    python evaluate_model.py attention --index 0
    python evaluate_model.py samples --decoding beam
    python evaluate_model.py curves
"""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

try:
    from . import common
    from .common import BOS_ID, EOS_ID, FIGURES, METRICS, RESULTS
    from .dataset import SQLDataset, collate
    from .decode import (
        ColumnConstraint,
        beam_search,
        greedy_decode,
        parse_to_wikisql,
        to_readable_sql,
    )
except ImportError:  # pragma: no cover - allows running as a script from the repo root
    import common
    from common import BOS_ID, EOS_ID, FIGURES, METRICS, RESULTS
    from dataset import SQLDataset, collate
    from decode import (
        ColumnConstraint,
        beam_search,
        greedy_decode,
        parse_to_wikisql,
        to_readable_sql,
    )

DECODINGS = ("greedy", "beam")


def pred_path(split, decoding):
    return RESULTS / f"{split}_{decoding}.jsonl"


def raw_path(split, decoding):
    return RESULTS / f"{split}_{decoding}_raw.txt"


# ------------------------------------------------------------------ predict
def predict(
    split: str, decoding: str, ckpt=common.BEST_CKPT, batch_size=128, limit=None
):
    """Decode every example of `split` in dev.jsonl/test.jsonl order and write one JSON
    line per example: {"query": {...}} or {"error": "parse"}. Never skips a line."""
    device = common.get_device()
    model, _ = common.load_model(ckpt, device)
    sp = common.load_sp()
    constraint = ColumnConstraint(sp, sp.get_piece_size())
    examples, tables = common.load_split(split)
    n_cols_all = [len(tables[e["table_id"]]["header"]) for e in examples]
    ds = SQLDataset(
        str(common.pairs_path(split)), sp, train=False
    )  # keeps every row, in order
    assert len(ds) == len(examples), "prediction file must align with the gold file"
    n = len(ds) if limit is None else min(limit, len(ds))

    outputs: list[list[int]] = []
    if decoding == "greedy":
        for start in tqdm(range(0, n, batch_size), desc=f"{split} greedy"):
            idx = list(range(start, min(start + batch_size, n)))
            src, _ = collate([ds[i] for i in idx])
            n_cols = torch.tensor([n_cols_all[i] for i in idx])
            outputs += greedy_decode(model, src.to(device), n_cols, constraint)
    else:
        for i in tqdm(range(n), desc=f"{split} beam(4)"):
            src = torch.tensor([ds[i][0]], device=device)
            outputs.append(
                beam_search(
                    model, src, torch.tensor([n_cols_all[i]]), constraint, beam_size=4
                )
            )

    rows, raws = [], []
    for i, ids in enumerate(outputs):
        text = sp.decode(ids)
        q = parse_to_wikisql(text, n_cols_all[i], examples[i]["question"])
        rows.append({"query": q} if q is not None else {"error": "parse"})
        raws.append(text.replace("\n", " "))
    common.write_jsonl(pred_path(split, decoding), rows)
    raw_path(split, decoding).write_text("\n".join(raws) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} lines -> {pred_path(split, decoding)}")
    return rows


def official(split: str, decoding: str):
    path = pred_path(split, decoding)
    m = common.run_official_evaluator(split, path)
    m["parse_failure_rate"] = common.parse_failure_rate(path)
    common.update_json(METRICS, f"{split}_{decoding}", m)
    print(
        f"[official] {split} {decoding}: LF={m['lf_accuracy']:.2%} EX={m['ex_accuracy']:.2%} "
        f"parse failures={m['parse_failure_rate']:.2%}"
    )
    return m


# --------------------------------------------------------------- components
def _norm_conds(conds):
    return {(int(c), int(o), str(v).lower()) for c, o, v in conds}


def components(decoding: str, split="dev"):
    """Task 5.2: % of examples with correct sel column, agg, and WHERE clause
    (same set of conditions, order ignored). Parse failures count as wrong."""
    examples, _ = common.load_split(split)
    preds = common.read_jsonl(pred_path(split, decoding))
    sel = agg = where = 0
    for ex, p in zip(examples, preds):
        if "query" not in p:
            continue
        g, q = ex["sql"], p["query"]
        sel += q["sel"] == g["sel"]
        agg += q["agg"] == g["agg"]
        where += _norm_conds(q["conds"]) == _norm_conds(g["conds"])
    n = len(examples)
    m = {"sel_accuracy": sel / n, "agg_accuracy": agg / n, "where_accuracy": where / n}
    common.update_json(METRICS, f"components_{split}_{decoding}", m)
    print(
        f"[components] {decoding}: sel={m['sel_accuracy']:.2%} agg={m['agg_accuracy']:.2%} "
        f"where={m['where_accuracy']:.2%}"
    )
    return m


# ---------------------------------------------------------------- attention
@torch.no_grad()
def _cross_attention(model, sp, src_ids, gen_ids, device):
    """Teacher-force the generated sequence once and return the last decoder
    layer's cross-attention averaged over heads: (len(gen)+1, S). Row t is the
    attention used while producing output token t (the last row produces </s>)."""
    src = torch.tensor([src_ids], device=device)
    ys = torch.tensor([[BOS_ID] + gen_ids], device=device)
    model(src, ys)
    return model.cross_attention_maps()[-1][0].mean(0).float().cpu()


def _column_spans(src_pieces):
    """source piece index -> column number: the '\u2581' right before <cK>, <cK> itself, and its name pieces."""
    owner, current = {}, None
    for i, p in enumerate(src_pieces):
        if p.startswith("<c") and p.endswith(">") and p[2:-1].isdigit():
            current = int(p[2:-1])
            if i > 0 and src_pieces[i - 1] == "\u2581":
                owner[i - 1] = current
        if current is not None and p != "</s>":
            owner[i] = current
    return owner


def attention(index: int = 0, n_stats: int = 1000, ckpt=common.BEST_CKPT):
    """Task 5.3: heat-map for one dev example + alignment rate of generated <cK>
    tokens with source column K over the first `n_stats` dev examples."""
    device = common.get_device()
    model, _ = common.load_model(ckpt, device)
    sp = common.load_sp()
    constraint = ColumnConstraint(sp, sp.get_piece_size())
    examples, tables = common.load_split("dev")
    ds = SQLDataset(str(common.pairs_path("dev")), sp, train=False)
    col_piece = {sp.piece_to_id(f"<c{k}>"): k for k in range(common.MAX_COLS)}

    def run(i):
        src_ids = ds[i][0]
        n_cols = torch.tensor([len(tables[examples[i]["table_id"]]["header"])])
        gen = greedy_decode(
            model, torch.tensor([src_ids], device=device), n_cols, constraint
        )[0]
        return src_ids, gen, _cross_attention(model, sp, src_ids, gen, device)

    # aggregate check
    hits = total = 0
    for i in tqdm(range(min(n_stats, len(ds))), desc="attention alignment"):
        src_ids, gen, att = run(i)
        owner = _column_spans(
            [sp.id_to_piece(t) if t != EOS_ID else "</s>" for t in src_ids]
        )
        for t, tok in enumerate(gen):
            if tok in col_piece:
                total += 1
                hits += owner.get(int(att[t].argmax()), -1) == col_piece[tok]
    rate = hits / max(total, 1)
    common.update_json(
        METRICS,
        "attention_alignment",
        {
            "examples": min(n_stats, len(ds)),
            "column_tokens": total,
            "aligned_rate": rate,
        },
    )
    print(
        f"[attention] generated <cK> whose top-attended source piece belongs to column K: "
        f"{hits}/{total} = {rate:.2%}"
    )

    # figure for one example
    src_ids, gen, att = run(index)
    src_lab = [sp.id_to_piece(t) if t != EOS_ID else "</s>" for t in src_ids]
    out_lab = [sp.id_to_piece(t) for t in gen] + ["</s>"]
    fig, ax = plt.subplots(
        figsize=(max(8, 0.32 * len(src_lab)), max(3, 0.38 * len(out_lab)))
    )
    ax.imshow(att.numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(src_lab)), src_lab, rotation=90, fontsize=8)
    ax.set_yticks(range(len(out_lab)), out_lab, fontsize=8)
    for t, tok in enumerate(gen):  # outline the matching source column token
        if tok in col_piece and sp.id_to_piece(tok) in src_lab:
            j = src_lab.index(sp.id_to_piece(tok))
            ax.add_patch(
                plt.Rectangle((j - 0.5, t - 0.5), 1, 1, fill=False, ec="red", lw=1.5)
            )
    ax.set_xlabel("source tokens")
    ax.set_ylabel("generated tokens")
    ax.set_title(
        f"Decoder cross-attention, last layer, mean over heads (dev #{index})\n"
        f"Q: {examples[index]['question']}",
        fontsize=9,
    )
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "cross_attention.png", dpi=200)
    print(f"saved {FIGURES / 'cross_attention.png'}")


# ------------------------------------------------------------------ samples
def _engine(split):
    """WikiSQL's own engine, wrapped so a query that cannot run (e.g. a text value on a
    numeric column) returns an error marker instead of crashing, the same way
    evaluate.py scores it: as wrong."""
    sys.path.insert(0, str(common.WIKISQL))
    from lib.dbengine import DBEngine
    from lib.query import Query
    engine = DBEngine(str(common.DATA_DIR / f"{split}.db"))
    run = engine.execute_query

    def safe_execute(*args, **kwargs):
        try:
            return run(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e!r}"

    engine.execute_query = safe_execute
    return engine, Query


def failure_types(gold: dict, pred: dict | None) -> list[str]:
    if pred is None:
        return ["parse failure"]
    out = []
    if pred["sel"] != gold["sel"]:
        out.append("wrong column")
    if pred["agg"] != gold["agg"]:
        out.append("wrong aggregation")
    g = {int(c): (o, str(v).lower()) for c, o, v in gold["conds"]}
    p = {int(c): (o, str(v).lower()) for c, o, v in pred["conds"]}
    if len(p) < len(g):
        out.append("missing condition")
    elif len(p) > len(g):
        out.append("extra condition")
    elif set(p) != set(g):
        out.append("wrong condition column")
    for c in set(g) & set(p):
        if g[c][0] != p[c][0]:
            out.append("wrong operator")
        if g[c][1] != p[c][1]:
            out.append("wrong value")
    return out or ["logical form differs"]


def samples(decoding: str, split="dev", n_each=5, seed=0):
    """Task 4.6: five correct and five wrong examples with real column names.
    "Correct" = logical form AND execution match (labelled with WikiSQL's own engine)."""
    examples, tables = common.load_split(split)
    preds = common.read_jsonl(pred_path(split, decoding))
    raws = raw_path(split, decoding).read_text(encoding="utf-8").splitlines()
    engine, Query = _engine(split)
    order = list(range(min(len(examples), len(preds))))
    random.Random(seed).shuffle(order)

    correct, wrong, seen_types = [], [], set()
    for i in order:
        ex, p = examples[i], preds[i]
        q = p.get("query")
        gold_q = Query.from_dict(ex["sql"])
        lf = q is not None and Query.from_dict(q) == gold_q
        ex_ok = False
        if q is not None:
            try:
                ex_ok = engine.execute_query(ex["table_id"], Query.from_dict(q)) == \
                    engine.execute_query(ex["table_id"], gold_q)
            except (TypeError, ValueError):
                # Predicted value cannot be cast for a numeric column, so the query is invalid.
                ex_ok = False
        if lf and ex_ok and len(correct) < n_each:
            correct.append((i, []))
        elif not lf and len(wrong) < n_each:
            types = failure_types(ex["sql"], q)
            if types[0] not in seen_types or len(seen_types) >= 6:  # prefer variety
                seen_types.add(types[0])
                wrong.append(
                    (i, types + (["(still executes correctly)"] if ex_ok else []))
                )
        if len(correct) == n_each and len(wrong) == n_each:
            break

    lines = [
        f"# Qualitative samples ({split}, {decoding} decoding)",
        "",
        (
            "Correct = logical form and execution both match the gold query. "
            "Values are lower-cased by the preprocessing, so the generated SQL is lower-case."
        ),
        "",
    ]
    for title, group in (("Correct", correct), ("Wrong", wrong)):
        lines += [f"## {title}", ""]
        for k, (i, types) in enumerate(group, 1):
            ex = examples[i]
            header = tables[ex["table_id"]]["header"]
            q = preds[i].get("query")
            lines += [
                f"### {title} {k} ({split} #{i})",
                "",
                f"**Question:** {ex['question']}  ",
                f"**Columns:** {', '.join(header)}  ",
                f"**Gold:** `{to_readable_sql(ex['sql'], header)}`  ",
                f"**Ours:** `{to_readable_sql(q, header) if q else 'PARSE FAILURE: ' + raws[i]}`  ",
            ]
            if types:
                lines.append(f"**Failure:** {', '.join(types)}  ")
            lines.append("")
    (RESULTS / "samples.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {RESULTS / 'samples.md'}")


# ------------------------------------------------------------------- curves
def curves(log=RESULTS / "train_log.csv"):
    with open(log) as f:
        rows = list(csv.DictReader(f))
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, [float(r["train_loss"]) for r in rows], marker="o", label="train")
    ax.plot(ep, [float(r["dev_loss"]) for r in rows], marker="o", label="dev")
    best = min(rows, key=lambda r: float(r["dev_loss"]))
    ax.axvline(
        int(best["epoch"]),
        ls="--",
        c="grey",
        lw=1,
        label=f"best dev (epoch {best['epoch']})",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("label-smoothed cross-entropy per token")
    ax.set_title("Training and dev loss")
    ax.set_xticks(ep)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "loss_curves.png", dpi=200)
    print(f"saved {FIGURES / 'loss_curves.png'}")


# ------------------------------------------------------------------- report
def _pct(x):
    return "–" if x is None else f"{100 * x:.2f}"


def report():
    import json

    m = json.loads(METRICS.read_text()) if METRICS.exists() else {}
    L = [
        "# Results",
        "",
        "Generated by `python evaluate_model.py report` from `results/metrics.json`.",
        "",
    ]

    d = m.get("data")
    if d:
        L += ["## Table 1 – Data", "", "| | Train | Dev | Test |", "|---|---|---|---|"]
        L.append(
            "| Pairs | "
            + " | ".join(f"{d[s]['pairs']:,}" for s in common.SPLITS)
            + " |"
        )
        L.append(
            "| Mean / max source length (tokens) | "
            + " | ".join(
                f"{d[s]['src_mean']:.1f} / {d[s]['src_max']}" for s in common.SPLITS
            )
            + " |"
        )
        L.append(
            "| Mean / max target length (tokens) | "
            + " | ".join(
                f"{d[s]['tgt_mean']:.1f} / {d[s]['tgt_max']}" for s in common.SPLITS
            )
            + " |"
        )
        L.append(f"| Pairs dropped as too long | {d['train']['dropped']} | – | – |")
        L += [
            "",
            (
                "Lengths are SentencePiece tokens as fed to the model: source includes `</s>`, "
                "target includes `<s>` and `</s>`."
            ),
            "",
        ]

    t = m.get("training")
    if t:
        L += [
            "## Table 2 – Model and training",
            "",
            "| | |",
            "|---|---|",
            f"| Trainable parameters | {t['trainable_parameters']:,} |",
            f"| Epochs trained / best epoch | {t['epochs_trained']} / {t['best_epoch']} |",
            f"| Best dev loss | {t['best_dev_loss']:.4f} |",
            f"| Training time and GPU | {t['training_minutes']} min on {t['device']} |",
            "",
        ]

    L += [
        "## Table 3 – Official metrics",
        "",
        "| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |",
        "|---|---|---|---|---|",
    ]
    final = m.get("final_decoding")
    for key, split, dec in (
        ("dev_greedy", "Dev", "greedy"),
        ("dev_beam", "Dev", "beam (4)"),
        (f"test_{final}", "Test", (final or "–") + (" (4)" if final == "beam" else "")),
    ):
        r = m.get(key, {})
        L.append(
            f"| {split} | {dec} | {_pct(r.get('lf_accuracy'))} | {_pct(r.get('ex_accuracy'))} | "
            f"{_pct(r.get('parse_failure_rate'))} |"
        )
    if final:
        L += ["", f"Final decoding (highest dev execution accuracy): **{final}**."]
    L.append("")

    c = m.get(f"components_dev_{final}") if final else None
    if c:
        L += [
            f"## Table 4 – Component accuracy (dev, {final})",
            "",
            "| | |",
            "|---|---|",
            f"| sel column correct (%) | {_pct(c['sel_accuracy'])} |",
            f"| agg correct (%) | {_pct(c['agg_accuracy'])} |",
            f"| WHERE clause correct (%) | {_pct(c['where_accuracy'])} |",
            "",
        ]
    a = m.get("attention_alignment")
    if a:
        L += [
            "## Attention check",
            "",
            (
                f"Over the first {a['examples']} dev examples, {_pct(a['aligned_rate'])}% of the "
                f"{a['column_tokens']} generated `<cK>` tokens put their highest last-layer "
                "cross-attention (mean over heads) on a source piece of column K."
            ),
            "",
        ]
    (RESULTS / "tables.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


# --------------------------------------------------------------- pipelines
def run_dev():
    for dec in DECODINGS:
        predict("dev", dec)
        official("dev", dec)
        components(dec)
    import json

    m = json.loads(METRICS.read_text())
    final = max(DECODINGS, key=lambda d: m[f"dev_{d}"]["ex_accuracy"])
    common.update_json(METRICS, "final_decoding", final)
    print(f"final decoding chosen on dev: {final}")
    attention()
    samples(final)
    if (RESULTS / "train_log.csv").exists():
        curves()
    report()


def run_test(force=False):
    import json

    m = json.loads(METRICS.read_text())
    final = m.get("final_decoding")
    if final is None:
        sys.exit(
            "run `python evaluate_model.py dev` first: the decoding is chosen on dev"
        )
    if f"test_{final}" in m and not force:
        sys.exit(
            "test has already been evaluated once; use --force only to regenerate the same run"
        )
    predict("test", final)
    official("test", final)
    report()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dev")
    t = sub.add_parser("test")
    t.add_argument("--force", action="store_true")
    sub.add_parser("report")
    sub.add_parser("curves")
    for name in ("predict", "official"):
        p = sub.add_parser(name)
        p.add_argument("--split", choices=("dev", "test"), default="dev")
        p.add_argument("--decoding", choices=DECODINGS, default="greedy")
        if name == "predict":
            p.add_argument("--ckpt", default=str(common.BEST_CKPT))
            p.add_argument(
                "--limit",
                type=int,
                default=None,
                help="debug only: decode first N rows",
            )
    for name in ("components", "samples"):
        p = sub.add_parser(name)
        p.add_argument("--decoding", choices=DECODINGS, default="greedy")
    p = sub.add_parser("attention")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--n-stats", type=int, default=1000)
    a = ap.parse_args()

    if a.cmd == "dev":
        run_dev()
    elif a.cmd == "test":
        run_test(a.force)
    elif a.cmd == "report":
        report()
    elif a.cmd == "curves":
        curves()
    elif a.cmd == "predict":
        predict(a.split, a.decoding, a.ckpt, limit=a.limit)
    elif a.cmd == "official":
        official(a.split, a.decoding)
    elif a.cmd == "components":
        components(a.decoding)
    elif a.cmd == "samples":
        samples(a.decoding)
    elif a.cmd == "attention":
        attention(a.index, a.n_stats)


if __name__ == "__main__":
    main()
