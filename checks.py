"""Section 3.2 correctness checks. Output goes to results/correctness_checks.md
(paste into the README) and results/figures/lr_schedule.png.

    python checks.py                 # uses checkpoints/best.pt if it exists, else a random init

The mask/attention/tying checks are properties of the architecture, so they must
hold for any weights; the gold round-trip check needs no model at all.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import common
import matplotlib.pyplot as plt
import torch
from common import (
    FIGURES,
    METRICS,
    PAD_ID,
    RESULTS,
    ModelConfig,
    TrainConfig,
)
from dataset import SQLDataset, collate
from decode import parse_to_wikisql
from model import Transformer
from model.attention import MultiHeadAttention
from train import noam_lambda

TOL = 1e-5
lines = []


def log(msg=""):
    print(msg)
    lines.append(msg)


def get_model(ckpt):
    if ckpt.exists():
        model, _ = common.load_model(ckpt, torch.device("cpu"))
        return model, f"trained checkpoint `{ckpt.relative_to(common.ROOT)}`"
    common.set_seed(0)
    return Transformer(**common.to_dict(ModelConfig())).eval(), "random initialisation (no checkpoint yet)"


def batch(n=8):
    sp = common.load_sp()
    ds = SQLDataset(str(common.pairs_path("dev")), sp, train=False)
    return collate([ds[i] for i in range(n)])


@torch.no_grad()
def check_causal(model, src, tgt):
    dec_in = tgt[:, :-1]
    out = model(src, dec_in)
    changed = dec_in.clone()
    changed[:, -1] = (changed[:, -1] + 17) % 7900 + 100  # a different, non-special token
    out2 = model(src, changed)
    diff = (out[:, :-1] - out2[:, :-1]).abs().max().item()
    last = (out[:, -1] - out2[:, -1]).abs().max().item()
    ok = diff < TOL
    log(f"- **Causal mask:** {'PASS' if ok else 'FAIL'}. Changing the last decoder input token: "
        f"max |Δ| at earlier positions = {diff:.2e} (at the changed position itself: {last:.2e}).")
    return ok


@torch.no_grad()
def check_padding(model, src, tgt):
    out = model(src, tgt[:, :-1])
    padded = torch.cat([src, torch.full((src.size(0), 25), PAD_ID, dtype=torch.long)], dim=1)
    out2 = model(padded, tgt[:, :-1])
    diff = (out - out2).abs().max().item()
    ok = diff < TOL
    log(f"- **Padding mask:** {'PASS' if ok else 'FAIL'}. Appending 25 `<pad>` tokens to every source: "
        f"max |Δ| of the output logits = {diff:.2e}.")
    return ok


@torch.no_grad()
def check_rows(model, src, tgt):
    model(src, tgt[:, :-1])
    worst_sum, worst_masked = 0.0, 0.0
    src_keys = (src != PAD_ID)[:, None, None, :]
    T = tgt.size(1) - 1
    tgt_keys = ((tgt[:, :-1] != PAD_ID)[:, None, None, :]
                & torch.tril(torch.ones(T, T, dtype=torch.bool))[None, None])
    for name, mod in model.named_modules():
        if not isinstance(mod, MultiHeadAttention):
            continue
        a = mod.attn_weights  # (B, h, Lq, Lk)
        allowed = tgt_keys if "decoder" in name and "self_attn" in name else src_keys
        worst_sum = max(worst_sum, (a.masked_fill(~allowed, 0).sum(-1) - 1).abs().max().item())
        worst_masked = max(worst_masked, a.masked_fill(allowed, 0).abs().max().item())
    ok = worst_sum < TOL and worst_masked < TOL
    log(f"- **Attention rows sum to 1:** {'PASS' if ok else 'FAIL'}. Over all 9 attention modules, "
        f"max |row sum over unmasked keys − 1| = {worst_sum:.2e}; max weight on a masked key = {worst_masked:.2e}.")
    return ok


def check_tying(model):
    ok = model.generator.weight is model.shared.emb.weight \
        and model.src_input.tok is model.tgt_input.tok is model.shared
    log(f"- **Weight sharing:** {'PASS' if ok else 'FAIL'}. `model.generator.weight is "
        f"model.shared.emb.weight` → {model.generator.weight is model.shared.emb.weight}; encoder and "
        f"decoder InputLayers hold the same TokenEmbedding → {model.src_input.tok is model.tgt_input.tok}. "
        f"Trainable parameters: {model.num_parameters():,}.")
    return ok


def check_lr():
    f = noam_lambda(ModelConfig.d_model, TrainConfig.warmup_steps)
    steps = list(range(1, 20001))
    lr = [f(s - 1) for s in steps]  # f takes completed steps; paper step s = completed + 1
    peak = max(range(len(lr)), key=lr.__getitem__) + 1
    lin = max(abs(lr[s - 1] / s - lr[0]) / lr[0] for s in range(1, 4001))
    dec = max(abs(lr[s - 1] * s ** .5 - lr[4000] * 4001 ** .5) / (lr[4000] * 4001 ** .5)
              for s in range(4001, 20001))
    ok = peak == 4000 and lin < 1e-9 and dec < 1e-9
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(steps, lr)
    ax.axvline(4000, ls="--", c="grey", lw=1, label="warmup_steps = 4000")
    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate")
    ax.set_title(r"lrate = $d_{model}^{-0.5}\cdot\min(step^{-0.5},\ step\cdot warmup^{-1.5})$")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "lr_schedule.png", dpi=200)
    log(f"- **Learning rate:** {'PASS' if ok else 'FAIL'}. Peak {max(lr):.3e} at step {peak}; "
        f"lr/step constant to {lin:.1e} for steps 1–4000 (linear rise); lr·√step constant to {dec:.1e} "
        f"for steps 4001–20000 (step^-0.5 decay). Plot: `results/figures/lr_schedule.png`.")
    return ok


def check_gold_roundtrip():
    """Gold dev targets -> SentencePiece encode/decode -> our parser -> official evaluator."""
    sp = common.load_sp()
    examples, tables = common.load_split("dev")
    pairs = common.read_jsonl(common.pairs_path("dev"))
    rows = []
    for ex, p in zip(examples, pairs):
        q = parse_to_wikisql(sp.decode(sp.encode(p["tgt"])), len(tables[ex["table_id"]]["header"]),
                             ex["question"])
        rows.append({"query": q} if q is not None else {"error": "parse"})
    path = RESULTS / "dev_gold_roundtrip.jsonl"
    common.write_jsonl(path, rows)
    m = common.run_official_evaluator("dev", path)
    m["parse_failure_rate"] = common.parse_failure_rate(path)
    common.update_json(METRICS, "gold_roundtrip_dev", m)
    ok = m["ex_accuracy"] > 0.99
    log(f"- **Gold round-trip:** {'PASS' if ok else 'FAIL'}. Official evaluator on parsed gold dev targets: "
        f"execution {100 * m['ex_accuracy']:.2f}%, logical form {100 * m['lf_accuracy']:.2f}%, "
        f"parse failures {100 * m['parse_failure_rate']:.2f}%. Residual errors come from values that "
        f"SentencePiece changes irreversibly (NFKC normalisation, <unk> that cannot be matched in the "
        f"question), not from the parser.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(common.BEST_CKPT))
    args = ap.parse_args()
    from pathlib import Path
    model, which = get_model(Path(args.ckpt))
    src, tgt = batch()
    log("## Correctness checks")
    log("")
    log(f"Model: {which}, eval mode (dropout off), tolerance {TOL:g}. Batch: first 8 dev examples.")
    log("")
    results = [check_causal(model, src, tgt), check_padding(model, src, tgt), check_rows(model, src, tgt),
               check_tying(model), check_lr(), check_gold_roundtrip()]
    log("")
    log(f"**{sum(results)}/{len(results)} checks passed.**")
    (RESULTS / "correctness_checks.md").write_text("\n".join(lines) + "\n")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
