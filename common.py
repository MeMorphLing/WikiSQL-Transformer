"""Shared paths, configuration and helpers.

Every top-level script imports this module first. It puts ``starter/`` on the
import path (the starter files import each other by bare module name, so the
folder must stay flat and unchanged) and points the starter's relative data
directory at the WikiSQL checkout in the repository root.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STARTER = ROOT / "starter"
WIKISQL = ROOT / "WikiSQL"
DATA_DIR = WIKISQL / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
CKPT_DIR = ROOT / "checkpoints"
BEST_CKPT = CKPT_DIR / "best.pt"
SP_MODEL = STARTER / "sql_sp.model"
METRICS = RESULTS / "metrics.json"

if str(STARTER) not in sys.path:
    sys.path.insert(0, str(STARTER))

import data_prep  # noqa: E402  (starter module)
import torch  # noqa: E402

data_prep.DATA_DIR = DATA_DIR  # absolute path; the starter file itself is untouched

from data_prep import AGG_OPS, COND_OPS, MAX_COLS  # noqa: E402,F401
from tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID  # noqa: E402,F401

SPLITS = ("train", "dev", "test")


def pairs_path(split: str) -> Path:
    return STARTER / f"{split}_pairs.jsonl"


@dataclass
class ModelConfig:
    """Task 2 configuration (fixed by the assignment)."""
    vocab_size: int = 8000
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 512
    pad_id: int = PAD_ID


@dataclass
class TrainConfig:
    """Task 3 configuration (fixed by the assignment)."""
    epochs: int = 20
    batch_size: int = 64
    label_smoothing: float = 0.1
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-9
    warmup_steps: int = 4000
    seed: int = 42


def to_dict(cfg) -> dict:
    return asdict(cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sp():
    import sentencepiece as spm
    return spm.SentencePieceProcessor(model_file=str(SP_MODEL))


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_split(split: str):
    """(examples, tables) from the official WikiSQL files, in file order."""
    return data_prep.load_split(split)


def update_json(path, key: str, value) -> None:
    """Merge one entry into a JSON results file (creates it if missing)."""
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_model(path=BEST_CKPT, device=None):
    """Rebuild the Transformer from a checkpoint written by train.py."""
    from model.transformer import Transformer
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Transformer(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def run_official_evaluator(split: str, pred_path) -> dict:
    """Run WikiSQL's own evaluate.py (unmodified) and return its two numbers.
    Equivalent to: cd WikiSQL && python evaluate.py data/<split>.jsonl data/<split>.db <pred>"""
    import subprocess
    pred_path = Path(pred_path).resolve()
    cmd = [sys.executable, "evaluate.py", f"data/{split}.jsonl", f"data/{split}.db", str(pred_path)]
    res = subprocess.run(cmd, cwd=WIKISQL, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"official evaluator failed:\n{res.stderr[-2000:]}")
    out = json.loads(res.stdout[res.stdout.index("{"):])
    return {"lf_accuracy": out["lf_accuracy"], "ex_accuracy": out["ex_accuracy"]}


def parse_failure_rate(pred_path) -> float:
    rows = read_jsonl(pred_path)
    return sum("error" in r for r in rows) / len(rows)
