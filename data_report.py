"""Task 1.3 and 1.4: data statistics (Table 1), starter shapes, positional-encoding heat-map.

    python data_report.py
"""
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import common
import matplotlib.pyplot as plt
from common import BOS_ID, EOS_ID, FIGURES, METRICS, RESULTS
from embeddings import PositionalEncoding

MAX_SRC, MAX_TGT = 160, 64  # SQLDataset defaults


def data_stats():
    sp = common.load_sp()
    stats = {}
    for split in common.SPLITS:
        pairs = common.read_jsonl(common.pairs_path(split))
        src = [len(sp.encode(p["src"])) + 1 for p in pairs]          # + </s>
        tgt = [len(sp.encode(p["tgt"])) + 2 for p in pairs]          # <s> ... </s>
        dropped = sum(s > MAX_SRC or t > MAX_TGT for s, t in zip(src, tgt))
        stats[split] = {"pairs": len(pairs), "src_mean": sum(src) / len(src), "src_max": max(src),
                        "tgt_mean": sum(tgt) / len(tgt), "tgt_max": max(tgt),
                        "dropped": dropped if split == "train" else 0}
        print(f"{split:5s} pairs={len(pairs):6,d}  src mean/max={stats[split]['src_mean']:.1f}/{max(src)}"
              f"  tgt mean/max={stats[split]['tgt_mean']:.1f}/{max(tgt)}  over-long={dropped}")
    assert BOS_ID != EOS_ID
    common.update_json(METRICS, "data", stats)


def starter_shapes():
    res = subprocess.run([sys.executable, "check_starter.py"], cwd=common.STARTER,
                         capture_output=True, text=True, check=True)
    (RESULTS / "check_starter.txt").write_text(res.stdout)
    print(res.stdout)


def pe_heatmap():
    pe = PositionalEncoding(256, max_len=100, dropout=0.0).pe[0].numpy()  # (100, 256)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(pe, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xlabel("embedding dimension")
    ax.set_ylabel("position")
    ax.set_title("Sinusoidal positional encoding (first 100 positions x 256 dimensions)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "positional_encoding.png", dpi=200)
    print(f"saved {FIGURES / 'positional_encoding.png'}")


if __name__ == "__main__":
    data_stats()
    starter_shapes()
    pe_heatmap()
