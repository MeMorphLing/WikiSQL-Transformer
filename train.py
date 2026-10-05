"""Task 3: train the Transformer once from random initialisation.

    python train.py                       # full run: 20 epochs, batch 64 (assignment setting)
    python train.py --epochs 1 --max-steps 50 --out checkpoints/smoke.pt   # pipeline smoke test

Per epoch it logs train loss, dev loss and learning rate to results/train_log.csv
and keeps only the checkpoint with the lowest dev loss. The test set is never touched.
"""
from __future__ import annotations

import argparse
import csv
import math
import time

import common
import torch
import torch.nn as nn
from common import PAD_ID, ModelConfig, TrainConfig
from dataset import make_loader  # starter
from model import Transformer
from tqdm import tqdm


def noam_lambda(d_model: int, warmup: int):
    """Eq. 3: lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5).
    LambdaLR calls this with the number of completed optimiser steps (0, 1, ...);
    the paper counts from 1, hence step + 1."""
    def f(step: int) -> float:
        step += 1
        return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)
    return f


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None,
              scaler=None, amp=False, max_steps=None, desc=""):
    """One pass over `loader`. Trains if an optimiser is given. Returns per-token
    loss (label-smoothed, same criterion for train and dev) and token accuracy."""
    training = optimizer is not None
    model.train(training)
    tot_loss, tot_tok, tot_ok = 0.0, 0, 0
    for step, (src, tgt) in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_steps is not None and step >= max_steps:
            break
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        dec_in, gold = tgt[:, :-1], tgt[:, 1:]  # teacher forcing
        with torch.set_grad_enabled(training), \
                torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(src, dec_in)
            loss_sum = criterion(logits.reshape(-1, logits.size(-1)).float(), gold.reshape(-1))
        n_tok = int((gold != PAD_ID).sum())
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss_sum / n_tok).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        with torch.no_grad():
            ok = ((logits.argmax(-1) == gold) & (gold != PAD_ID)).sum().item()
        tot_loss += loss_sum.item()
        tot_tok += n_tok
        tot_ok += ok
    return tot_loss / tot_tok, tot_ok / tot_tok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--max-steps", type=int, default=None, help="cap batches per epoch (smoke tests only)")
    ap.add_argument("--out", default=str(common.BEST_CKPT))
    ap.add_argument("--log", default=str(common.RESULTS / "train_log.csv"))
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 mixed precision on GPU")
    args = ap.parse_args()

    tcfg = TrainConfig(epochs=args.epochs)
    common.set_seed(tcfg.seed)
    device = common.get_device()
    amp = device.type == "cuda" and not args.no_amp

    sp = common.load_sp()
    mcfg = ModelConfig(vocab_size=sp.get_piece_size())
    train_dl = make_loader(str(common.pairs_path("train")), sp, train=True, batch_size=tcfg.batch_size)
    dev_dl = make_loader(str(common.pairs_path("dev")), sp, train=False, batch_size=tcfg.batch_size)

    model = Transformer(**common.to_dict(mcfg)).to(device)
    n_params = model.num_parameters()
    print(f"device={device} amp={amp} trainable parameters={n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=tcfg.label_smoothing,
                                    reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=tcfg.betas, eps=tcfg.eps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, noam_lambda(mcfg.d_model, tcfg.warmup_steps))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    common.RESULTS.mkdir(exist_ok=True)
    common.CKPT_DIR.mkdir(exist_ok=True)
    fields = ["epoch", "step", "lr", "train_loss", "train_acc", "dev_loss", "dev_acc", "epoch_sec", "best"]
    best_loss, best_epoch, t0 = math.inf, -1, time.time()
    with open(args.log, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, tcfg.epochs + 1):
            te = time.time()
            tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer, scheduler,
                                        scaler, amp, args.max_steps, f"epoch {epoch} train")
            dv_loss, dv_acc = run_epoch(model, dev_dl, criterion, device, amp=amp,
                                        max_steps=args.max_steps, desc=f"epoch {epoch} dev")
            is_best = dv_loss < best_loss
            if is_best:
                best_loss, best_epoch = dv_loss, epoch
                torch.save({"model": model.state_dict(), "model_config": common.to_dict(mcfg),
                            "train_config": common.to_dict(tcfg), "epoch": epoch,
                            "dev_loss": dv_loss}, args.out)
            row = {"epoch": epoch, "step": scheduler.last_epoch,
                   "lr": f"{optimizer.param_groups[0]['lr']:.3e}",
                   "train_loss": f"{tr_loss:.4f}", "train_acc": f"{tr_acc:.4f}",
                   "dev_loss": f"{dv_loss:.4f}", "dev_acc": f"{dv_acc:.4f}",
                   "epoch_sec": f"{time.time() - te:.1f}", "best": int(is_best)}
            writer.writerow(row)
            f.flush()
            print(" | ".join(f"{k}={v}" for k, v in row.items()))

    total_min = (time.time() - t0) / 60
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    summary = {"trainable_parameters": n_params, "epochs_trained": tcfg.epochs,
               "best_epoch": best_epoch, "best_dev_loss": round(best_loss, 4),
               "training_minutes": round(total_min, 1), "device": gpu,
               "total_steps": scheduler.last_epoch}
    if args.max_steps is None and args.out == str(common.BEST_CKPT):  # real run only
        common.update_json(common.METRICS, "training", summary)
    print(summary)


if __name__ == "__main__":
    main()
