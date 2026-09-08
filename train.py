"""Train the text-region style/language classifier."""
from __future__ import annotations

import cache_paths  # noqa: F401 — keep caches on D:
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import _print_class_counts, get_dataloaders, get_held_out_loader
from model import DEFAULT_BACKBONE, build_model, freeze_backbone_layers


def evaluate(model, loader, device, *, desc: str = "val"):
    model.eval()
    correct, total = 0, 0
    num_classes = len(loader.dataset.classes)
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for t, p in zip(labels.tolist(), preds.tolist()):
                confusion[t][p] += 1

    acc = correct / total if total else 0.0
    return acc, confusion


def recall_from_confusion(confusion: list[list[int]]) -> dict[str, float]:
    recalls: dict[str, float] = {}
    for i, row in enumerate(confusion):
        support = sum(row)
        recalls[str(i)] = (row[i] / support if support else 0.0)
    return recalls


def min_class_recall(confusion: list[list[int]]) -> float:
    recalls = recall_from_confusion(confusion)
    return min(recalls.values()) if recalls else 0.0


def calibrate_hw_threshold(
    model,
    loader,
    device,
    *,
    hw_idx: int,
    pr_idx: int,
) -> tuple[float, dict[str, float]]:
    """Pick threshold on p(handwritten) that maximizes min per-class recall."""
    model.eval()
    pairs: list[tuple[float, int]] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)[:, hw_idx].cpu().tolist()
            pairs.extend(zip(probs, labels.tolist()))

    best_t = 0.5
    best_stats = {"handwritten": 0.0, "printed": 0.0, "min_recall": 0.0}
    for t_int in range(5, 96):
        t = t_int / 100.0
        counts = {hw_idx: {"ok": 0, "n": 0}, pr_idx: {"ok": 0, "n": 0}}
        for p_hw, true in pairs:
            pred = hw_idx if p_hw >= t else pr_idx
            bucket = counts[true]
            bucket["n"] += 1
            bucket["ok"] += int(pred == true)
        hw_r = counts[hw_idx]["ok"] / counts[hw_idx]["n"] if counts[hw_idx]["n"] else 0.0
        pr_r = counts[pr_idx]["ok"] / counts[pr_idx]["n"] if counts[pr_idx]["n"] else 0.0
        min_r = min(hw_r, pr_r)
        if min_r >= best_stats["min_recall"]:
            best_t = t
            best_stats = {"handwritten": hw_r, "printed": pr_r, "min_recall": min_r}
    return best_t, best_stats


def train_epoch(model, loader, criterion, optimizer, device, epoch: int, epochs: int):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{epochs} train", unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.4f}" if total else "0.0000",
        )

    train_loss = running_loss / len(loader.dataset)
    train_acc = correct / total if total else 0.0
    return train_loss, train_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_crops")
    parser.add_argument("--held-out-dir", default="held_out_crops",
                        help="Real-document crops for early stopping (never in train)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.08)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--patience", type=int, default=4,
                        help="Early stopping patience on held-out real-doc accuracy")
    parser.add_argument("--max-overfit-gap", type=float, default=0.0,
                        help="Stop if train_acc - held_out_acc exceeds this (0=disabled)")
    parser.add_argument("--max-synthetic-en-printed", type=int, default=0,
                        help="Cap font-rendered English printed crops in training")
    parser.add_argument("--max-synthetic-ar-printed", type=int, default=0,
                        help="Cap font-rendered Arabic printed crops in training")
    parser.add_argument("--handwritten-weight-mult", type=float, default=2.5,
                        help="Extra loss weight multiplier for handwritten class")
    parser.add_argument("--early-stop-metric", choices=("acc", "min_recall"), default="min_recall",
                        help="Metric for checkpointing / early stopping on held-out crops")
    parser.add_argument("--style-only", action="store_true", default=True,
                        help="Train printed vs handwritten only (matches inference)")
    parser.add_argument("--four-class", action="store_false", dest="style_only",
                        help="Legacy 4-class language+style training")
    parser.add_argument("--backbone", default="efficientnet_b0",
                        choices=[
                            "resnet18", "efficientnet_b0",
                            "mobilenet_v3_large", "mobilenet_v3_small", "mobilenet_v2",
                        ])
    parser.add_argument("--freeze-early", action="store_true", default=True)
    parser.add_argument("--no-freeze-early", action="store_false", dest="freeze_early")
    parser.add_argument("--balance", action="store_true",
                        help="Oversample minority class (can inflate train acc / overfit)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default="best_model.pt")
    parser.add_argument("--log-file", default="training.log",
                        help="Append epoch summaries to this file ('' to disable)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_to_idx, class_weights = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balance_classes=args.balance,
        max_synthetic_en_printed=args.max_synthetic_en_printed,
        max_synthetic_ar_printed=args.max_synthetic_ar_printed,
        style_only=args.style_only,
        handwritten_weight_mult=args.handwritten_weight_mult,
    )
    held_out_loader = get_held_out_loader(
        args.held_out_dir, class_to_idx, batch_size=args.batch_size,
        style_only=args.style_only,
    )

    task = "style (printed vs handwritten)" if args.style_only else "4-class language+style"
    print(f"Task: {task}")

    print("Class mapping:", class_to_idx)
    print(f"Training on crops from: {args.data_dir}")
    sampling = "balanced oversampling" if args.balance else "natural shuffle + loss weights"
    print(f"  train: {len(train_loader.dataset):,} binarized crops ({sampling})")
    print(f"  class loss weights: {dict(zip(class_to_idx.keys(), class_weights.tolist()))}")
    print(f"  val:   {len(val_loader.dataset):,} random crop split")
    if held_out_loader:
        print(f"  held-out real docs: {len(held_out_loader.dataset):,} crops")
    else:
        print("  held-out: not found (optional — skipped)")
    _print_class_counts(Path(args.data_dir), "train")
    print(f"  batches/epoch: {len(train_loader):,} (batch_size={args.batch_size})")

    model = build_model(
        args.backbone,
        pretrained=True,
        num_classes=len(class_to_idx),
        dropout=args.dropout,
    )
    if args.freeze_early:
        freeze_backbone_layers(model, unfreeze_last_n=20)
    model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.08,
    )
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_metric = 0.0
    no_improve = 0
    t0 = time.time()
    log_path = Path(args.log_file) if args.log_file else None

    def log_line(msg: str) -> None:
        print(msg, flush=True)
        if log_path:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

    if log_path and log_path.exists():
        try:
            log_path.unlink()
        except PermissionError:
            log_line(f"--- new run (could not truncate {log_path}; appending) ---")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs,
        )
        scheduler.step()
        val_acc, _ = evaluate(model, val_loader, device, desc=f"Epoch {epoch:02d}/{args.epochs} val")

        if held_out_loader:
            held_acc, held_conf = evaluate(
                model, held_out_loader, device, desc=f"Epoch {epoch:02d}/{args.epochs} held-out"
            )
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            held_recalls = {
                idx_to_class[int(i)]: r for i, r in recall_from_confusion(held_conf).items()
            }
            min_recall = min(held_recalls.values()) if held_recalls else 0.0
            metric = min_recall if args.early_stop_metric == "min_recall" else held_acc
            metric_name = "held_out_min_recall" if args.early_stop_metric == "min_recall" else "held_out_acc"
        else:
            held_acc = None
            held_recalls = {}
            min_recall = None
            metric = val_acc
            metric_name = "val_acc"

        elapsed = time.time() - t0
        eta = elapsed / epoch * (args.epochs - epoch)
        held_str = f" held_out_acc={held_acc:.4f} |" if held_acc is not None else ""
        recall_str = ""
        if held_recalls:
            recall_str = (
                f" hw_recall={held_recalls.get('handwritten', 0):.3f}"
                f" pr_recall={held_recalls.get('printed', 0):.3f}"
                f" min_recall={min_recall:.3f} |"
            )
        log_line(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} |{held_str}{recall_str} "
            f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m"
        )

        if metric > best_metric:
            best_metric = metric
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "class_to_idx": class_to_idx,
                "dropout": args.dropout,
                "style_only": args.style_only,
                "hw_threshold": 0.5,
            }, args.output)
            log_line(f"  -> saved best model ({metric_name}={metric:.4f}) to {args.output}")
        else:
            no_improve += 1
            log_line(f"  -> no improvement ({no_improve}/{args.patience})")
            if no_improve >= args.patience:
                log_line(f"Early stopping at epoch {epoch} ({metric_name}={best_metric:.4f})")
                break

        if (
            args.max_overfit_gap > 0
            and held_acc is not None
            and (train_acc - held_acc) > args.max_overfit_gap
        ):
            log_line(
                f"  -> stopping: overfit gap {train_acc - held_acc:.4f} "
                f"> {args.max_overfit_gap:.2f} (train memorizing, held-out lagging)"
            )
            break

    print(f"\nBest {metric_name}: {best_metric:.4f}")

    if Path(args.output).is_file():
        ckpt = torch.load(args.output, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

    eval_loader = held_out_loader or val_loader
    _, confusion = evaluate(model, eval_loader, device, desc="final eval")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    print("Confusion matrix (rows=true, cols=pred):")
    print("           " + " ".join(f"{c:>14}" for c in classes))
    for i, row in enumerate(confusion):
        print(f"{classes[i]:>10} " + " ".join(f"{v:>14}" for v in row))
    for i, row in enumerate(confusion):
        support = sum(row)
        recall = row[i] / support if support else 0.0
        print(f"  recall {classes[i]}: {recall:.1%} ({row[i]}/{support})")

    if held_out_loader and args.style_only:
        hw_idx = class_to_idx["handwritten"]
        pr_idx = class_to_idx["printed"]
        hw_t, stats = calibrate_hw_threshold(
            model, held_out_loader, device, hw_idx=hw_idx, pr_idx=pr_idx,
        )
        print(
            f"\nCalibrated hw_threshold={hw_t:.2f} on held-out "
            f"(hw_recall={stats['handwritten']:.1%}, pr_recall={stats['printed']:.1%}, "
            f"min={stats['min_recall']:.1%})"
        )
        ckpt = torch.load(args.output, map_location="cpu", weights_only=False)
        ckpt["hw_threshold"] = hw_t
        ckpt["held_out_recall"] = stats
        torch.save(ckpt, args.output)
        print(f"Updated {args.output} with hw_threshold={hw_t:.2f}")


if __name__ == "__main__":
    main()
