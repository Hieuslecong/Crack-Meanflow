#!/usr/bin/env python3
"""Standalone supervised segmentation baseline for OmniCrack30K.

Diagnostic only. Does not import crackmeanflow or crackdiff packages.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

UNET_FILE = Path("/home/hieulc/avitech11/crack_diff/crackdiff/multi_task/mlt_unet.py")
DEFAULT_THRESHOLDS = [
    -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
]


def load_unet_class(unet_file: Path = UNET_FILE) -> type[nn.Module]:
    spec = importlib.util.spec_from_file_location("standalone_mlt_unet", unet_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load UNet module from {unet_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UNet


class PairedCrackDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str, str]], image_size: int = 256, augment: bool = False):
        self.pairs = list(pairs)
        self.augment = bool(augment)
        self.image_tf = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor()])
        self.mask_tf = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        name, image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        image_t = self.image_tf(image)
        mask_t = (self.mask_tf(mask) > 0.5).float()
        return {"name": name, "crack": image_t, "mask": mask_t}


def compute_segmentation_metrics(pred_binary: torch.Tensor, mask_gt: torch.Tensor, eps: float = 1e-7) -> dict[str, float]:
    pred = (pred_binary.float() > 0.5).float().view(-1)
    gt = (mask_gt.float() > 0.5).float().view(-1)
    tp = (pred * gt).sum()
    fp = (pred * (1.0 - gt)).sum()
    fn = ((1.0 - pred) * gt).sum()
    tn = ((1.0 - pred) * (1.0 - gt)).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    dice = f1
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    if gt.sum() == 0 and pred.sum() == 0:
        precision = recall = f1 = dice = iou = accuracy.new_tensor(1.0)
    return {
        "iou": float(iou.item()),
        "dice": float(dice.item()),
        "f1": float(f1.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "accuracy": float(accuracy.item()),
        "tp": float(tp.item()),
        "fp": float(fp.item()),
        "fn": float(fn.item()),
        "tn": float(tn.item()),
    }


class DirectSegWrapper(nn.Module):
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b, _, h, w = image.shape
        x_dummy = torch.zeros(b, 1, h, w, device=image.device, dtype=image.dtype)
        t_dummy = torch.zeros(b, device=image.device, dtype=torch.long)
        _, seg_logits = self.unet(x_dummy, t_dummy, image)
        return seg_logits


def dice_loss_from_probs(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()


def tversky_loss_from_probs(probs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.3, beta: float = 0.7) -> torch.Tensor:
    tp = (probs * targets).sum(dim=(2, 3))
    fp = (probs * (1.0 - targets)).sum(dim=(2, 3))
    fn = ((1.0 - probs) * targets).sum(dim=(2, 3))
    score = (tp + 1e-5) / (tp + alpha * fp + beta * fn + 1e-5)
    return (1.0 - score).mean()


def thin_proxy_loss_from_probs(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Lightweight boundary/thin proxy: emphasize pixels near positives via max-pool dilation.
    weight = 1.0 + 4.0 * F.max_pool2d(targets, kernel_size=3, stride=1, padding=1)
    return F.binary_cross_entropy(probs.clamp(1e-5, 1.0 - 1e-5), targets, weight=weight)


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    return bce + dice_loss_from_probs(probs, targets)


def make_loss(cfg: dict[str, Any]):
    loss_name = str(cfg.get("loss", "bce_dice"))
    alpha = float(cfg.get("tversky_alpha", 0.3))
    beta = float(cfg.get("tversky_beta", 0.7))
    gamma = float(cfg.get("focal_gamma", 2.0))
    thin_weight = float(cfg.get("thin_weight", 0.5))

    def _loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = dice_loss_from_probs(probs, targets)
        if loss_name == "bce_dice":
            return bce + dice
        if loss_name == "bce_dice_tversky":
            return bce + dice + tversky_loss_from_probs(probs, targets, alpha=alpha, beta=beta)
        if loss_name == "focal_bce_dice":
            bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            pt = torch.exp(-bce_map)
            focal = ((1.0 - pt) ** gamma * bce_map).mean()
            return focal + dice
        if loss_name == "bce_dice_thin":
            return bce + dice + thin_weight * thin_proxy_loss_from_probs(probs, targets)
        raise ValueError(f"Unknown supervised loss: {loss_name}")

    return _loss


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def load_split_csv(path: Path) -> list[tuple[str, str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [(r["name"], r["image_path"], r["mask_path"]) for r in reader]


def make_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("supervised_baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_file)
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def aggregate_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    tp = sum(m["tp"] for m in metrics)
    fp = sum(m["fp"] for m in metrics)
    fn = sum(m["fn"] for m in metrics)
    tn = sum(m["tn"] for m in metrics)
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "iou": float(iou),
        "dice": float(f1),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
    selected_threshold: float | None = None,
) -> dict[str, Any]:
    model.eval()
    use_thresholds = [selected_threshold] if selected_threshold is not None else thresholds
    metrics_by_th: dict[float, list[dict[str, float]]] = {float(th): [] for th in use_thresholds}

    for batch in loader:
        image = batch["crack"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(image)
            probs = torch.sigmoid(logits)
        for th in use_thresholds:
            pred = (probs > th).float()
            metrics_by_th[float(th)].append(compute_segmentation_metrics(pred, mask))

    by_threshold = {str(th): aggregate_metrics(ms) for th, ms in metrics_by_th.items()}
    if selected_threshold is not None:
        best_th = float(selected_threshold)
    else:
        best_th = max(use_thresholds, key=lambda th: by_threshold[str(float(th))]["f1"])
    out = dict(by_threshold[str(float(best_th))])
    out["best_th"] = float(best_th)
    out["best_f1"] = out["f1"]
    out["by_threshold"] = by_threshold
    model.train()
    return out


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    test = payload["test_metrics"]
    lines = [
        "# Supervised Baseline OmniCrack30K Report",
        "",
        "Diagnostic only: direct supervised image-to-mask segmentation, not CrackMeanFlow success metric.",
        "",
        "## Data",
        f"- Train: {payload['split_sizes']['train']}",
        f"- Val: {payload['split_sizes']['val']}",
        f"- Test: {payload['split_sizes']['test']}",
        "- Masks: background=0, crack=255 -> tensor {0,1}",
        "",
        "## Validation threshold",
        f"- Selected on val: {payload['best_threshold']}",
        f"- Best val F1/Dice: {payload['best_val_f1']:.6f}",
        "",
        "## Test metrics",
        f"- Baseline F1: {test['f1']:.6f}",
        f"- Dice: {test['dice']:.6f}",
        f"- IoU: {test['iou']:.6f}",
        f"- Precision: {test['precision']:.6f}",
        f"- Recall: {test['recall']:.6f}",
        "- Thin recall/F1: not computed in standalone supervised baseline",
        "",
        "## Decision",
    ]
    if test["f1"] < 0.60:
        lines.append("- Supervised baseline F1 < 0.60 -> data/mask/split is blocker.")
    elif test["f1"] >= 0.70:
        lines.append("- Supervised baseline F1 >= 0.70 -> continue CrackMeanFlow objective.")
    else:
        lines.append("- Supervised baseline F1 in [0.60, 0.70) -> inspect errors before CrackMeanFlow objective.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--report-file", type=Path, default=None)
    parser.add_argument("--metrics-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--debug-smoke", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    cfg = load_config(args.config)
    train_csv = resolve_path(args.train_csv or cfg.get("train_csv", "splits/omnicrack30k/omnicrack30k_5k_train.csv"), root)
    val_csv = resolve_path(args.val_csv or cfg.get("val_csv", "splits/omnicrack30k/omnicrack30k_5k_val.csv"), root)
    test_csv = resolve_path(args.test_csv or cfg.get("test_csv", "splits/omnicrack30k/omnicrack30k_5k_test.csv"), root)
    out_dir = resolve_path(args.out_dir or cfg.get("out_dir", "outputs/supervised_baseline"), root)
    ckpt_dir = resolve_path(args.ckpt_dir or cfg.get("ckpt_dir", "checkpoints_supervised"), root)
    log_file = resolve_path(args.log_file or cfg.get("log_file", "logs/supervised_baseline.log"), root)
    report_file = resolve_path(args.report_file or cfg.get("report_file", "reports/SUPERVISED_BASELINE_OMNICRACK_REPORT.md"), root)
    metrics_file = resolve_path(args.metrics_file or cfg.get("metrics_file", "outputs/supervised_baseline/metrics.json"), root)
    image_size = int(cfg.get("image_size", 256))
    batch_size = int(args.batch_size or cfg.get("batch_size", 16))
    epochs = int(args.epochs or cfg.get("epochs", 30))
    lr = float(args.lr or cfg.get("learning_rate", 3e-4))
    thresholds = [float(x) for x in cfg.get("thresholds", DEFAULT_THRESHOLDS)]

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    logger = make_logger(log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pairs = load_split_csv(train_csv)
    val_pairs = load_split_csv(val_csv)
    test_pairs = load_split_csv(test_csv)
    if args.debug_smoke:
        train_pairs = train_pairs[: min(len(train_pairs), batch_size * 2)]
        val_pairs = val_pairs[: min(len(val_pairs), batch_size)]
        test_pairs = test_pairs[: min(len(test_pairs), batch_size)]
        epochs = min(epochs, 1)

    logger.info("Loaded splits: train=%d val=%d test=%d", len(train_pairs), len(val_pairs), len(test_pairs))
    logger.info("Device=%s batch_size=%d epochs=%d lr=%s debug_smoke=%s", device, batch_size, epochs, lr, args.debug_smoke)

    train_ds = PairedCrackDataset(train_pairs, image_size=image_size, augment=True)
    val_ds = PairedCrackDataset(val_pairs, image_size=image_size, augment=False)
    test_ds = PairedCrackDataset(test_pairs, image_size=image_size, augment=False)
    nw = 0 if args.debug_smoke else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=device.type == "cuda")

    UNet = load_unet_class()
    model = DirectSegWrapper(UNet(T=1000, ch=32, ch_mult=[1, 2], attn=[], num_res_blocks=2, dropout=0.1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=device.type == "cuda")
    criterion = make_loss(cfg)

    best_val_f1 = -1.0
    best_threshold = thresholds[0]
    history: list[dict[str, Any]] = []
    best_ckpt = ckpt_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            image = batch["crack"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(image)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())

        train_loss = epoch_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, device, thresholds)
        elapsed = time.time() - t0
        logger.info(
            "Epoch %03d/%03d loss=%.6f val_f1=%.6f val_iou=%.6f th=%.3f time=%.1fs",
            epoch,
            epochs,
            train_loss,
            val_metrics["f1"],
            val_metrics["iou"],
            val_metrics["best_th"],
            elapsed,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_metrics": val_metrics})
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_threshold = float(val_metrics["best_th"])
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_threshold": best_threshold, "best_val_f1": best_val_f1}, best_ckpt)

    logger.info("Training complete. best_val_f1=%.6f threshold=%.3f", best_val_f1, best_threshold)
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device, thresholds, selected_threshold=best_threshold)
    logger.info(
        "Test metrics: f1=%.6f dice=%.6f iou=%.6f precision=%.6f recall=%.6f threshold=%.3f",
        test_metrics["f1"],
        test_metrics["dice"],
        test_metrics["iou"],
        test_metrics["precision"],
        test_metrics["recall"],
        best_threshold,
    )

    payload = {
        "config": cfg,
        "debug_smoke": args.debug_smoke,
        "split_sizes": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
        "best_val_f1": best_val_f1,
        "best_threshold": best_threshold,
        "history": history,
        "test_metrics": test_metrics,
        "artifacts": {"best_ckpt": str(best_ckpt), "log_file": str(log_file), "report_file": str(report_file)},
    }
    with metrics_file.open("w") as f:
        json.dump(payload, f, indent=2)
    write_report(report_file, payload)
    logger.info("Wrote metrics=%s report=%s ckpt=%s", metrics_file, report_file, best_ckpt)


if __name__ == "__main__":
    main()
