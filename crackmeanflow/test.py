import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .adapter import CrackMeanFlowModel
from .checkpointing import has_nan_weights, load_checkpoint_strict
from .data import PairedCrackDataset, list_pairs, load_split_csv, split_pairs
from .metrics import compute_segmentation_metrics
from .paths import ensure_paths
from .sampler import crack_meanflow_sampler
from .thin_metrics import compute_thin_crack_metrics

ensure_paths()
from multi_task.mlt_unet import UNet  # noqa: E402


class EMAWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def _save_mask_png(path, pred):
    arr = (pred.squeeze().detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(path)


@torch.no_grad()
def evaluate(cfg, ckpt_path, output_dir, num_steps=1, threshold=0.0, use_ema=True, split="test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    paths_cfg = cfg["paths"]
    if all(k in paths_cfg for k in ("train_csv", "val_csv", "test_csv")):
        _train_pairs = load_split_csv(paths_cfg["train_csv"])
        _val_pairs = load_split_csv(paths_cfg["val_csv"])
        test_pairs = load_split_csv(paths_cfg["test_csv"])
    else:
        pairs = list_pairs(paths_cfg["image_dir"], paths_cfg["mask_dir"])
        _train_pairs, _val_pairs, test_pairs, _policy = split_pairs(pairs, train_ratio=cfg["train"].get("train_ratio", 0.8), val_ratio=cfg["train"].get("val_ratio", 0.1), seed=42)
    eval_pairs = test_pairs if split == "test" else _val_pairs
    ds = PairedCrackDataset(eval_pairs, image_size=cfg["model"]["img_size"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    unet = UNet(
        T=cfg["model"]["T"],
        ch=cfg["model"]["ch"],
        ch_mult=cfg["model"]["ch_mult"],
        attn=cfg["model"]["attn"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
        dropout=cfg["model"]["dropout"],
    )
    model = CrackMeanFlowModel(unet, T=cfg["model"]["T"]).to(device)
    ckpt, _ = load_checkpoint_strict(model, ckpt_path, map_location=device)
    if use_ema and ckpt.get("ema") is not None:
        if has_nan_weights(ckpt["ema"]):
            raise RuntimeError("EMA weights contain NaN/Inf, refusing to load.")
        ema_unet = UNet(
            T=cfg["model"]["T"],
            ch=cfg["model"]["ch"],
            ch_mult=cfg["model"]["ch_mult"],
            attn=cfg["model"]["attn"],
            num_res_blocks=cfg["model"]["num_res_blocks"],
            dropout=cfg["model"]["dropout"],
        )
        model = CrackMeanFlowModel(ema_unet, T=cfg["model"]["T"]).to(device)
        load_checkpoint_strict(model, ckpt_path, key="ema", map_location=device)
    model.eval()

    totals = []
    latencies = []
    for batch in loader:
        name = batch["name"][0]
        image = batch["crack"].to(device)
        mask = batch["mask"].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        z = torch.randn_like(mask)
        sampled, seg_logits = crack_meanflow_sampler(model, z, image, num_steps=num_steps)
        pred = (sampled > threshold).float()

        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

        metrics = compute_segmentation_metrics(pred, mask)


        thin = compute_thin_crack_metrics(pred, mask)
        merged = {**{k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()}, **{k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in thin.items()}}
        merged["name"] = name
        totals.append(merged)
        _save_mask_png(pred_dir / f"{name}.png", pred[0])

    def avg(key):
        return sum(item[key] for item in totals) / len(totals) if totals else 0.0

    metrics_json = {
        "num_samples": len(totals),
        "f1": avg("f1"),
        "iou": avg("iou"),
        "dice": avg("dice"),
        "precision": avg("precision"),
        "recall": avg("recall"),
        "accuracy": avg("accuracy"),
        "thin_recall": avg("thin_recall"),
        "thin_precision": avg("thin_precision"),
        "thin_f1": avg("thin_f1"),
        "boundary_f1": avg("boundary_f1"),
        "latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "throughput_fps": (len(latencies) / sum(latencies)) if latencies and sum(latencies) > 0 else 0.0,
        "num_steps": int(num_steps),
        "threshold": float(threshold),
        "checkpoint": str(ckpt_path),
        "use_ema": bool(use_ema),
        "per_sample": totals,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2) + "\n")

    report = [
        "# CrackMeanFlow Test Report",
        "",
        f"Checkpoint: `{ckpt_path}`",
        f"Samples: {metrics_json['num_samples']}",
        f"F1: {metrics_json['f1']:.6f}",
        f"IoU: {metrics_json['iou']:.6f}",
        f"Dice: {metrics_json['dice']:.6f}",
        f"Precision: {metrics_json['precision']:.6f}",
        f"Recall: {metrics_json['recall']:.6f}",
        f"Thin recall: {metrics_json['thin_recall']:.6f}",
        f"Thin F1: {metrics_json['thin_f1']:.6f}",
        f"Latency (s/img): {metrics_json['latency_seconds']:.6f}",
        f"Throughput (img/s): {metrics_json['throughput_fps']:.6f}",
    ]
    (output_dir / "TEST_REPORT.md").write_text("\n".join(report) + "\n")
    return metrics_json
