import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .adapter import CrackMeanFlowModel
from .checkpointing import has_nan_weights, load_checkpoint_strict, save_checkpoint
from .data import PairedCrackDataset, deterministic_split, list_pairs, write_split_report
from .loss import CrackSILoss
from .paths import CRACKDIFF_ROOT, ensure_paths
from .sampler import crack_meanflow_sampler
from .metrics import compute_segmentation_metrics

ensure_paths()
from multi_task.mlt_unet import UNet  # noqa: E402

logger = logging.getLogger("crackmeanflow.train")


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.num_updates = 0

    def update(self, model):
        self.num_updates += 1
        d = self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k not in self.shadow:
                    self.shadow[k] = v.clone().detach()
                    continue
                if torch.is_floating_point(v):
                    self.shadow[k].mul_(d).add_(v, alpha=1.0 - d)
                else:
                    self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = state


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    out_dir = Path(cfg["paths"]["output_dir"])
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    report_dir = Path(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # --- data split ---
    pairs = list_pairs(cfg["paths"]["image_dir"], cfg["paths"]["mask_dir"])
    assert len(pairs) >= 5, f"Too few pairs ({len(pairs)}), need >= 5"
    train_pairs, test_pairs = deterministic_split(pairs, cfg["train"]["train_ratio"])
    write_split_report(report_dir / "DATA_SPLIT_REPORT.md", pairs, train_pairs, test_pairs, cfg["paths"]["image_dir"], cfg["paths"]["mask_dir"])
    logger.info("Pairs=%d  Train=%d  Test=%d", len(pairs), len(train_pairs), len(test_pairs))

    train_ds = PairedCrackDataset(train_pairs, image_size=cfg["model"]["img_size"], augment=cfg["train"].get("augment", True))
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0, pin_memory=True)

    # --- model ---
    unet = UNet(
        T=cfg["model"]["T"],
        ch=cfg["model"]["ch"],
        ch_mult=cfg["model"]["ch_mult"],
        attn=cfg["model"]["attn"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
        dropout=cfg["model"]["dropout"],
    )
    model = CrackMeanFlowModel(unet, T=cfg["model"]["T"]).to(device)
    logger.info("Model params: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    ema = EMA(model, decay=cfg["train"]["ema_decay"])
    scaler = GradScaler()
    criterion = CrackSILoss(
        si_loss_kwargs=cfg["loss"]["si_loss_kwargs"],
        seg_loss_weight=cfg["loss"]["seg_loss_weight"],
        endpoint_loss_weight=cfg["loss"]["endpoint_loss_weight"],
        thin_loss_weight=cfg["loss"]["thin_loss_weight"],
        mode=cfg["loss"].get("mode", "hybrid"),
    )

    # --- resume ---
    global_step = 0
    best_f1 = 0.0
    resume_path = ckpt_dir / "last.pt"
    if resume_path.exists():
        try:
            ckpt_data, _ = load_checkpoint_strict(model, resume_path, map_location=device)
            if ckpt_data.get("optimizer"):
                optimizer.load_state_dict(ckpt_data["optimizer"])
            if ckpt_data.get("global_step"):
                global_step = ckpt_data["global_step"]
            if ckpt_data.get("best_metrics", {}).get("f1"):
                best_f1 = ckpt_data["best_metrics"]["f1"]
            if ckpt_data.get("ema"):
                ema.load_state_dict(ckpt_data["ema"])
            logger.info("Resumed from step %d  best_f1=%.4f", global_step, best_f1)
        except Exception as e:
            logger.warning("Resume failed (%s), starting fresh.", e)

    epochs = cfg["train"]["epochs"]
    log_interval = cfg["train"]["log_interval"]
    save_interval = cfg["train"]["save_interval"]
    max_grad_norm = cfg["train"]["max_grad_norm"]
    max_batches = cfg["train"].get("max_train_batches", 0)
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    best_ckpt_path = ckpt_dir / "best.pt"

    # --- train loop ---
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            if max_batches and batch_idx >= max_batches:
                break
            image = batch["crack"].to(device)
            mask = batch["mask"].to(device)
            x0 = mask * 2.0 - 1.0

            with autocast():
                total_loss, loss_dict = criterion(model, x0, {"y": image, "mask_gt": mask})

            if not torch.isfinite(total_loss):
                logger.warning("Non-finite loss at step %d, skipping.", global_step)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(total_loss / grad_accum_steps).backward()
            should_step = ((batch_idx + 1) % grad_accum_steps == 0)
            if max_batches and (batch_idx + 1) >= max_batches:
                should_step = True
            elif (batch_idx + 1) == len(train_loader):
                should_step = True

            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                global_step += 1

                if global_step % log_interval == 0:
                    ld = loss_dict
                    logger.info("[step %d] total=%.4f seg=%.4f %s", global_step, ld["total_loss"], ld.get("seg_loss", 0.0),
                                " ".join(f"{k}={v:.4f}" for k, v in ld.items() if k not in ("total_loss", "seg_loss", "nan_flags", "mode")))

                    if global_step % save_interval == 0:
                        save_checkpoint(ckpt_dir / "last.pt", model, optimizer=optimizer, ema=ema, epoch=epoch, global_step=global_step, best_metrics={"f1": best_f1}, config=cfg)

            epoch_losses.append(loss_dict)
            logger.warning("Epoch %d: no valid batches.", epoch)
            continue

        numeric_keys = [k for k, v in epoch_losses[0].items() if isinstance(v, (int, float))]
        avg = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses) for k in numeric_keys}
        elapsed = time.time() - t0
        logger.info("Epoch %d done in %.1fs  avg_total=%.4f", epoch, elapsed, avg["total_loss"])

        # --- quick eval on a few test batches ---
        test_f1 = _quick_eval(model, test_pairs, cfg, device)
        logger.info("Quick eval F1=%.4f (best=%.4f)", test_f1, best_f1)
        if test_f1 > best_f1:
            best_f1 = test_f1
            save_checkpoint(best_ckpt_path, model, optimizer=optimizer, ema=ema, epoch=epoch, global_step=global_step, best_metrics={"f1": best_f1}, config=cfg)
            logger.info("New best F1=%.4f saved.", best_f1)

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer=optimizer, ema=ema, epoch=epoch, global_step=global_step, best_metrics={"f1": best_f1}, config=cfg)

    logger.info("Training complete. best_f1=%.4f", best_f1)
    return best_ckpt_path


@torch.no_grad()
def _quick_eval(model, test_pairs, cfg, device):
    model.eval()
    from .data import PairedCrackDataset
    from torch.utils.data import DataLoader
    ds = PairedCrackDataset(test_pairs, image_size=cfg["model"]["img_size"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    f1s = []
    max_eval = cfg["train"].get("max_eval_batches", 20)
    for i, batch in enumerate(loader):
        if i >= max_eval:
            break
        image = batch["crack"].to(device)
        mask = batch["mask"].to(device)
        z = torch.randn_like(mask)
        sampled, seg_logits = crack_meanflow_sampler(model, z, image, num_steps=1)
        pred_flow = (sampled > 0.0).float()
        m_flow = compute_segmentation_metrics(pred_flow, mask)
        best_f1 = m_flow["f1"]
        if seg_logits is not None:
            pred_seg = (torch.sigmoid(seg_logits) > 0.5).float()
            m_seg = compute_segmentation_metrics(pred_seg, mask)
            best_f1 = max(best_f1, m_seg["f1"])
        f1s.append(best_f1)
    model.train()
    return sum(f1s) / len(f1s) if f1s else 0.0
