import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from crackmeanflow.adapter import CrackMeanFlowModel
from crackmeanflow.checkpointing import load_checkpoint_strict, save_checkpoint
from crackmeanflow.loss import CrackSILoss
from crackmeanflow.metrics import compute_segmentation_metrics
from crackmeanflow.paths import CRACKDIFF_ROOT, CRACKMEANFLOW_ROOT, MEANFLOW_ROOT
from crackmeanflow.sampler import crack_meanflow_sampler
from crackmeanflow.thin_metrics import compute_thin_crack_metrics

sys.path.insert(0, CRACKDIFF_ROOT)
from multi_task.mlt_unet import UNet  # noqa: E402


def assert_finite_grad(model):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients created"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"


def main():
    assert Path(CRACKDIFF_ROOT).exists()
    assert Path(MEANFLOW_ROOT).exists()
    assert Path(CRACKMEANFLOW_ROOT).exists()
    torch.manual_seed(0)
    device = torch.device("cpu")
    unet = UNet(T=16, ch=32, ch_mult=[1, 2], attn=[], num_res_blocks=1, dropout=0.0).to(device)
    model = CrackMeanFlowModel(unet, T=16).to(device)
    b = 2
    x = torch.randn(b, 1, 256, 256, device=device)
    image = torch.randn(b, 3, 256, 256, device=device)
    mask = (torch.rand(b, 1, 256, 256, device=device) > 0.8).float()
    r = torch.zeros(b, device=device)
    t = torch.ones(b, device=device)
    out = model(x, r, t, y=image)
    assert tuple(out.shape) == (b, 1, 256, 256)
    assert model.get_seg_logits() is not None
    assert tuple(model.get_seg_logits().shape) == (b, 1, 256, 256)

    criterion = CrackSILoss(si_loss_kwargs={"time_sampler": "uniform", "label_dropout_prob": 0.0}, seg_loss_weight=0.1, endpoint_loss_weight=1.0)
    x0 = mask * 2.0 - 1.0
    total_loss, loss_dict = criterion(model, x0, {"y": image, "mask_gt": mask})
    assert total_loss.ndim == 0
    assert torch.isfinite(total_loss)
    assert {"total_loss", "si_loss", "seg_loss", "endpoint_loss"}.issubset(loss_dict)
    total_loss.backward()
    assert_finite_grad(model)

    with torch.no_grad():
        sampled, seg = crack_meanflow_sampler(model, torch.randn_like(mask), image, num_steps=1)
        assert tuple(sampled.shape) == tuple(mask.shape)
        assert seg is not None and tuple(seg.shape) == tuple(mask.shape)
        sampled4, _ = crack_meanflow_sampler(model, torch.randn_like(mask), image, num_steps=4)
        assert tuple(sampled4.shape) == tuple(mask.shape)

    metrics = compute_segmentation_metrics((sampled > 0).float(), mask)
    thin = compute_thin_crack_metrics((sampled > 0).float(), mask)
    for key in ["iou", "dice", "f1", "precision", "recall"]:
        assert 0.0 <= metrics[key] <= 1.0, (key, metrics[key])
    for key in ["thin_recall", "thin_precision", "thin_f1"]:
        assert 0.0 <= thin[key] <= 1.0, (key, thin[key])

    ckpt = ROOT / "outputs" / "smoke_ckpt.pt"
    save_checkpoint(ckpt, model, epoch=0, global_step=1, architecture={"T": 16})
    reloaded = CrackMeanFlowModel(UNet(T=16, ch=32, ch_mult=[1, 2], attn=[], num_res_blocks=1, dropout=0.0).to(device), T=16).to(device)
    load_checkpoint_strict(reloaded, ckpt, map_location=device)
    print("SMOKE_TEST_PASS")


if __name__ == "__main__":
    main()
