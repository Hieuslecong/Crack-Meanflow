import sys
import types

import torch
from torch import nn
import torch.nn.functional as F

from .paths import ensure_paths

ensure_paths()
if "torch.func" not in sys.modules:
    from functorch import jvp as _functorch_jvp

    _torch_func_module = types.ModuleType("torch.func")
    _torch_func_module.jvp = _functorch_jvp
    sys.modules["torch.func"] = _torch_func_module
    torch.func = _torch_func_module
from loss import SILoss  # noqa: E402
from multi_task.mltdiff import FocalTverskyLoss  # noqa: E402


class _SILossModelProxy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_classes = 0
        self.module = self

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


class CrackSILoss(nn.Module):
    def __init__(self, si_loss_kwargs=None, seg_loss_weight=1.0, endpoint_loss_weight=0.5, thin_loss_weight=0.0, mode="hybrid"):
        super().__init__()
        kwargs = dict(si_loss_kwargs or {})
        kwargs.setdefault("label_dropout_prob", 0.0)
        kwargs.setdefault("cfg_omega", 1.0)
        kwargs.setdefault("cfg_kappa", 0.0)
        self.si_loss = SILoss(**kwargs)
        self.seg_loss = FocalTverskyLoss()
        self.seg_loss_weight = float(seg_loss_weight)
        self.endpoint_loss_weight = float(endpoint_loss_weight)
        self.thin_loss_weight = float(thin_loss_weight)
        self.mode = str(mode)

    def _check_finite(self, name, tensor, context):
        if not torch.isfinite(tensor).all():
            ranges = []
            for key, value in context.items():
                if torch.is_tensor(value):
                    ranges.append(f"{key}: shape={tuple(value.shape)} min={value.min().item():.6g} max={value.max().item():.6g}")
            raise RuntimeError(f"Non-finite {name}. " + " | ".join(ranges))

    def forward(self, model, x0, model_kwargs=None):
        model_kwargs = dict(model_kwargs or {})
        y = model_kwargs.get("y")
        mask_gt = model_kwargs.get("mask_gt")
        if y is None or mask_gt is None:
            raise ValueError("CrackSILoss requires model_kwargs with `y` and `mask_gt`.")

        base_model = _unwrap(model)

        # segmentation branch should condition on image + noisy latent, not clean GT x0.
        z = torch.randn_like(x0)
        r = torch.zeros(x0.shape[0], device=x0.device)
        t = torch.ones(x0.shape[0], device=x0.device)
        u = model(z, r, t, y=y)
        seg_logits = base_model.get_seg_logits()
        if seg_logits is None:
            raise RuntimeError("Segmentation logits unavailable after noisy forward.")
        seg_loss = self.seg_loss(seg_logits, mask_gt)
        x0_pred = z - u
        endpoint_loss = F.l1_loss(x0_pred, x0)

        if self.mode == "seg_only":
            si_loss = x0.new_tensor(0.0)
            si_loss_ref_mean = x0.new_tensor(0.0)
        else:
            proxy = model if hasattr(model, "module") else _SILossModelProxy(model)
            si_loss_vec, si_loss_ref = self.si_loss(proxy, x0, {"y": y})
            si_loss = si_loss_vec.mean()
            si_loss_ref_mean = si_loss_ref.mean() if torch.is_tensor(si_loss_ref) else torch.as_tensor(si_loss_ref, device=x0.device)

        thin_loss = x0.new_tensor(0.0)
        total_loss = si_loss + self.seg_loss_weight * seg_loss + self.endpoint_loss_weight * endpoint_loss + self.thin_loss_weight * thin_loss

        context = {"x0": x0, "z": z, "u": u, "x0_pred": x0_pred, "seg_logits": seg_logits}
        for name, value in [("total_loss", total_loss), ("si_loss", si_loss), ("seg_loss", seg_loss), ("endpoint_loss", endpoint_loss)]:
            self._check_finite(name, value, context)

        loss_dict = {
            "total_loss": float(total_loss.detach().cpu().item()),
            "si_loss": float(si_loss.detach().cpu().item()),
            "si_loss_ref": float(si_loss_ref_mean.detach().cpu().item()),
            "seg_loss": float(seg_loss.detach().cpu().item()),
            "endpoint_loss": float(endpoint_loss.detach().cpu().item()),
            "thin_loss": float(thin_loss.detach().cpu().item()),
            "nan_flags": {},
            "mode": self.mode,
        }
        return total_loss, loss_dict
