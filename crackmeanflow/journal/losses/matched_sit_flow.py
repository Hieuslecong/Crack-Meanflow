from __future__ import annotations

import torch
from torch import nn

try:
    from torch.func import jvp
except ImportError:  # pragma: no cover
    from functorch import jvp


def _b(v: torch.Tensor) -> torch.Tensor:
    return v.reshape(-1, 1, 1, 1)


def _stratified_mask(batch: int, fraction: float, offset: int, device) -> torch.Tensor:
    fraction = float(fraction)
    if fraction <= 0:
        return torch.zeros(batch, dtype=torch.bool, device=device)
    if fraction >= 1:
        return torch.ones(batch, dtype=torch.bool, device=device)
    idx = torch.arange(int(offset), int(offset) + batch, device=device, dtype=torch.float64)
    return torch.floor((idx + 1) * fraction) > torch.floor(idx * fraction)


class MatchedSiTFlowLoss(nn.Module):
    """Capacity/sampling-matched MF vs iMF objective for CrackMeanFlow-SiT V1.

    Both modes share the exact same (r,t) sampler and adaptive normalization.
    No endpoint, thin, segmentation, geometry, GIC, or clean auxiliary loss is
    permitted in V1. Thus S0/S1 differ only in the core flow objective.
    """

    MODES = {"mf", "imf"}

    def __init__(
        self,
        mode: str,
        fm_fraction: float = 0.5,
        time_mu: float = -0.4,
        time_sigma: float = 1.0,
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
        fm_sampling: str = "stratified",
    ):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        self.fm_fraction = float(fm_fraction)
        if not (0.0 <= self.fm_fraction <= 1.0):
            raise ValueError("fm_fraction must lie in [0,1]")
        self.time_mu = float(time_mu)
        self.time_sigma = float(time_sigma)
        self.norm_p = float(norm_p)
        self.norm_eps = float(norm_eps)
        self.fm_sampling = str(fm_sampling)
        if self.fm_sampling != "stratified":
            raise ValueError("CRACKMEANFLOW_SIT_V1 requires deterministic stratified FM sampling")

    def _sample_rt(self, batch: int, device, sample_offset: int):
        n = torch.randn(2, batch, device=device) * self.time_sigma + self.time_mu
        a, b = torch.sigmoid(n[0]), torch.sigmoid(n[1])
        t, r = torch.maximum(a, b), torch.minimum(a, b)
        fm = _stratified_mask(batch, self.fm_fraction, sample_offset, device)
        r = torch.where(fm, t, r)
        return t, r, fm

    def _adaptive(self, per_sample: torch.Tensor) -> torch.Tensor:
        return (per_sample / (per_sample.detach() + self.norm_eps).pow(self.norm_p)).mean()

    def _mf(self, model, z, t, r, image, true_v):
        u = model(z, r, t, y=image)

        def fn(z_, r_, t_):
            return model(z_, r_, t_, y=image)

        with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
            _, du_dt = jvp(fn, (z, r, t), (true_v, torch.zeros_like(r), torch.ones_like(t)))
        target = (true_v - _b(t - r) * du_dt).detach()
        loss = self._adaptive((u - target).pow(2).flatten(1).sum(1))
        return loss, {"mf_loss": loss, "imf_u_loss": z.new_tensor(0.0), "imf_v_loss": z.new_tensor(0.0)}

    def _imf(self, model, z, t, r, image, true_v):
        out = model.flow_outputs(z, t, r, image)
        u, v = out["u"], out["v"]
        with torch.no_grad():
            v_c = model.flow_outputs(z, t, t, image)["v"]

            def fn(z_, t_, r_):
                return model.flow_outputs(z_, t_, r_, image)["u"]

            _, du = jvp(fn, (z, t, r), (v_c, torch.ones_like(t), torch.zeros_like(r)))
        corrected = u + _b(t - r) * du.detach()
        loss_u = self._adaptive((corrected - true_v.detach()).pow(2).flatten(1).sum(1))
        loss_v = self._adaptive((v - true_v.detach()).pow(2).flatten(1).sum(1))
        total = loss_u + loss_v
        return total, {"mf_loss": z.new_tensor(0.0), "imf_u_loss": loss_u, "imf_v_loss": loss_v}

    def forward(self, model, x0, image, sample_offset: int = 0):
        x0, image = x0.float(), image.float()
        batch = x0.shape[0]
        t, r, fm = self._sample_rt(batch, x0.device, sample_offset)
        eps = torch.randn_like(x0)
        z = (1 - _b(t)) * x0 + _b(t) * eps
        true_v = eps - x0
        if self.mode == "mf":
            total, parts = self._mf(model, z, t, r, image, true_v)
        else:
            total, parts = self._imf(model, z, t, r, image, true_v)
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite {self.mode} loss")
        logs = {
            "total_loss": float(total.detach()),
            "mf_loss": float(parts["mf_loss"].detach()),
            "imf_u_loss": float(parts["imf_u_loss"].detach()),
            "imf_v_loss": float(parts["imf_v_loss"].detach()),
            "fm_count": int(fm.sum().item()),
            "gic_active_samples": 0,
            "gic_active_batches": 0,
            "near_deployment_count": 0,
            "exact_deployment_count": 0,
        }
        return total, logs
