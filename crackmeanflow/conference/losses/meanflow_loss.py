from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch.func import jvp
except ImportError:
    from functorch import jvp


def _b(v):
    return v.reshape(-1, 1, 1, 1)


class ConferenceMeanFlowLoss(nn.Module):
    """Conference MeanFlow objective with backward-compatible curriculum control.

    Historical configs use epoch-based curriculum stages. Paper-V3 configs may
    instead use ``time_curriculum.basis=optimizer_step`` and provide
    ``effective_batch_size`` plus ``optimizer_steps`` ranges. The canonical
    trainer already supplies ``sample_offset``; for a fixed effective batch this
    gives an exact optimizer-step index without changing the historical trainer
    call signature.
    """

    def __init__(
        self,
        ratio_r_not_equal_t=.35,
        boundary_prob=.25,
        time_mu=-.4,
        time_sigma=1.,
        adaptive_p=.75,
        adaptive_eps=1e-3,
        endpoint_loss_weight=.5,
        thin_loss_weight=0.,
        seg_loss_weight=0.,
        cfg_drop_prob=0.,
        time_curriculum=None,
        boundary_sampling='bernoulli',
    ):
        super().__init__()
        self.base_interval = float(ratio_r_not_equal_t)
        self.boundary_prob = float(boundary_prob)
        self.time_mu = float(time_mu)
        self.time_sigma = float(time_sigma)
        self.adaptive_p = float(adaptive_p)
        self.adaptive_eps = float(adaptive_eps)
        self.endpoint_loss_weight = float(endpoint_loss_weight)
        self.thin_loss_weight = float(thin_loss_weight)
        self.seg_loss_weight = float(seg_loss_weight)
        self.cfg_drop_prob = float(cfg_drop_prob)
        self.time_curriculum = time_curriculum or {'enabled': False}
        self.boundary_sampling = str(boundary_sampling)
        self.epoch = 0
        self.optimizer_step = 0
        if self.boundary_sampling not in {'bernoulli', 'stratified'}:
            raise ValueError(f'unknown boundary_sampling={self.boundary_sampling!r}')
        self._validate_curriculum()

    def _validate_curriculum(self):
        tc = self.time_curriculum or {}
        if not tc.get('enabled'):
            return
        basis = str(tc.get('basis', 'epoch'))
        if basis not in {'epoch', 'optimizer_step'}:
            raise ValueError(f'unknown time_curriculum basis={basis!r}')
        stages = list(tc.get('stages', []))
        if not stages:
            raise ValueError('enabled time_curriculum requires non-empty stages')
        key = 'optimizer_steps' if basis == 'optimizer_step' else 'epochs'
        if basis == 'optimizer_step' and int(tc.get('effective_batch_size', 0)) < 1:
            raise ValueError('optimizer-step curriculum requires effective_batch_size >= 1')
        previous_hi = None
        for index, stage in enumerate(stages):
            bounds = stage.get(key)
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f'curriculum stage {index} requires {key}: [lo, hi]')
            lo, hi = int(bounds[0]), int(bounds[1])
            if lo < 0 or hi <= lo:
                raise ValueError(f'invalid curriculum stage range {bounds!r}')
            if previous_hi is not None and lo != previous_hi:
                raise ValueError('curriculum stages must be contiguous and non-overlapping')
            previous_hi = hi

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def set_optimizer_step(self, optimizer_step):
        self.optimizer_step = int(optimizer_step)

    def _curriculum_position(self, sample_offset=None):
        tc = self.time_curriculum or {}
        basis = str(tc.get('basis', 'epoch'))
        if basis == 'optimizer_step':
            if sample_offset is not None:
                effective_batch = int(tc['effective_batch_size'])
                return int(sample_offset) // effective_batch
            return int(self.optimizer_step)
        return int(self.epoch)

    def _stage(self, sample_offset=None):
        interval_prob = self.base_interval
        max_gap = 1.0
        boundary_prob = self.boundary_prob
        tc = self.time_curriculum or {}
        if tc.get('enabled'):
            basis = str(tc.get('basis', 'epoch'))
            key = 'optimizer_steps' if basis == 'optimizer_step' else 'epochs'
            position = self._curriculum_position(sample_offset)
            matched = False
            for stage in tc.get('stages', []):
                lo, hi = [int(x) for x in stage[key]]
                if lo <= position < hi:
                    if 'fm_ratio' in stage:
                        interval_prob = 1.0 - float(stage['fm_ratio'])
                    if 'interval_ratio' in stage:
                        interval_prob = float(stage['interval_ratio'])
                    max_gap = float(stage.get('max_gap', max_gap))
                    boundary_prob = float(stage.get('boundary_prob', boundary_prob))
                    matched = True
                    break
            if not matched:
                raise RuntimeError(
                    f'time curriculum has no stage for {basis} position={position}; '
                    'the scientific schedule is incomplete'
                )
        if max_gap < 1.0:
            boundary_prob = 0.0
        return (
            min(max(interval_prob, 0.0), 1.0),
            min(max(max_gap, 0.0), 1.0),
            min(max(boundary_prob, 0.0), 1.0),
        )

    def _sample_r_t(self, b, device, return_stage=False, sample_offset=None):
        interval_prob, max_gap, boundary_prob = self._stage(sample_offset=sample_offset)
        n = torch.randn(2, b, device=device) * self.time_sigma + self.time_mu
        a, bb = torch.sigmoid(n[0]), torch.sigmoid(n[1])
        t = torch.maximum(a, bb)
        r = torch.minimum(a, bb)
        if max_gap < 1.0:
            r = torch.maximum(r, t - max_gap)
        collapse = torch.rand(b, device=device) >= interval_prob
        r = torch.where(collapse, t, r)
        if boundary_prob <= 0:
            boundary = torch.zeros(b, device=device, dtype=torch.bool)
        elif self.boundary_sampling == 'stratified' and sample_offset is not None:
            idx = torch.arange(int(sample_offset), int(sample_offset) + b, device=device, dtype=torch.float64)
            p = float(boundary_prob)
            boundary = torch.floor((idx + 1) * p) > torch.floor(idx * p)
        else:
            boundary = torch.rand(b, device=device) < boundary_prob
        r = torch.where(boundary, torch.zeros_like(r), r)
        t = torch.where(boundary, torch.ones_like(t), t)
        stage = {
            'curriculum_basis': str((self.time_curriculum or {}).get('basis', 'epoch')),
            'curriculum_position': int(self._curriculum_position(sample_offset)),
            'interval_prob': interval_prob,
            'max_gap': max_gap,
            'boundary_prob': boundary_prob,
            'boundary_sampling': self.boundary_sampling,
            'boundary_count': int(boundary.sum().item()),
            'fm_count': int((r == t).sum().item()),
            'realized_fm_fraction': float((r == t).float().mean().item()),
        }
        return (r, t, stage) if return_stage else (r, t)

    @staticmethod
    def _dice(pred, target, eps=1e-6):
        p = ((pred + 1) * .5).clamp(0, 1)
        g = (target + 1) * .5
        dims = tuple(range(1, p.ndim))
        inter = (p * g).sum(dims)
        den = p.sum(dims) + g.sum(dims)
        return (1 - (2 * inter + eps) / (den + eps)).mean()

    @staticmethod
    def _thin(pred, target):
        pos = (target > 0).float()
        near = (F.conv2d(pos, torch.ones(1, 1, 3, 3, device=pos.device), padding=1) > 0).float()
        return ((pred - target).abs() * (1 + 4 * near)).mean()

    def forward(self, model, x0, model_kwargs=None):
        kw = dict(model_kwargs or {})
        y = kw.get('y')
        if y is None:
            raise ValueError('ConferenceMeanFlowLoss requires conditioning image y')
        x0 = x0.float()
        y = y.float()
        b = x0.shape[0]
        device = x0.device
        if self.cfg_drop_prob > 0:
            drop = (torch.rand(b, device=device) < self.cfg_drop_prob).reshape(-1, 1, 1, 1)
            y = torch.where(drop, torch.zeros_like(y), y)
        sample_offset = kw.get('sample_offset', None)
        r, t, stage = self._sample_r_t(b, device, return_stage=True, sample_offset=sample_offset)
        eps = torch.randn_like(x0)
        z = (1 - _b(t)) * x0 + _b(t) * eps
        v = eps - x0
        u = model(z, r, t, y=y)
        base = model.module if hasattr(model, 'module') else model
        seg_logits = base.get_seg_logits()

        def fn(z_, r_, t_):
            return model(z_, r_, t_, y=y)

        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            _, dudt = jvp(fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
        target = (v - _b(t - r) * dudt).detach()
        sq = (u - target).pow(2).flatten(1).mean(1)
        weight = (sq.detach() + self.adaptive_eps).pow(self.adaptive_p)
        mf = (sq / weight).mean()
        total = mf.clone()
        endpoint = x0.new_tensor(0.)
        thin = x0.new_tensor(0.)
        seg = x0.new_tensor(0.)
        at_zero = (r == 0)
        if at_zero.any() and (self.endpoint_loss_weight > 0 or self.thin_loss_weight > 0):
            idx = at_zero.nonzero(as_tuple=True)[0]
            xhat = z[idx] - _b(t[idx]) * u[idx]
            if self.endpoint_loss_weight > 0:
                endpoint = F.l1_loss(xhat, x0[idx]) + self._dice(xhat, x0[idx])
                total += self.endpoint_loss_weight * endpoint
            if self.thin_loss_weight > 0:
                thin = self._thin(xhat, x0[idx])
                total += self.thin_loss_weight * thin
        if self.seg_loss_weight > 0 and seg_logits is not None:
            seg = F.binary_cross_entropy_with_logits(seg_logits.float(), (x0 + 1) * .5)
            total += self.seg_loss_weight * seg
        if not torch.isfinite(total):
            raise RuntimeError('non-finite Conference MeanFlow loss')
        logs = {
            key: float(value.detach())
            for key, value in {
                'total_loss': total,
                'mf_loss': mf,
                'endpoint_loss': endpoint,
                'thin_loss': thin,
                'seg_loss': seg,
            }.items()
        }
        logs.update(stage)
        return total, logs
