from dataclasses import dataclass

import torch
from torch import nn

from .paths import ensure_paths

ensure_paths()
from multi_task.mlt_unet import UNet  # noqa: E402


@dataclass
class CrackMeanFlowConfig:
    T: int = 500


class CrackMeanFlowModel(nn.Module):
    """Adapter from CrackDiff UNet to MeanFlow velocity model interface."""

    def __init__(self, unet: UNet, T: int = 500):
        super().__init__()
        self.unet = unet
        self.T = int(T)
        self.num_classes = 0
        self._last_seg_logits = None

    def _to_batch_time(self, t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=torch.float32)
        t = t.to(device=device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.repeat(batch_size)
        elif t.ndim == 1 and t.shape[0] == 1 and batch_size > 1:
            t = t.repeat(batch_size)
        elif t.ndim != 1:
            t = t.view(batch_size)
        t = t.clamp(0.0, 1.0)
        return torch.round(t * (self.T - 1)).long()

    def clear_seg_logits(self):
        self._last_seg_logits = None

    def get_seg_logits(self):
        return self._last_seg_logits

    def forward(self, x, r, t, y=None, **kwargs):
        del r, kwargs
        if y is None:
            raise ValueError("CrackMeanFlowModel.forward requires conditioning image `y`.")

        t_int = self._to_batch_time(t, batch_size=x.shape[0], device=x.device)
        velocity_pred, seg_logits = self.unet(x, t_int, y)
        self._last_seg_logits = seg_logits
        return velocity_pred

