from __future__ import annotations
import torch

class EMA:
    def __init__(self, model, decay=.999, shadow=None):
        self.decay=float(decay); self.backup=None
        current=model.state_dict(); expected={k:v for k,v in current.items() if torch.is_tensor(v) and v.dtype.is_floating_point}
        if shadow is None:
            self.shadow={k:v.detach().clone() for k,v in expected.items()}
        else:
            if not isinstance(shadow,dict): raise RuntimeError('EMA shadow must be a state dictionary')
            missing=sorted(set(expected)-set(shadow)); unexpected=sorted(set(shadow)-set(current))
            if missing or unexpected:
                raise RuntimeError(f'EMA state mismatch: missing={missing[:10]} ({len(missing)}), unexpected={unexpected[:10]} ({len(unexpected)})')
            self.shadow={}
            for k,v in expected.items():
                sv=shadow[k]
                if not torch.is_tensor(sv) or tuple(sv.shape)!=tuple(v.shape):
                    raise RuntimeError(f'EMA tensor mismatch for {k}: checkpoint_shape={getattr(sv,"shape",None)} model_shape={tuple(v.shape)}')
                self.shadow[k]=sv.detach().to(device=v.device,dtype=v.dtype).clone()
    def update(self, model):
        for k,v in model.state_dict().items():
            if k in self.shadow: self.shadow[k].mul_(self.decay).add_(v.detach(),alpha=1-self.decay)
    def apply(self, model):
        self.backup={k:v.detach().clone() for k,v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({**model.state_dict(),**self.shadow},strict=True)
    def restore(self, model):
        if self.backup is None: raise RuntimeError('EMA.restore called before apply')
        model.load_state_dict({**model.state_dict(),**self.backup},strict=True); self.backup=None
