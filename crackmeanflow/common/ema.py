from __future__ import annotations

class EMA:
    def __init__(self, model, decay=.999, shadow=None):
        self.decay=float(decay); self.backup=None
        self.shadow={k:v.detach().clone() for k,v in (model.state_dict().items() if shadow is None else shadow.items()) if v.dtype.is_floating_point}
    def update(self, model):
        for k,v in model.state_dict().items():
            if k in self.shadow: self.shadow[k].mul_(self.decay).add_(v.detach(),alpha=1-self.decay)
    def apply(self, model):
        self.backup={k:v.detach().clone() for k,v in model.state_dict().items() if k in self.shadow}; model.load_state_dict({**model.state_dict(),**self.shadow},strict=False)
    def restore(self, model):
        if self.backup is None: raise RuntimeError('EMA.restore called before apply')
        model.load_state_dict({**model.state_dict(),**self.backup},strict=False); self.backup=None
