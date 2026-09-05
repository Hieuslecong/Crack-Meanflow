import torch
from types import SimpleNamespace
from scripts.train_journal import _set_loader_epoch

class S:
    def __init__(self): self.epoch=None
    def set_epoch(self,e): self.epoch=e

def _draw(epoch):
    loader=SimpleNamespace(sampler=S(),generator=torch.Generator())
    _set_loader_epoch(loader,epoch,42)
    return loader.sampler.epoch, torch.randint(0,2**31,(4,),generator=loader.generator).tolist()

def test_loader_epoch_seed_is_directly_reproducible():
    assert _draw(7)==_draw(7)
    assert _draw(7)!=_draw(8)
