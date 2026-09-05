import pytest,torch
from torch import nn
from crackmeanflow.common.ema import EMA

def test_ema_resume_rejects_missing_and_shape_mismatch():
    m=nn.Sequential(nn.Linear(3,4),nn.Linear(4,1));good={k:v.detach().clone() for k,v in m.state_dict().items() if v.dtype.is_floating_point}
    bad=dict(good);bad.pop(next(iter(bad)))
    with pytest.raises(RuntimeError,match='EMA state mismatch'):EMA(m,shadow=bad)
    bad=dict(good);k=next(iter(bad));bad[k]=torch.zeros(1)
    with pytest.raises(RuntimeError,match='EMA tensor mismatch'):EMA(m,shadow=bad)

def test_ema_resume_converts_to_model_dtype_and_updates():
    m=nn.Linear(3,2).double();shadow={k:v.detach().float().clone() for k,v in m.state_dict().items() if v.dtype.is_floating_point};e=EMA(m,shadow=shadow)
    assert all(v.dtype==torch.float64 for v in e.shadow.values());e.update(m)
