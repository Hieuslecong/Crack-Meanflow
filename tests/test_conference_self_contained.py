import torch
from torch import nn
from crackmeanflow.adapter import CrackMeanFlowModel
from crackmeanflow.conference.models import build_conference_model

def test_adapter_preserves_historical_output_order():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__();self.p=nn.Parameter(torch.tensor(0.));self.x_head=nn.Conv2d(1,32,1);self.time_embedding=nn.Identity();self.downblocks=[]
        def forward(self,x,rt,y):return torch.ones_like(x)*2,torch.ones_like(x)*7
    m=CrackMeanFlowModel(Dummy(),T=500,ch=32);x=torch.zeros(2,1,8,8);y=torch.zeros(2,3,8,8);v=m(x,torch.zeros(2),torch.ones(2),y=y)
    assert torch.all(v==2);assert torch.all(m.get_seg_logits()==7)

def test_conference_is_self_contained_and_runs():
    cfg={'T':500,'ch':64,'ch_mult':[1,2,2,4],'attn':[2],'num_res_blocks':2,'dropout':0.0}
    m=build_conference_model(cfg).eval();x=torch.randn(1,1,32,32);y=torch.rand(1,3,32,32)
    with torch.no_grad():v=m(x,torch.zeros(1),torch.ones(1),y=y)
    assert v.shape==x.shape;assert m.get_seg_logits().shape==x.shape


def test_conference_cleanroom_parameter_count():
    cfg={'T':500,'ch':64,'ch_mult':[1,2,2,4],'attn':[2],'num_res_blocks':2,'dropout':0.0}
    m=build_conference_model(cfg)
    assert sum(p.numel() for p in m.parameters()) == 30_839_346
