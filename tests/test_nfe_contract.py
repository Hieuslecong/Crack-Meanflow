import torch
from torch import nn
from crackmeanflow.sampler import crack_meanflow_sampler
class M(nn.Module):
    def __init__(self):super().__init__();self.p=nn.Parameter(torch.tensor(0.));self.calls=0;self._last_seg_logits=None
    def forward(self,x,r,t,y=None):self.calls+=1;self._last_seg_logits=x;return torch.zeros_like(x)
    def get_seg_logits(self):return self._last_seg_logits

def test_nfe1_calls_model_once():
    m=M();z=torch.randn(1,1,8,8);y=torch.randn(1,3,8,8);crack_meanflow_sampler(m,z,y,num_steps=1);assert m.calls==1
