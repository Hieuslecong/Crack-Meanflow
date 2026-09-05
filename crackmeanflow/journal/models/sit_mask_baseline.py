from __future__ import annotations
import math
import torch
from torch import nn
from crackmeanflow.sit import Block,RTEmbedder

def _b(t):return t.reshape(-1,1,1,1)
class MaskIMFSiTModel(nn.Module):
    PRESETS={'T':dict(dim=256,depth=8,heads=4),'S':dict(dim=384,depth=10,heads=6)}
    def __init__(self,img_size=256,patch=8,size='T',background_init=-.95):
        super().__init__();cfg=self.PRESETS[size];d=cfg['dim'];self.patch=patch;self.grid=img_size//patch
        if img_size%patch:raise ValueError('img_size must be divisible by patch')
        n=self.grid**2;self.x=nn.Conv2d(1,d,patch,stride=patch);self.y=nn.Conv2d(3,d,patch,stride=patch);self.pos=nn.Parameter(torch.zeros(1,n,d));nn.init.trunc_normal_(self.pos,std=.02);self.rt=RTEmbedder(d);self.blocks=nn.ModuleList([Block(d,cfg['heads']) for _ in range(cfg['depth'])]);self.norm=nn.LayerNorm(d,elementwise_affine=False);self.hu=nn.Linear(d,patch*patch);self.hv=nn.Linear(d,patch*patch)
        if not(-.999<float(background_init)<.999):raise ValueError('background_init must lie inside (-0.999,0.999)')
        b=float(math.atanh(float(background_init)));nn.init.zeros_(self.hu.weight);nn.init.constant_(self.hu.bias,b);nn.init.zeros_(self.hv.weight);nn.init.constant_(self.hv.bias,b)
    def _unpatch(self,a):
        B=a.shape[0];g,p=self.grid,self.patch;return a.reshape(B,g,g,p,p,1).permute(0,5,1,3,2,4).reshape(B,1,g*p,g*p)
    def clean_predictions(self,z,t,r,image):
        h=(self.x(z)+self.y(image)).flatten(2).transpose(1,2)+self.pos;c=self.rt(r,t)
        for blk in self.blocks:h=blk(h,c)
        h=self.norm(h);return torch.tanh(self._unpatch(self.hu(h))),torch.tanh(self._unpatch(self.hv(h)))
    def flow_outputs(self,z,t,r,image):
        cu,cv=self.clean_predictions(z,t,r,image);den=_b(t.clamp(min=.05));return {'u':(z-cu)/den,'v':(z-cv)/den,'clean_u':cu,'clean_v':cv}
    def forward(self,z,r,t,y=None,**kw):return self.flow_outputs(z,t,r,y)['u']
class HybridMaskIMFModel(nn.Module):
    def __init__(self,img_size=256,patch=8,size='S',background_init=-.95):
        super().__init__();from .geocrack_imf import GeoCrackIMFModel;self.core=GeoCrackIMFModel(img_size=img_size,patch=patch,size=size,geometry_ch=1,cond_ch=3,local_refine=True,background_init=background_init)
    def flow_outputs(self,z,t,r,image):return self.core.flow_outputs(z,t,r,image)
    def forward(self,z,r,t,y=None,**kw):return self.core(z,r,t,y,**kw)
