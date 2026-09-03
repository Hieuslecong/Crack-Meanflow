"""GeoCrack-iMF: hybrid local/global conditional Transformer.

The network predicts clean geometry twice (u-head and v-head), then applies the
same pixel-MeanFlow clean-output parameterization used by official pMF:
    velocity = (z_t - clean_prediction) / clip(t, 0.05, 1).
The final A5 scientific state is joint centerline + dense interior-distance geometry; A3/A4 retain centerline-radius as ablations. The mask is decoded through the representation-specific differentiable rasterizer.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from crackmeanflow.sit import Attention, Block, RTEmbedder

def _b(t): return t.reshape(-1,1,1,1)

class GeoCrackIMFModel(nn.Module):
    PRESETS={'S':dict(dim=384,depth=10,heads=6),'T':dict(dim=256,depth=8,heads=4)}
    def __init__(self,img_size=256,patch=8,size='S',geometry_ch=2,cond_ch=3,local_refine=False,background_init=-0.95):
        super().__init__(); cfg=self.PRESETS[size]; d=cfg['dim']; self.patch=int(patch); self.img_size=int(img_size); self.geometry_ch=int(geometry_ch); self.local_refine=bool(local_refine)
        self.background_init=float(background_init)
        if not (-0.999 < self.background_init < 0.999): raise ValueError('background_init must lie strictly inside (-0.999,0.999)')
        init_bias=float(math.atanh(self.background_init))
        if img_size%patch: raise ValueError('img_size must be divisible by patch')
        self.grid=img_size//patch; n=self.grid**2
        self.cond_local=nn.Sequential(nn.Conv2d(cond_ch,64,3,1,1),nn.SiLU(),nn.Conv2d(64,64,3,1,1),nn.SiLU())
        self.state_local=nn.Sequential(nn.Conv2d(geometry_ch,64,3,1,1),nn.SiLU(),nn.Conv2d(64,64,3,1,1),nn.SiLU())
        self.cond_patch=nn.Conv2d(64,d,patch,stride=patch)
        self.state_patch=nn.Conv2d(64,d,patch,stride=patch)
        self.pos=nn.Parameter(torch.zeros(1,n,d)); nn.init.trunc_normal_(self.pos,std=.02)
        self.rt=RTEmbedder(d); self.blocks=nn.ModuleList([Block(d,cfg['heads']) for _ in range(cfg['depth'])])
        self.norm=nn.LayerNorm(d,elementwise_affine=False,eps=1e-6); self.ada=nn.Sequential(nn.SiLU(),nn.Linear(d,2*d))
        self.clean_u=None; self.clean_v=None
        if self.local_refine:
            self.global_local=nn.Sequential(nn.Conv2d(d,64,1),nn.SiLU())
            self.fuse=nn.Sequential(nn.Conv2d(64*3,128,3,1,1),nn.SiLU(),nn.Conv2d(128,128,3,1,1),nn.SiLU())
            self.film=nn.Sequential(nn.SiLU(),nn.Linear(d,256))
            self.clean_u_local=nn.Conv2d(128,geometry_ch,3,1,1); self.clean_v_local=nn.Conv2d(128,geometry_ch,3,1,1)
            for h in (self.clean_u_local,self.clean_v_local):
                nn.init.zeros_(h.weight); nn.init.constant_(h.bias,init_bias)
        else:
            self.clean_u=nn.Linear(d,patch*patch*geometry_ch); self.clean_v=nn.Linear(d,patch*patch*geometry_ch)
            for h in (self.clean_u,self.clean_v):
                nn.init.zeros_(h.weight); nn.init.constant_(h.bias,init_bias)
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)
    def _unpatch(self,x):
        B=x.shape[0]; g,p,c=self.grid,self.patch,self.geometry_ch
        return x.reshape(B,g,g,p,p,c).permute(0,5,1,3,2,4).reshape(B,c,g*p,g*p)
    def _features(self,z,t,r,image):
        cond_local=self.cond_local(image); state_local=self.state_local(z)
        cond=self.cond_patch(cond_local); st=self.state_patch(state_local)
        h=(st+cond).flatten(2).transpose(1,2)+self.pos; c=self.rt(r,t)
        for blk in self.blocks: h=blk(h,c)
        s,b=self.ada(c).chunk(2,-1); h=self.norm(h)*(1+s[:,None])+b[:,None]
        return h,c,cond_local,state_local
    def clean_predictions(self,z,t,r,image):
        h,c,cond_local,state_local=self._features(z,t,r,image)
        if not self.local_refine:
            return torch.tanh(self._unpatch(self.clean_u(h))), torch.tanh(self._unpatch(self.clean_v(h)))
        B=h.shape[0]; g=self.grid; d=h.shape[-1]
        glob=h.transpose(1,2).reshape(B,d,g,g)
        glob=torch.nn.functional.interpolate(self.global_local(glob),size=(self.img_size,self.img_size),mode='bilinear',align_corners=False)
        f=self.fuse(torch.cat([cond_local,state_local,glob],dim=1))
        shift,scale=self.film(c).chunk(2,-1); f=f*(1+scale[:,:,None,None])+shift[:,:,None,None]
        return torch.tanh(self.clean_u_local(f)),torch.tanh(self.clean_v_local(f))
    @staticmethod
    def clean_to_velocity(z,clean,t,min_t=.05): return (z-clean)/_b(t.clamp(min=float(min_t)))
    def flow_outputs(self,z,t,r,image):
        cu,cv=self.clean_predictions(z,t,r,image); u=self.clean_to_velocity(z,cu,t); v=self.clean_to_velocity(z,cv,t)
        return {'u':u,'v':v,'clean_u':cu,'clean_v':cv}
    def forward(self,z,r,t,y=None,**kwargs):
        if y is None: raise ValueError('GeoCrackIMFModel requires RGB condition y')
        return self.flow_outputs(z,t,r,y)['u']

def build_geocrack_imf(cfg):
    return GeoCrackIMFModel(img_size=cfg.get('img_size',256),patch=cfg.get('patch',8),size=cfg.get('size','S'),geometry_ch=2,cond_ch=3,local_refine=cfg.get('local_refine',False),background_init=cfg.get('background_init',-0.95))
