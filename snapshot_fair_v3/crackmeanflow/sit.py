"""Compact SiT backbone for CrackMeanFlow-SiT (the Paper-B model)."""
import math
import torch
from torch import nn

def _sinusoidal(v: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=v.device, dtype=torch.float32) / half)
    args = v.float()[:, None] * freqs[None, :] * 1000.0
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class RTEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 128):
        super().__init__(); self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, r, t):
        e = torch.cat([_sinusoidal(t, self.freq_dim), _sinusoidal(r, self.freq_dim)], dim=-1)
        return self.mlp(e.to(self.mlp[0].weight.dtype))

class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads; self.dh = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True); self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B,N,C=x.shape; qkv=self.qkv(x).reshape(B,N,3,self.h,self.dh).permute(2,0,3,1,4); q,k,v=qkv[0],qkv[1],qkv[2]
        att=(q@k.transpose(-2,-1))*(self.dh**-0.5); att=att.softmax(dim=-1); out=(att@v).transpose(1,2).reshape(B,N,C)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__(); self.n1=nn.LayerNorm(dim,elementwise_affine=False,eps=1e-6); self.attn=Attention(dim,heads)
        self.n2=nn.LayerNorm(dim,elementwise_affine=False,eps=1e-6); hidden=int(dim*mlp_ratio)
        self.mlp=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(approximate='tanh'),nn.Linear(hidden,dim)); self.ada=nn.Sequential(nn.SiLU(),nn.Linear(dim,6*dim))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)
    def forward(self,x,c):
        sa,ba,ga,sm,bm,gm=self.ada(c).chunk(6,dim=-1); h=self.n1(x)*(1+sa[:,None])+ba[:,None]; x=x+ga[:,None]*self.attn(h)
        h=self.n2(x)*(1+sm[:,None])+bm[:,None]; return x+gm[:,None]*self.mlp(h)

class SiTBackbone(nn.Module):
    PRESETS={'S':dict(dim=384,depth=12,heads=6),'B':dict(dim=768,depth=12,heads=12),'L':dict(dim=1024,depth=24,heads=16)}
    def __init__(self,img_size=256,patch=8,size='S',in_ch=1,cond_ch=3):
        super().__init__(); cfg=self.PRESETS[size]; dim,depth,heads=cfg['dim'],cfg['depth'],cfg['heads']; self.patch,self.img_size,self.in_ch=patch,img_size,in_ch; self.grid=img_size//patch; n=self.grid**2
        self.x_embed=nn.Conv2d(in_ch,dim,patch,stride=patch); self.y_embed=nn.Conv2d(cond_ch,dim,patch,stride=patch); self.pos=nn.Parameter(torch.zeros(1,n,dim)); nn.init.trunc_normal_(self.pos,std=.02)
        self.rt=RTEmbedder(dim); self.blocks=nn.ModuleList([Block(dim,heads) for _ in range(depth)]); self.norm=nn.LayerNorm(dim,elementwise_affine=False,eps=1e-6)
        self.ada_out=nn.Sequential(nn.SiLU(),nn.Linear(dim,2*dim)); nn.init.zeros_(self.ada_out[-1].weight); nn.init.zeros_(self.ada_out[-1].bias)
        self.head_seg=nn.Linear(dim,patch*patch*in_ch); self.head_vel=nn.Linear(dim,patch*patch*in_ch)
        for h in (self.head_seg,self.head_vel): nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
    def _unpatch(self,x):
        B=x.shape[0]; g,p,c=self.grid,self.patch,self.in_ch; return x.reshape(B,g,g,p,p,c).permute(0,5,1,3,2,4).reshape(B,c,g*p,g*p)
    def forward(self,x_t,rt,image):
        r,t=rt[:,0],rt[:,1]; h=(self.x_embed(x_t)+self.y_embed(image)).flatten(2).transpose(1,2)+self.pos; c=self.rt(r,t)
        for blk in self.blocks: h=blk(h,c)
        s,b=self.ada_out(c).chunk(2,dim=-1); h=self.norm(h)*(1+s[:,None])+b[:,None]
        return self._unpatch(self.head_seg(h)),self._unpatch(self.head_vel(h))

def build_sit(img_size=256,patch=8,size='S',in_ch=1,cond_ch=3): return SiTBackbone(img_size=img_size,patch=patch,size=size,in_ch=in_ch,cond_ch=cond_ch)
