"""CrackMeanFlow adapter with continuous (r,t) conditioning."""
import math
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class CrackMeanFlowConfig:
    T:int=500

def _sinusoidal(v:torch.Tensor,dim:int)->torch.Tensor:
    half=dim//2
    freqs=torch.exp(-math.log(10000.0)*torch.arange(half,device=v.device,dtype=torch.float32)/half)
    args=v.float()[:,None]*freqs[None,:]*1000.0
    return torch.cat([torch.sin(args),torch.cos(args)],dim=-1)

class ContinuousRTEmbedding(nn.Module):
    def __init__(self,ch:int,tdim:int):
        super().__init__(); self.freq_dim=max(32,ch)
        self.mlp=nn.Sequential(nn.Linear(self.freq_dim*2,tdim),nn.SiLU(),nn.Linear(tdim,tdim))
        for m in self.mlp:
            if isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,rt):
        r,t=rt[:,0],rt[:,1]
        emb=torch.cat([_sinusoidal(t,self.freq_dim),_sinusoidal(r,self.freq_dim)],dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))

class CrackMeanFlowModel(nn.Module):
    """Adapter from the historical multi-task U-Net to the MeanFlow velocity interface.

    The historical U-Net contract is strictly:
      (velocity/noisy_head, segmentation_logits)
    Do not swap these outputs: doing so trains the segmentation head as the MeanFlow
    velocity field and silently corrupts the Conference baseline.
    """
    def __init__(self,unet,T:int=500,ch:int=None):
        super().__init__(); self.unet=unet; self.T=int(T); self.num_classes=0; self._last_seg_logits=None
        if hasattr(unet,'time_embedding'):
            if ch is None: ch=unet.x_head.out_channels
            self.unet.time_embedding=ContinuousRTEmbedding(ch=ch,tdim=ch*4)
    def clear_seg_logits(self): self._last_seg_logits=None
    def get_seg_logits(self): return self._last_seg_logits
    @staticmethod
    def _as_batch(v,batch_size,device,dtype):
        if not torch.is_tensor(v): v=torch.tensor(v,device=device,dtype=dtype)
        v=v.to(device=device,dtype=dtype)
        if v.ndim==0: v=v.expand(batch_size)
        elif v.ndim==1 and v.shape[0]==1 and batch_size>1: v=v.expand(batch_size)
        return v.reshape(batch_size)
    def forward(self,x,r,t,y=None,**kwargs):
        if y is None: raise ValueError('CrackMeanFlowModel.forward requires conditioning image `y`.')
        kwargs.pop('mask_gt',None)
        if hasattr(self.unet,'downblocks'):
            factor=2**(sum(1 for m in self.unet.downblocks if type(m).__name__=='DownSample'))
            if factor>1 and (x.shape[-1]%factor or x.shape[-2]%factor):
                raise ValueError(f'input {tuple(x.shape[-2:])} must be divisible by {factor} for this UNet')
        b=x.shape[0]
        r=self._as_batch(r,b,x.device,x.dtype); t=self._as_batch(t,b,x.device,x.dtype); rt=torch.stack([r,t],dim=-1)
        velocity,seg_logits=self.unet(x,rt,y)
        self._last_seg_logits=seg_logits
        return velocity
