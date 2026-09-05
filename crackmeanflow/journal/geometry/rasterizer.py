from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .targets import geometry_state_to_fields
from .radius_bins import soft_radius_bins

class GeometryRasterizer(nn.Module):
    def __init__(self,max_radius=16.0,bins=8,temperature=.75,sharpness=3.0,representation='centerline_radius',distance_encoding='linear'):
        super().__init__();self.max_radius=float(max_radius);self.bins=int(bins);self.temperature=float(temperature);self.sharpness=float(sharpness);self.representation=str(representation);self.distance_encoding=str(distance_encoding)
    @staticmethod
    def _disk(radius,device,dtype):
        if radius<=0:return torch.ones(1,1,1,1,device=device,dtype=dtype)
        yy,xx=torch.meshgrid(torch.arange(-radius,radius+1,device=device),torch.arange(-radius,radius+1,device=device),indexing='ij');return ((xx*xx+yy*yy)<=radius*radius).to(dtype)[None,None]
    def forward_fields(self,center_prob,radius):
        if self.representation=='centerline_edt': return 1.0-torch.exp(-self.sharpness*radius.clamp_min(0))
        if self.representation!='centerline_radius':raise ValueError(f'unknown geometry representation={self.representation!r}')
        weights,centers=soft_radius_bins(radius,self.max_radius,self.bins,self.temperature);union=torch.zeros_like(center_prob)
        for k in range(self.bins):
            r=int(round(float(centers[0,k,0,0].detach().cpu())));src=center_prob*weights[:,k:k+1];ker=self._disk(r,src.device,src.dtype);spread=1-torch.exp(-self.sharpness*F.conv2d(src,ker,padding=r));union=1-(1-union)*(1-spread.clamp(0,1))
        return union.clamp(0,1)
    def forward(self,state_pm1):
        c,r=geometry_state_to_fields(state_pm1,self.max_radius,self.representation,self.distance_encoding);return self.forward_fields(c,r)
    @torch.no_grad()
    def hard(self,state_pm1,threshold=.5):return (self.forward(state_pm1)>=threshold).float()
