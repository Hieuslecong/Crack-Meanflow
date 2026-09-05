from __future__ import annotations
import torch

def radius_bin_centers(max_radius,bins,device,dtype):
    return torch.linspace(0,float(max_radius),int(bins),device=device,dtype=dtype).reshape(1,-1,1,1)

def soft_radius_bins(radius,max_radius,bins,temperature=.75):
    centers=radius_bin_centers(max_radius,bins,radius.device,radius.dtype)
    logits=-(radius-centers).abs()/max(float(temperature),1e-6)
    return torch.softmax(logits,dim=1),centers
