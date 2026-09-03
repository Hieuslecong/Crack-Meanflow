import torch

def radius_bin_centers(max_radius:float=16.0, bins:int=8, device=None, dtype=torch.float32):
    if bins<2: raise ValueError('bins must be >=2')
    return torch.linspace(0.,float(max_radius),int(bins),device=device,dtype=dtype)

def soft_radius_bins(radius, max_radius=16.0, bins=8, temperature=.75):
    centers=radius_bin_centers(max_radius,bins,radius.device,radius.dtype).view(1,-1,1,1)
    logits=-((radius-centers)/max(float(temperature),1e-4)).pow(2)
    return torch.softmax(logits,dim=1), centers
