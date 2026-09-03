"""Deterministic medial-axis targets for GeoCrack-iMF."""
from __future__ import annotations
import math
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

def mask_to_geometry_state(mask_pm1: torch.Tensor, max_radius: float=16.0, representation: str='centerline_radius', distance_encoding: str='linear'):
    """Convert binary mask [-1,1] to two-channel geometry state [-1,1]."""
    mask01=(mask_pm1.detach().float()>0).cpu().numpy(); states=[]; valids=[]
    for sample in mask01:
        m=sample[0].astype(bool); c=skeletonize(m); d=distance_transform_edt(m).astype(np.float32)
        if representation == 'centerline_radius':
            field=(d*c.astype(np.float32)).clip(0,float(max_radius))/float(max_radius)
        elif representation == 'centerline_edt':
            field=d.clip(0,float(max_radius))/float(max_radius)
            if distance_encoding == 'sqrt': field=np.sqrt(field)
            elif distance_encoding == 'log1p': field=np.log1p(d.clip(0,float(max_radius)))/np.log1p(float(max_radius))
            elif distance_encoding != 'linear': raise ValueError(f'unknown distance_encoding={distance_encoding!r}')
        else: raise ValueError(f'unknown geometry representation={representation!r}')
        s=np.stack([c.astype(np.float32)*2-1, field*2-1],axis=0)
        states.append(torch.from_numpy(s)); valids.append(torch.from_numpy(c.astype(np.float32))[None])
    return torch.stack(states).to(mask_pm1.device,dtype=torch.float32), torch.stack(valids).to(mask_pm1.device,dtype=torch.float32)

def geometry_state_to_fields(state_pm1: torch.Tensor, max_radius: float=16.0, representation: str='centerline_radius', distance_encoding: str='linear'):
    center=((state_pm1[:,0:1]+1)*.5).clamp(0,1); q=((state_pm1[:,1:2]+1)*.5).clamp(0,1)
    if representation == 'centerline_edt':
        if distance_encoding == 'sqrt': q=q.square()
        elif distance_encoding == 'log1p': q=torch.expm1(q*math.log1p(float(max_radius)))/float(max_radius)
        elif distance_encoding != 'linear': raise ValueError(f'unknown distance_encoding={distance_encoding!r}')
    return center,q*float(max_radius)
