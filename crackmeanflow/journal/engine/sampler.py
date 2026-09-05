from __future__ import annotations
import torch
@torch.no_grad()
def sample_geometry_one_step(model,z_geometry,image,rasterizer):
    b=z_geometry.shape[0];device=z_geometry.device;r=torch.zeros(b,device=device);t=torch.ones(b,device=device);out=model.flow_outputs(z_geometry,t,r,image);geometry=z_geometry-out['u'];mask_prob=rasterizer(geometry);return geometry,mask_prob
