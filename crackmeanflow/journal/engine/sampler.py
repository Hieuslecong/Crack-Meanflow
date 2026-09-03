from __future__ import annotations
import torch
from crackmeanflow.journal.geometry.rasterizer import GeometryRasterizer

@torch.no_grad()
def sample_geometry_one_step(model,z_geometry,image,rasterizer:GeometryRasterizer):
    b=z_geometry.shape[0]; device=z_geometry.device; r=torch.zeros(b,device=device); t=torch.ones(b,device=device)
    out=model.flow_outputs(z_geometry,t,r,image); geometry=z_geometry-out['u']; return geometry,rasterizer(geometry)
