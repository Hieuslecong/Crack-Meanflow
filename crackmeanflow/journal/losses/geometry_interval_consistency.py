from __future__ import annotations
import torch
from crackmeanflow.journal.geometry.targets import geometry_state_to_fields

def _b(v): return v.reshape(-1,1,1,1)

def geometry_interval_consistency(model,z,r,t,image,rasterizer,mask_weight=.5,radius_weight=.5):
    """Training-only clean-endpoint interval consistency."""
    del r
    r0=torch.zeros_like(t); s=t*.5
    u_long=model.flow_outputs(z,t,r0,image)['u']; g_long=z-_b(t)*u_long
    u_ts=model.flow_outputs(z,t,s,image)['u']; g_s=z-_b(t-s)*u_ts
    u_s0=model.flow_outputs(g_s,s,r0,image)['u']; g_short=g_s-_b(s)*u_s0
    c_long,r_long=geometry_state_to_fields(g_long,rasterizer.max_radius,rasterizer.representation,rasterizer.distance_encoding)
    c_short,r_short=geometry_state_to_fields(g_short,rasterizer.max_radius,rasterizer.representation,rasterizer.distance_encoding)
    eps=1e-6; c_support=torch.maximum(c_long,c_short).detach(); c_denom=c_support.sum().clamp_min(1.)
    center_l1=((c_long-c_short).abs()*c_support).sum()/c_denom
    center_dice=1-(2*(c_long*c_short).sum()+eps)/((c_long.square()).sum()+(c_short.square()).sum()+eps)
    center=.5*(center_l1+center_dice)
    mask_long=rasterizer(g_long); mask_short=rasterizer(g_short)
    support=torch.maximum(mask_long,mask_short).detach() if rasterizer.representation == 'centerline_edt' else torch.maximum(c_long,c_short).detach()
    denom=support.sum().clamp_min(1.); radius=((r_long-r_short).abs()*support).sum()/denom
    mask_l1=((mask_long-mask_short).abs()*support).sum()/denom
    mask_dice=1-(2*(mask_long*mask_short).sum()+eps)/((mask_long.square()).sum()+(mask_short.square()).sum()+eps)
    mask=.5*(mask_l1+mask_dice)
    total=center+float(radius_weight)*radius/max(float(rasterizer.max_radius),1e-6)+float(mask_weight)*mask
    return total,{'gic_center':center,'gic_radius':radius,'gic_mask':mask}
