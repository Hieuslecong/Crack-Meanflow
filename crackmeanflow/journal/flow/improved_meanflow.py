"""Improved MeanFlow objective adapted to crack-geometry state transport.

Source-of-truth mapping: official Lyy-iiis/imeanflow `imf.py`.
We omit class-CFG because the condition is a dense RGB image; the core iMF change
is retained exactly: regress V = u + (t-r)*sg(du/dt) to instantaneous velocity,
with an auxiliary v-head. The JVP tangent uses the predicted conditional v field.
The clean-space->velocity parameterization follows official pMF.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch.func import jvp
except ImportError:  # pragma: no cover
    from functorch import jvp
from crackmeanflow.journal.losses.geometry_interval_consistency import geometry_interval_consistency

def _b(v): return v.reshape(-1,1,1,1)

def _stratified_mask(batch, fraction, offset, device):
    fraction=float(fraction)
    if fraction<=0: return torch.zeros(batch,dtype=torch.bool,device=device)
    if fraction>=1: return torch.ones(batch,dtype=torch.bool,device=device)
    idx=torch.arange(int(offset),int(offset)+batch,device=device,dtype=torch.float64)
    return torch.floor((idx+1)*fraction)>torch.floor(idx*fraction)

def _bce_dice(prob,target,eps=1e-6):
    prob=prob.clamp(eps,1-eps); target=target.float(); bce=F.binary_cross_entropy(prob,target)
    dims=tuple(range(1,prob.ndim)); inter=(prob*target).sum(dims); den=prob.sum(dims)+target.sum(dims)
    return bce+(1-(2*inter+eps)/(den+eps)).mean()

def _balanced_bce_dice(prob,target,eps=1e-6,max_pos_weight=50.0):
    prob=prob.clamp(eps,1-eps); target=target.float(); dims=tuple(range(1,prob.ndim))
    pos=target.sum(dims); neg=(1-target).sum(dims); posw=(neg/(pos+eps)).clamp(1.0,float(max_pos_weight))
    shape=(prob.shape[0],)+((1,)*(prob.ndim-1)); w=1+target*(posw.reshape(shape)-1)
    bce=(-(w*(target*torch.log(prob)+(1-target)*torch.log1p(-prob))).flatten(1).sum(1)/w.flatten(1).sum(1).clamp_min(eps)).mean()
    inter=(prob*target).sum(dims); den=prob.sum(dims)+target.sum(dims); dice=(1-(2*inter+eps)/(den+eps)).mean()
    return bce+dice

class ImprovedMeanFlowStateLoss(nn.Module):
    def __init__(self,data_proportion=.5,time_mu=-.4,time_sigma=1.,norm_p=1.,norm_eps=.01,clean_weight=.25,fm_sampling='stratified'):
        super().__init__(); self.data_proportion=float(data_proportion); self.time_mu=float(time_mu); self.time_sigma=float(time_sigma); self.norm_p=float(norm_p); self.norm_eps=float(norm_eps); self.clean_weight=float(clean_weight); self.fm_sampling=str(fm_sampling)
    def _sample(self,b,device,sample_offset=0):
        n=torch.randn(2,b,device=device)*self.time_sigma+self.time_mu; a,bb=torch.sigmoid(n[0]),torch.sigmoid(n[1]); t=torch.maximum(a,bb); r=torch.minimum(a,bb)
        fm=_stratified_mask(b,self.data_proportion,sample_offset,device) if self.fm_sampling=='stratified' else (torch.rand(b,device=device)<self.data_proportion)
        return t,torch.where(fm,t,r),fm
    def _adp(self,l): return (l/(l.detach()+self.norm_eps).pow(self.norm_p)).mean()
    def forward(self,model,x0,image,sample_offset=0):
        b=x0.shape[0]; t,r,fm=self._sample(b,x0.device,sample_offset); e=torch.randn_like(x0); z=(1-_b(t))*x0+_b(t)*e; vt=e-x0
        out=model.flow_outputs(z,t,r,image); u,v,clean=out['u'],out['v'],out['clean_u']
        with torch.no_grad():
            vc=model.flow_outputs(z,t,t,image)['v']
            def fn(z_,t_,r_): return model.flow_outputs(z_,t_,r_,image)['u']
            _,du=jvp(fn,(z,t,r),(vc,torch.ones_like(t),torch.zeros_like(r)))
        V=u+_b(t-r)*du.detach(); lu=self._adp((V-vt.detach()).pow(2).flatten(1).sum(1)); lv=self._adp((v-vt.detach()).pow(2).flatten(1).sum(1))
        clean_prob=((clean+1)*.5).clamp(0,1); target_prob=((x0+1)*.5).clamp(0,1); clean_loss=_bce_dice(clean_prob,target_prob); total=lu+lv+self.clean_weight*clean_loss
        return total,{'total_loss':float(total.detach()),'imf_u_loss':float(lu.detach()),'imf_v_loss':float(lv.detach()),'clean_loss':float(clean_loss.detach()),'fm_count':int(fm.sum().item()),'gic_count':0}

class ImprovedMeanFlowGeometryLoss(nn.Module):
    def __init__(self, data_proportion=.5, time_mu=-.4, time_sigma=1., norm_p=1., norm_eps=.01, geometry_weight=.5, mask_weight=.5, radius_weight=1., gic_weight=.1, gic_probability=.25, max_radius=16., rasterizer=None, fm_sampling='stratified', gic_sampling='stratified'):
        super().__init__(); self.data_proportion=float(data_proportion); self.time_mu=float(time_mu); self.time_sigma=float(time_sigma); self.norm_p=float(norm_p); self.norm_eps=float(norm_eps); self.geometry_weight=float(geometry_weight); self.mask_weight=float(mask_weight); self.radius_weight=float(radius_weight); self.gic_weight=float(gic_weight); self.gic_probability=float(gic_probability); self.max_radius=float(max_radius); self.rasterizer=rasterizer; self.fm_sampling=str(fm_sampling); self.gic_sampling=str(gic_sampling)
    def sample_tr(self,b,device,sample_offset=0):
        n=torch.randn(2,b,device=device)*self.time_sigma+self.time_mu; a,bb=torch.sigmoid(n[0]),torch.sigmoid(n[1]); t=torch.maximum(a,bb); r=torch.minimum(a,bb)
        fm=_stratified_mask(b,self.data_proportion,sample_offset,device) if self.fm_sampling=='stratified' else (torch.rand(b,device=device)<self.data_proportion); r=torch.where(fm,t,r); return t,r,fm
    def _adaptive(self,per_sample): return (per_sample/(per_sample.detach()+self.norm_eps).pow(self.norm_p)).mean()
    def forward(self,model,g0,image,radius_valid=None,mask_gt=None,sample_offset=0):
        b,device=g0.shape[0],g0.device; t,r,fm=self.sample_tr(b,device,sample_offset); eps=torch.randn_like(g0); z=(1-_b(t))*g0+_b(t)*eps; v_t=eps-g0
        main=model.flow_outputs(z,t,r,image); u=main['u']; v=main['v']; clean=main['clean_u']
        with torch.no_grad():
            v_c=model.flow_outputs(z,t,t,image)['v']
            def fn(z_,t_,r_): return model.flow_outputs(z_,t_,r_,image)['u']
            _,du_dt=jvp(fn,(z,t,r),(v_c,torch.ones_like(t),torch.zeros_like(r)))
        V=u+_b(t-r)*du_dt.detach(); target=v_t.detach(); lu=(V-target).pow(2).flatten(1).sum(1); lv=(v-target).pow(2).flatten(1).sum(1)
        loss_u=self._adaptive(lu); loss_v=self._adaptive(lv); total=loss_u+loss_v
        c_pred=((clean[:,0:1]+1)*.5).clamp(0,1); c_gt=((g0[:,0:1]+1)*.5).clamp(0,1); c_loss=_balanced_bce_dice(c_pred,c_gt)
        if self.rasterizer is not None and getattr(self.rasterizer,'representation','centerline_radius') == 'centerline_edt' and mask_gt is not None: field_valid=(mask_gt>0).float()
        else: field_valid=radius_valid if radius_valid is not None else ((g0[:,0:1]+1)*.5>0.5).float()
        denom=field_valid.sum().clamp_min(1.); r_loss=((clean[:,1:2]-g0[:,1:2]).abs()*field_valid).sum()/denom; geom=c_loss+self.radius_weight*r_loss; total=total+self.geometry_weight*geom
        mask_loss=g0.new_tensor(0.)
        if self.mask_weight>0 and self.rasterizer is not None:
            pred_mask=self.rasterizer(clean); gt_mask=((mask_gt+1)*.5).clamp(0,1) if mask_gt is not None else self.rasterizer(g0).detach(); mask_loss=_bce_dice(pred_mask,gt_mask); total=total+self.mask_weight*mask_loss
        gic=g0.new_tensor(0.); parts={}; gic_on=bool(_stratified_mask(1,self.gic_probability,sample_offset,device)[0].item()) if self.gic_sampling=='stratified' else bool((torch.rand((),device=device)<self.gic_probability).item())
        if self.gic_weight>0 and self.rasterizer is not None and gic_on:
            gic,parts=geometry_interval_consistency(model,z,r,t,image,self.rasterizer); total=total+self.gic_weight*gic
        if not torch.isfinite(total): raise RuntimeError('non-finite GeoCrack-iMF loss')
        logs={'total_loss':total,'imf_u_loss':loss_u,'imf_v_loss':loss_v,'geometry_loss':geom,'centerline_loss':c_loss,'field_loss':r_loss,'radius_loss':r_loss,'mask_loss':mask_loss,'gic_loss':gic,'fm_count':fm.sum(),'gic_count':g0.new_tensor(1 if (self.gic_weight>0 and self.rasterizer is not None and gic_on) else 0)}; logs.update(parts)
        return total,{k:float(vv.detach()) for k,vv in logs.items()}
