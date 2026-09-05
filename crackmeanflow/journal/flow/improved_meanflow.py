from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
try:
    from torch.func import jvp
except ImportError:
    from functorch import jvp
from crackmeanflow.journal.losses.geometry_interval_consistency import geometry_interval_consistency

def _b(v):return v.reshape(-1,1,1,1)

def _stratified_mask(batch,fraction,offset,device):
    fraction=float(fraction)
    if fraction<=0:return torch.zeros(batch,dtype=torch.bool,device=device)
    if fraction>=1:return torch.ones(batch,dtype=torch.bool,device=device)
    idx=torch.arange(int(offset),int(offset)+batch,device=device,dtype=torch.float64)
    return torch.floor((idx+1)*fraction)>torch.floor(idx*fraction)

def _independent_stratified_mask(batch,fraction,offset,device,stream='gic',block=1000):
    """Deterministic exact-rate block stratification with decorrelated streams.

    This avoids the nested-period aliasing of reusing `_stratified_mask` for FM,
    GIC and endpoint selection (e.g. p=.25 becomes a strict subset of p=.50).
    Canonical probabilities .25 and .15 are represented exactly in 1000-sample blocks.
    """
    fraction=float(fraction)
    if fraction<=0:return torch.zeros(batch,dtype=torch.bool,device=device)
    if fraction>=1:return torch.ones(batch,dtype=torch.bool,device=device)
    params={'gic':(37,137),'endpoint':(73,421)}
    if stream not in params:raise ValueError(f'unknown stratified stream={stream!r}')
    mult,shift=params[stream]
    idx=torch.arange(int(offset),int(offset)+batch,device=device,dtype=torch.int64)
    pos=torch.remainder(idx,int(block));bid=torch.div(idx,int(block),rounding_mode='floor')
    perm=torch.remainder(pos*int(mult)+int(shift)+bid*97,int(block))
    k=int(round(fraction*int(block)))
    return perm<k


def _disjoint_stratified_schedule(batch,fm_fraction,endpoint_fraction,offset,device,block=1000):
    """Deterministic disjoint endpoint/FM/interval schedule.

    Endpoint-aware ablations must not silently reduce the configured FM fraction.
    In each full block this assigns exactly endpoint_fraction to deployment
    endpoint, fm_fraction to r=t, and the remainder to interval samples.
    """
    fm_fraction=float(fm_fraction); endpoint_fraction=float(endpoint_fraction)
    if fm_fraction<0 or endpoint_fraction<0 or fm_fraction+endpoint_fraction>1+1e-12:
        raise ValueError('fm_fraction and endpoint_fraction must be >=0 and sum to <=1')
    idx=torch.arange(int(offset),int(offset)+batch,device=device,dtype=torch.int64)
    pos=torch.remainder(idx,int(block));bid=torch.div(idx,int(block),rounding_mode='floor')
    perm=torch.remainder(pos*61+283+bid*101,int(block))
    k_ep=int(round(endpoint_fraction*int(block)));k_fm=int(round(fm_fraction*int(block)))
    endpoint=perm<k_ep;fm=(perm>=k_ep)&(perm<k_ep+k_fm)
    return fm,endpoint

def _bce_dice(prob,target,eps=1e-6):
    prob=prob.clamp(eps,1-eps);target=target.float();bce=F.binary_cross_entropy(prob,target);dims=tuple(range(1,prob.ndim));inter=(prob*target).sum(dims);den=prob.sum(dims)+target.sum(dims);dice=(1-(2*inter+eps)/(den+eps)).mean();return bce+dice

def _balanced_bce_dice(prob,target,eps=1e-6,max_pos_weight=50.0):
    prob=prob.clamp(eps,1-eps);target=target.float();dims=tuple(range(1,prob.ndim));pos=target.sum(dims);neg=(1-target).sum(dims);posw=(neg/(pos+eps)).clamp(1.0,float(max_pos_weight));shape=(prob.shape[0],)+((1,)*(prob.ndim-1));w=1+target*(posw.reshape(shape)-1);bce=(-(w*(target*torch.log(prob)+(1-target)*torch.log1p(-prob))).flatten(1).sum(1)/w.flatten(1).sum(1).clamp_min(eps)).mean();inter=(prob*target).sum(dims);den=prob.sum(dims)+target.sum(dims);dice=(1-(2*inter+eps)/(den+eps)).mean();return bce+dice

class ImprovedMeanFlowStateLoss(nn.Module):
    def __init__(self,data_proportion=.5,time_mu=-.4,time_sigma=1.,norm_p=1.,norm_eps=.01,clean_weight=.25,fm_sampling='stratified',endpoint_probability=0.0,endpoint_sampling='stratified'):
        super().__init__();self.data_proportion=float(data_proportion);self.time_mu=float(time_mu);self.time_sigma=float(time_sigma);self.norm_p=float(norm_p);self.norm_eps=float(norm_eps);self.clean_weight=float(clean_weight);self.fm_sampling=str(fm_sampling);self.endpoint_probability=float(endpoint_probability);self.endpoint_sampling=str(endpoint_sampling)
    def _sample(self,b,device,sample_offset=0):
        n=torch.randn(2,b,device=device)*self.time_sigma+self.time_mu;a,bb=torch.sigmoid(n[0]),torch.sigmoid(n[1]);t=torch.maximum(a,bb);r=torch.minimum(a,bb)
        if self.endpoint_probability>0 and self.endpoint_sampling=='stratified_disjoint':
            if self.fm_sampling!='stratified':raise ValueError('stratified_disjoint endpoint schedule requires fm_sampling=stratified')
            fm,endpoint=_disjoint_stratified_schedule(b,self.data_proportion,self.endpoint_probability,sample_offset,device)
        else:
            fm=_stratified_mask(b,self.data_proportion,sample_offset,device) if self.fm_sampling=='stratified' else (torch.rand(b,device=device)<self.data_proportion);endpoint=torch.zeros(b,dtype=torch.bool,device=device)
        r=torch.where(fm,t,r)
        if endpoint.any():t=torch.where(endpoint,torch.ones_like(t),t);r=torch.where(endpoint,torch.zeros_like(r),r)
        return t,r,fm,endpoint
    def _adp(self,l):return (l/(l.detach()+self.norm_eps).pow(self.norm_p)).mean()
    def forward(self,model,x0,image,sample_offset=0):
        b=x0.shape[0];t,r,fm,endpoint=self._sample(b,x0.device,sample_offset);e=torch.randn_like(x0);z=(1-_b(t))*x0+_b(t)*e;vt=e-x0;out=model.flow_outputs(z,t,r,image);u,v,clean=out['u'],out['v'],out['clean_u']
        with torch.no_grad():
            vc=model.flow_outputs(z,t,t,image)['v']
            def fn(z_,t_,r_):return model.flow_outputs(z_,t_,r_,image)['u']
            _,du=jvp(fn,(z,t,r),(vc,torch.ones_like(t),torch.zeros_like(r)))
        V=u+_b(t-r)*du.detach();lu=self._adp((V-vt.detach()).pow(2).flatten(1).sum(1));lv=self._adp((v-vt.detach()).pow(2).flatten(1).sum(1));clean_prob=((clean+1)*.5).clamp(0,1);target_prob=((x0+1)*.5).clamp(0,1);clean_loss=_bce_dice(clean_prob,target_prob);total=lu+lv+self.clean_weight*clean_loss
        return total,{'total_loss':float(total.detach()),'imf_u_loss':float(lu.detach()),'imf_v_loss':float(lv.detach()),'clean_loss':float(clean_loss.detach()),'fm_count':int(fm.sum().item()),'gic_active_samples':0,'gic_active_batches':0,'near_deployment_count':int(((t>=.95)&(r<=.05)).sum().item()),'exact_deployment_count':int(endpoint.sum().item())}

class ImprovedMeanFlowGeometryLoss(nn.Module):
    def __init__(self,data_proportion=.5,time_mu=-.4,time_sigma=1.,norm_p=1.,norm_eps=.01,geometry_weight=.5,mask_weight=.5,radius_weight=1.,gic_weight=.1,gic_probability=.25,endpoint_probability=0.0,max_radius=16.,rasterizer=None,fm_sampling='stratified',gic_sampling='stratified',endpoint_sampling='stratified'):
        super().__init__();self.data_proportion=float(data_proportion);self.time_mu=float(time_mu);self.time_sigma=float(time_sigma);self.norm_p=float(norm_p);self.norm_eps=float(norm_eps);self.geometry_weight=float(geometry_weight);self.mask_weight=float(mask_weight);self.radius_weight=float(radius_weight);self.gic_weight=float(gic_weight);self.gic_probability=float(gic_probability);self.max_radius=float(max_radius);self.rasterizer=rasterizer;self.fm_sampling=str(fm_sampling);self.gic_sampling=str(gic_sampling);self.endpoint_probability=float(endpoint_probability);self.endpoint_sampling=str(endpoint_sampling)
    def sample_tr(self,b,device,sample_offset=0):
        n=torch.randn(2,b,device=device)*self.time_sigma+self.time_mu;a,bb=torch.sigmoid(n[0]),torch.sigmoid(n[1]);t=torch.maximum(a,bb);r=torch.minimum(a,bb)
        if self.endpoint_probability>0 and self.endpoint_sampling=='stratified_disjoint':
            if self.fm_sampling!='stratified': raise ValueError('stratified_disjoint endpoint schedule requires fm_sampling=stratified')
            fm,endpoint=_disjoint_stratified_schedule(b,self.data_proportion,self.endpoint_probability,sample_offset,device)
        else:
            fm=_stratified_mask(b,self.data_proportion,sample_offset,device) if self.fm_sampling=='stratified' else (torch.rand(b,device=device)<self.data_proportion)
            endpoint=(_independent_stratified_mask(b,self.endpoint_probability,sample_offset,device,'endpoint') if self.endpoint_sampling=='stratified_independent' else _stratified_mask(b,self.endpoint_probability,sample_offset,device)) if self.endpoint_sampling in {'stratified','stratified_independent'} else (torch.rand(b,device=device)<self.endpoint_probability)
            fm=fm & (~endpoint)
        r=torch.where(fm,t,r)
        if endpoint.any(): t=torch.where(endpoint,torch.ones_like(t),t);r=torch.where(endpoint,torch.zeros_like(r),r)
        return t,r,fm,endpoint
    def _adaptive(self,per_sample):return (per_sample/(per_sample.detach()+self.norm_eps).pow(self.norm_p)).mean()
    def _gic_mask(self,b,device,sample_offset):
        return (_independent_stratified_mask(b,self.gic_probability,sample_offset,device,'gic') if self.gic_sampling=='stratified_independent' else _stratified_mask(b,self.gic_probability,sample_offset,device)) if self.gic_sampling in {'stratified','stratified_independent'} else (torch.rand(b,device=device)<self.gic_probability)
    def forward(self,model,g0,image,radius_valid=None,mask_gt=None,sample_offset=0):
        b,device=g0.shape[0],g0.device;t,r,fm,endpoint=self.sample_tr(b,device,sample_offset);eps=torch.randn_like(g0);z=(1-_b(t))*g0+_b(t)*eps;v_t=eps-g0;main=model.flow_outputs(z,t,r,image);u,v,clean=main['u'],main['v'],main['clean_u']
        with torch.no_grad():
            v_c=model.flow_outputs(z,t,t,image)['v']
            def fn(z_,t_,r_):return model.flow_outputs(z_,t_,r_,image)['u']
            _,du_dt=jvp(fn,(z,t,r),(v_c,torch.ones_like(t),torch.zeros_like(r)))
        V=u+_b(t-r)*du_dt.detach();target=v_t.detach();loss_u=self._adaptive((V-target).pow(2).flatten(1).sum(1));loss_v=self._adaptive((v-target).pow(2).flatten(1).sum(1));total=loss_u+loss_v
        c_pred=((clean[:,0:1]+1)*.5).clamp(0,1);c_gt=((g0[:,0:1]+1)*.5).clamp(0,1);c_loss=_balanced_bce_dice(c_pred,c_gt)
        if self.rasterizer is not None and getattr(self.rasterizer,'representation','centerline_radius')=='centerline_edt' and mask_gt is not None:field_valid=(mask_gt>0).float()
        else:field_valid=radius_valid if radius_valid is not None else (((g0[:,0:1]+1)*.5)>0.5).float()
        denom=field_valid.sum().clamp_min(1.);r_loss=((clean[:,1:2]-g0[:,1:2]).abs()*field_valid).sum()/denom;geom=c_loss+self.radius_weight*r_loss;total=total+self.geometry_weight*geom
        mask_loss=g0.new_tensor(0.)
        if self.mask_weight>0 and self.rasterizer is not None:
            pred_mask=self.rasterizer(clean);gt_mask=((mask_gt+1)*.5).clamp(0,1) if mask_gt is not None else self.rasterizer(g0).detach();mask_loss=_bce_dice(pred_mask,gt_mask);total=total+self.mask_weight*mask_loss
        gic=g0.new_tensor(0.);parts={};gic_mask=self._gic_mask(b,device,sample_offset) if self.gic_weight>0 and self.rasterizer is not None else torch.zeros(b,dtype=torch.bool,device=device);gic_count=int(gic_mask.sum().item())
        if gic_count:
            gic,parts=geometry_interval_consistency(model,z[gic_mask],r[gic_mask],t[gic_mask],image[gic_mask],self.rasterizer);gic_scale=float(gic_count)/float(b);total=total+self.gic_weight*gic*gic_scale;parts['gic_active_fraction']=g0.new_tensor(gic_scale)
        if not torch.isfinite(total):raise RuntimeError('non-finite GeoCrack-iMF loss')
        near_deploy=((t>=.95)&(r<=.05)).sum(); exact_deploy=endpoint.sum()
        logs={'total_loss':total,'imf_u_loss':loss_u,'imf_v_loss':loss_v,'geometry_loss':geom,'centerline_loss':c_loss,'field_loss':r_loss,'radius_loss':r_loss,'mask_loss':mask_loss,'gic_loss':gic,'fm_count':fm.sum(),'gic_active_samples':g0.new_tensor(gic_count),'gic_active_batches':g0.new_tensor(1 if gic_count else 0),'batch_size':g0.new_tensor(b),'t_mean':t.mean(),'r_mean':r.mean(),'gap_mean':(t-r).mean(),'near_deployment_count':near_deploy,'exact_deployment_count':exact_deploy};logs.update(parts);return total,{k:float(vv.detach()) for k,vv in logs.items()}
