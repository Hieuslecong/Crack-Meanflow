from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
try:
    from torch.func import jvp
except ImportError:
    from functorch import jvp

def _b(v):return v.reshape(-1,1,1,1)
class ConferenceMeanFlowLoss(nn.Module):
    def __init__(self,ratio_r_not_equal_t=.35,boundary_prob=.25,time_mu=-.4,time_sigma=1.,adaptive_p=.75,adaptive_eps=1e-3,endpoint_loss_weight=.5,thin_loss_weight=0.,seg_loss_weight=0.,cfg_drop_prob=0.,time_curriculum=None,boundary_sampling='bernoulli'):
        super().__init__();self.base_interval=float(ratio_r_not_equal_t);self.boundary_prob=float(boundary_prob);self.time_mu=float(time_mu);self.time_sigma=float(time_sigma);self.adaptive_p=float(adaptive_p);self.adaptive_eps=float(adaptive_eps);self.endpoint_loss_weight=float(endpoint_loss_weight);self.thin_loss_weight=float(thin_loss_weight);self.seg_loss_weight=float(seg_loss_weight);self.cfg_drop_prob=float(cfg_drop_prob);self.time_curriculum=time_curriculum or {'enabled':False};self.boundary_sampling=str(boundary_sampling);self.epoch=0
        if self.boundary_sampling not in {'bernoulli','stratified'}:raise ValueError(f'unknown boundary_sampling={self.boundary_sampling!r}')
    def set_epoch(self,epoch):self.epoch=int(epoch)
    def _stage(self):
        interval_prob=self.base_interval;max_gap=1.;boundary_prob=self.boundary_prob;tc=self.time_curriculum or {}
        if tc.get('enabled'):
            for st in tc.get('stages',[]):
                lo,hi=st.get('epochs',[0,10**9])
                if lo<=self.epoch<hi:
                    if 'fm_ratio' in st:interval_prob=1.-float(st['fm_ratio'])
                    if 'interval_ratio' in st:interval_prob=float(st['interval_ratio'])
                    max_gap=float(st.get('max_gap',max_gap));boundary_prob=float(st.get('boundary_prob',boundary_prob));break
        if max_gap<1.:boundary_prob=0.
        return min(max(interval_prob,0.),1.),min(max(max_gap,0.),1.),min(max(boundary_prob,0.),1.)
    def _sample_r_t(self,b,device,return_stage=False,sample_offset=None):
        interval_prob,max_gap,boundary_prob=self._stage();n=torch.randn(2,b,device=device)*self.time_sigma+self.time_mu;a,bb=torch.sigmoid(n[0]),torch.sigmoid(n[1]);t=torch.maximum(a,bb);r=torch.minimum(a,bb)
        if max_gap<1.:r=torch.maximum(r,t-max_gap)
        collapse=torch.rand(b,device=device)>=interval_prob;r=torch.where(collapse,t,r)
        if boundary_prob<=0:boundary=torch.zeros(b,device=device,dtype=torch.bool)
        elif self.boundary_sampling=='stratified' and sample_offset is not None:
            idx=torch.arange(int(sample_offset),int(sample_offset)+b,device=device,dtype=torch.float64);p=float(boundary_prob);boundary=torch.floor((idx+1)*p)>torch.floor(idx*p)
        else:boundary=torch.rand(b,device=device)<boundary_prob
        r=torch.where(boundary,torch.zeros_like(r),r);t=torch.where(boundary,torch.ones_like(t),t);stage={'interval_prob':interval_prob,'max_gap':max_gap,'boundary_prob':boundary_prob,'boundary_sampling':self.boundary_sampling,'boundary_count':int(boundary.sum().item()),'fm_count':int((r==t).sum().item()),'realized_fm_fraction':float((r==t).float().mean().item())};return (r,t,stage) if return_stage else (r,t)
    @staticmethod
    def _dice(pred,target,eps=1e-6):
        p=((pred+1)*.5).clamp(0,1);g=(target+1)*.5;dims=tuple(range(1,p.ndim));inter=(p*g).sum(dims);den=p.sum(dims)+g.sum(dims);return (1-(2*inter+eps)/(den+eps)).mean()
    @staticmethod
    def _thin(pred,target):
        pos=(target>0).float();near=(F.conv2d(pos,torch.ones(1,1,3,3,device=pos.device),padding=1)>0).float();return ((pred-target).abs()*(1+4*near)).mean()
    def forward(self,model,x0,model_kwargs=None):
        kw=dict(model_kwargs or {});y=kw.get('y')
        if y is None:raise ValueError('ConferenceMeanFlowLoss requires conditioning image y')
        x0=x0.float();y=y.float();b=x0.shape[0];device=x0.device
        if self.cfg_drop_prob>0:
            drop=(torch.rand(b,device=device)<self.cfg_drop_prob).reshape(-1,1,1,1);y=torch.where(drop,torch.zeros_like(y),y)
        sample_offset=kw.get('sample_offset',None);r,t,stage=self._sample_r_t(b,device,return_stage=True,sample_offset=sample_offset);eps=torch.randn_like(x0);z=(1-_b(t))*x0+_b(t)*eps;v=eps-x0;u=model(z,r,t,y=y);base=model.module if hasattr(model,'module') else model;seg_logits=base.get_seg_logits()
        def fn(z_,r_,t_):return model(z_,r_,t_,y=y)
        with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):_,dudt=jvp(fn,(z,r,t),(v,torch.zeros_like(r),torch.ones_like(t)))
        target=(v-_b(t-r)*dudt).detach();sq=(u-target).pow(2).flatten(1).mean(1);weight=(sq.detach()+self.adaptive_eps).pow(self.adaptive_p);mf=(sq/weight).mean();total=mf.clone();endpoint=x0.new_tensor(0.);thin=x0.new_tensor(0.);seg=x0.new_tensor(0.);at_zero=(r==0)
        if at_zero.any() and (self.endpoint_loss_weight>0 or self.thin_loss_weight>0):
            idx=at_zero.nonzero(as_tuple=True)[0];xhat=z[idx]-_b(t[idx])*u[idx]
            if self.endpoint_loss_weight>0:endpoint=F.l1_loss(xhat,x0[idx])+self._dice(xhat,x0[idx]);total+=self.endpoint_loss_weight*endpoint
            if self.thin_loss_weight>0:thin=self._thin(xhat,x0[idx]);total+=self.thin_loss_weight*thin
        if self.seg_loss_weight>0 and seg_logits is not None:
            seg=F.binary_cross_entropy_with_logits(seg_logits.float(),(x0+1)*.5);total+=self.seg_loss_weight*seg
        if not torch.isfinite(total):raise RuntimeError('non-finite Conference MeanFlow loss')
        logs={k:float(vv.detach()) for k,vv in {'total_loss':total,'mf_loss':mf,'endpoint_loss':endpoint,'thin_loss':thin,'seg_loss':seg}.items()};logs.update(stage);return total,logs
