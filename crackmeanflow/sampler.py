"""One-step (and few-step) MeanFlow sampling. t=1 noise, t=0 data."""
import torch

def _get(model,name): return getattr(model.module,name) if hasattr(model,'module') else getattr(model,name)
@torch.no_grad()
def crack_meanflow_sampler(model,z,crack_image,num_steps=1,cfg_scale=1.0,clamp=False):
    if num_steps<1: raise ValueError(f'num_steps must be >=1, got {num_steps}')
    b,device=z.shape[0],z.device; do_cfg=cfg_scale>1.; x=z
    def step(x_cur,r_val,t_val):
        r=torch.full((b,),float(r_val),device=device); t=torch.full((b,),float(t_val),device=device)
        if do_cfg:
            xx=torch.cat([x_cur,x_cur],0); rr=torch.cat([r,r],0); tt=torch.cat([t,t],0); yy=torch.cat([crack_image,torch.zeros_like(crack_image)],0); uc,uu=torch.chunk(model(xx,rr,tt,y=yy),2,dim=0); u=uu+cfg_scale*(uc-uu)
        else: u=model(x_cur,r,t,y=crack_image)
        return x_cur-(t_val-r_val)*u
    grid=torch.linspace(1.,0.,num_steps+1).tolist()
    for i in range(num_steps): x=step(x,grid[i+1],grid[i])
    if clamp: x=x.clamp(-1.,1.)
    base=model.module if hasattr(model,'module') else model; seg=base.get_seg_logits() if hasattr(base,'get_seg_logits') else None; return x,seg
