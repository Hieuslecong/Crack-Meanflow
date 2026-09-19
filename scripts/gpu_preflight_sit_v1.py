from __future__ import annotations
import argparse,json,os,sys,torch,yaml
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crackmeanflow.factory import build_training_components
from crackmeanflow.common.ema import EMA
from crackmeanflow.sampler import crack_meanflow_sampler


class ForwardCounter(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model=model; self.forward_calls=0
    def forward(self,*args,**kwargs):
        self.forward_calls+=1
        return self.model(*args,**kwargs)
    def get_seg_logits(self):
        return self.model.get_seg_logits()
    def __getattr__(self,name):
        if name in {'model','forward_calls'}:
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model,name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    a=ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for SiT-V1 GPU preflight')

    cfg=yaml.safe_load(open(a.config))
    device=torch.device('cuda')
    torch.cuda.reset_peak_memory_stats(device)
    model,_,lossfn=build_training_components(cfg,device)
    model.train()

    tr=cfg['train']
    opt=torch.optim.AdamW(
        model.parameters(),
        lr=float(tr['lr']),
        weight_decay=float(tr['weight_decay']),
        betas=(float(tr.get('adam_beta1',0.9)),float(tr.get('adam_beta2',0.999))),
    )
    ema=EMA(model,float(tr['ema_decay']))

    b=int(tr['batch_size']); s=int(cfg['model']['img_size'])
    x0=torch.where(torch.rand(b,1,s,s,device=device)>.95,torch.ones(1,device=device),-torch.ones(1,device=device))
    image=torch.randn(b,3,s,s,device=device)

    loss,logs=lossfn(model,x0,image,sample_offset=0)
    loss.backward()
    grad_norm=torch.nn.utils.clip_grad_norm_(
        model.parameters(),float(tr['max_grad_norm']),error_if_nonfinite=True
    )
    opt.step(); ema.update(model); opt.zero_grad(set_to_none=True)

    model.eval()
    counted=ForwardCounter(model)
    z=torch.randn_like(x0)
    pred,_=crack_meanflow_sampler(counted,z,image,num_steps=1,cfg_scale=1.0,clamp=False)
    if counted.forward_calls!=1:
        raise RuntimeError(f'NFE=1 contract violated: forward_calls={counted.forward_calls}')
    if not torch.isfinite(pred).all():
        raise RuntimeError('non-finite NFE=1 inference output')

    out={
        'status':'PASS',
        'variant':cfg['sit_variant'],
        'objective':cfg['loss']['mode'],
        'loss':float(loss.detach()),
        'grad_norm':float(grad_norm.detach()),
        'params':sum(p.numel() for p in model.parameters()),
        'optimizer_betas':[float(tr.get('adam_beta1',0.9)),float(tr.get('adam_beta2',0.999))],
        'ema_decay':float(tr['ema_decay']),
        'nfe_contract':1,
        'measured_forward_calls':int(counted.forward_calls),
        'inference_finite':True,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
        'logs':logs,
    }
    print(json.dumps(out,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
