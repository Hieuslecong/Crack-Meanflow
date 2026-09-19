from __future__ import annotations
import argparse,json,torch,yaml
from crackmeanflow.factory import build_training_components

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for SiT-V1 GPU preflight')
    cfg=yaml.safe_load(open(a.config));device=torch.device('cuda');torch.cuda.reset_peak_memory_stats(device)
    model,_,lossfn=build_training_components(cfg,device);model.train();opt=torch.optim.AdamW(model.parameters(),lr=cfg['train']['lr'],weight_decay=cfg['train']['weight_decay'])
    b=int(cfg['train']['batch_size']);s=int(cfg['model']['img_size']);x0=torch.where(torch.rand(b,1,s,s,device=device)>.95,torch.ones(1,device=device),-torch.ones(1,device=device));image=torch.randn(b,3,s,s,device=device)
    loss,logs=lossfn(model,x0,image,sample_offset=0);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['max_grad_norm'],error_if_nonfinite=True);opt.step()
    out={'status':'PASS','variant':cfg['sit_variant'],'objective':cfg['loss']['mode'],'loss':float(loss.detach()),'params':sum(p.numel() for p in model.parameters()),'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),'nfe_contract':1,'logs':logs}
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
