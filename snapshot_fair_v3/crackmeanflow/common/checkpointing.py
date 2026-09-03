from __future__ import annotations
import os, platform, subprocess, sys, random, json, hashlib
import numpy as np
import torch

def _git_commit():
    try: return subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception: return 'NO_GIT_METADATA'

def environment_info():
    return {'python':sys.version,'platform':platform.platform(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}

def config_hash(cfg) -> str:
    payload=json.dumps(cfg,sort_keys=True,separators=(',',':'),default=str).encode('utf-8'); return hashlib.sha256(payload).hexdigest()

def capture_rng_state():
    state={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
    if torch.cuda.is_available(): state['cuda']=torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state):
    if not state: return
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state: torch.cuda.set_rng_state_all(state['cuda'])

def save_checkpoint_atomic(path, *, model, ema, optimizer, scheduler, epoch, global_optimizer_step, cfg,best_val_metric, best_val_threshold, split_manifest_hashes, seed, threshold_metric='f1',extra_state=None):
    payload={'model':model.state_dict(),'ema':getattr(ema,'shadow',ema),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':int(epoch),'global_optimizer_step':int(global_optimizer_step),'cfg':cfg,'config_hash':config_hash(cfg),'git_commit':_git_commit(),'seed':int(seed),'rng_state':capture_rng_state(),'best_val_metric':float(best_val_metric),'best_val_threshold':float(best_val_threshold),'threshold_metric':threshold_metric,'split_manifest_hashes':dict(split_manifest_hashes),'environment':environment_info(),'extra_state':dict(extra_state or {})}
    os.makedirs(os.path.dirname(path) or '.',exist_ok=True); tmp=path+'.tmp'; torch.save(payload,tmp); os.replace(tmp,path)

def _optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k,v in list(state.items()):
            if torch.is_tensor(v): state[k]=v.to(device)

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location='cpu'):
    ck=torch.load(path,map_location=map_location,weights_only=False); model.load_state_dict(ck['model'])
    if optimizer is not None and 'optimizer' in ck:
        optimizer.load_state_dict(ck['optimizer']); device=next(model.parameters()).device; _optimizer_to(optimizer,device)
    if scheduler is not None and 'scheduler' in ck: scheduler.load_state_dict(ck['scheduler'])
    return ck
