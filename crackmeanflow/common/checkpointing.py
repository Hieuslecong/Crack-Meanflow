from __future__ import annotations
import os,platform,subprocess,sys,random,json,hashlib,importlib.metadata
from pathlib import Path
import numpy as np
import torch

def _git_commit():
    try: return subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception: return 'NO_GIT_METADATA'


def source_tree_manifest(root=None):
    """Return a stable manifest of execution-relevant source files.

    Configs are intentionally excluded: the exact EFFECTIVE_CONFIG has its own
    independent config hash. Audit/report scripts are also excluded so adding a
    diagnostic tool cannot invalidate an otherwise identical trained checkpoint.
    """
    root=Path(root) if root is not None else Path(__file__).resolve().parents[2]
    files=[]
    base=root/'crackmeanflow'
    if base.exists(): files.extend(p for p in base.rglob('*.py') if '__pycache__' not in p.parts)
    for name in ('scripts/train_journal.py','scripts/evaluate_journal.py','scripts/freeze_source_threshold.py'):
        q=root/name
        if q.exists(): files.append(q)
    for name in ('requirements.txt','pytest.ini'):
        q=root/name
        if q.exists(): files.append(q)
    rows=[]
    for q in sorted(set(files),key=lambda x:str(x.relative_to(root))):
        rel=str(q.relative_to(root)).replace(os.sep,'/')
        rows.append({'path':rel,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
    return rows

def source_tree_hash(root=None):
    rows=source_tree_manifest(root)
    h=hashlib.sha256()
    for row in rows:
        h.update(row['path'].encode());h.update(b'\0');h.update(row['sha256'].encode());h.update(b'\n')
    return h.hexdigest()


def protocol_bundle_manifest(root=None):
    """Stable manifest of preregistered paper/fairness protocol files.

    Execution code/config provenance is tracked separately. This bundle binds a
    scientific run to the exact paper protocol and fairness matrices in force
    when the checkpoint was created.
    """
    root=Path(root) if root is not None else Path(__file__).resolve().parents[2]
    files=[]
    for rel in ('configs/protocol','configs/fairness'):
        base=root/rel
        if base.exists(): files.extend(p for p in base.rglob('*.yaml') if p.is_file())
    rows=[]
    for q in sorted(set(files),key=lambda x:str(x.relative_to(root))):
        rel=str(q.relative_to(root)).replace(os.sep,'/')
        rows.append({'path':rel,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
    if not rows: raise RuntimeError('protocol bundle is empty; configs/protocol and configs/fairness are required')
    return rows

def protocol_bundle_hash(root=None):
    h=hashlib.sha256()
    for row in protocol_bundle_manifest(root):
        h.update(row['path'].encode());h.update(b'\0');h.update(row['sha256'].encode());h.update(b'\n')
    return h.hexdigest()

def file_sha256(path,chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()

def environment_info():
    cuda_available=torch.cuda.is_available()
    package_versions={}
    for dist in ('numpy','Pillow','PyYAML','scipy','scikit-image'):
        try: package_versions[dist]=importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError: package_versions[dist]=None
    info={'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),'torch':str(torch.__version__),'package_versions':package_versions,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,'cuda_available':cuda_available,'cuda_device_count':torch.cuda.device_count() if cuda_available else 0,'cuda_devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda_available else [],'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),'cudnn_benchmark':bool(torch.backends.cudnn.benchmark),'cudnn_deterministic':bool(torch.backends.cudnn.deterministic),'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'pythonhashseed':os.environ.get('PYTHONHASHSEED')}
    if cuda_available:
        info['cuda_device_capability']=[list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
        for key, getter in (
            ('cuda_matmul_allow_tf32', lambda: bool(torch.backends.cuda.matmul.allow_tf32)),
            ('cudnn_allow_tf32', lambda: bool(torch.backends.cudnn.allow_tf32)),
        ):
            try:
                info[key]=getter()
            except (AttributeError, RuntimeError) as exc:
                info[key]=None
                info[key+'_probe_error']=type(exc).__name__
    return info

def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()

def capture_rng_state():
    state={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
    if torch.cuda.is_available(): state['cuda']=torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state):
    if not state:return
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:torch.cuda.set_rng_state_all(state['cuda'])

def save_checkpoint_atomic(path,*,model,ema,optimizer,scheduler,epoch,global_optimizer_step,cfg,best_val_metric,best_val_threshold,split_manifest_hashes=None,split_manifest_content_hashes=None,source_provenance=None,seed,threshold_metric='f1',extra_state=None):
    payload={'model':model.state_dict(),'ema':getattr(ema,'shadow',ema),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':int(epoch),'global_optimizer_step':int(global_optimizer_step),'cfg':cfg,'config_hash':config_hash(cfg),'git_commit':_git_commit(),'source_tree_sha256':source_tree_hash(),'source_tree_manifest':source_tree_manifest(),'protocol_bundle_sha256':protocol_bundle_hash(),'protocol_bundle_manifest':protocol_bundle_manifest(),'seed':int(seed),'rng_state':capture_rng_state(),'best_val_metric':float(best_val_metric) if best_val_metric is not None else None,'best_val_threshold':float(best_val_threshold) if best_val_threshold is not None else None,'threshold_metric':threshold_metric,'environment':environment_info(),'extra_state':dict(extra_state or {})}
    if split_manifest_hashes is not None: payload['split_manifest_hashes']=dict(split_manifest_hashes)
    if split_manifest_content_hashes is not None: payload['split_manifest_content_hashes']=dict(split_manifest_content_hashes)
    if source_provenance is not None: payload['source_provenance']=source_provenance
    os.makedirs(os.path.dirname(path) or '.',exist_ok=True);tmp=path+'.tmp';torch.save(payload,tmp);os.replace(tmp,path)

def _optimizer_to(optimizer,device):
    for state in optimizer.state.values():
        for k,v in list(state.items()):
            if torch.is_tensor(v): state[k]=v.to(device)

def load_checkpoint(path,model,optimizer=None,scheduler=None,map_location='cpu'):
    ck=torch.load(path,map_location=map_location,weights_only=False);model.load_state_dict(ck['model'])
    if optimizer is not None and 'optimizer' in ck:
        optimizer.load_state_dict(ck['optimizer']);_optimizer_to(optimizer,next(model.parameters()).device)
    if scheduler is not None and 'scheduler' in ck:scheduler.load_state_dict(ck['scheduler'])
    return ck
