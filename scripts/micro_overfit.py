from __future__ import annotations
import argparse,copy,itertools,json,os,random,sys
from pathlib import Path
import numpy as np,torch,yaml
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crackmeanflow.common import PairedCrackDataset,discover_required_splits,calibrate_threshold_on_validation,resolve_thresholds
from crackmeanflow.journal.engine.dataset import GeometryDataset
from crackmeanflow.journal import calibrate_geometry_threshold_on_validation
from crackmeanflow.sampler import crack_meanflow_sampler
from scripts.train_journal import build_track

def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def evaluate(model,rast,cfg,loader,device,seed):
    model.eval()
    if cfg['backbone']=='geocrack_imf':
        thresholds=resolve_thresholds(cfg['eval'],final=True);sweep,th=calibrate_geometry_threshold_on_validation(model,loader,device,rast,thresholds,seed,cfg['model'].get('max_radius',16));return sweep[th],float(th)
    thresholds=resolve_thresholds(cfg['eval'],final=True);sweep,th=calibrate_threshold_on_validation(model,loader,device,crack_meanflow_sampler,thresholds,1,seed,cfg['eval'].get('cfg_scale',1.0));return sweep[th],float(th)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--data',required=True);ap.add_argument('--subset',type=int,default=8);ap.add_argument('--steps',type=int,default=200);ap.add_argument('--image-size',type=int,default=128);ap.add_argument('--batch-size',type=int,default=1);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--lr',type=float,default=None);ap.add_argument('--out',default=None);ap.add_argument('--max-loss-ratio',type=float,default=.70);ap.add_argument('--min-f1-gain',type=float,default=.15);ap.add_argument('--min-final-f1',type=float,default=None);a=ap.parse_args();seed_all(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');cfg=copy.deepcopy(yaml.safe_load(open(a.config)));cfg['model']['img_size']=a.image_size;cfg['train']['batch_size']=a.batch_size;cfg['train']['drop_last']=False;cfg['train']['seed']=a.seed
    splits=discover_required_splits(a.data);pairs=sorted(splits['train'])[:max(1,a.subset)];base=PairedCrackDataset(pairs,a.image_size,augment=False,photometric=False,mask_resize_mode=cfg['train'].get('mask_resize_mode','nearest'));ds=GeometryDataset(base,cfg['model'].get('max_radius',16),cfg['model'].get('representation','centerline_radius'),cfg['model'].get('distance_encoding','linear')) if cfg['backbone']=='geocrack_imf' else base;loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,drop_last=False);eval_loader=DataLoader(ds,batch_size=1,shuffle=False);model,rast,lossfn=build_track(cfg,device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr or cfg['train']['lr'],weight_decay=cfg['train']['weight_decay']);initial,initial_th=evaluate(model,rast,cfg,eval_loader,device,a.seed);losses=[];offset=0;it=itertools.cycle(loader);model.train()
    for step in range(a.steps):
        b=next(it);img=b['crack'].to(device);bs=img.shape[0];opt.zero_grad(set_to_none=True)
        if cfg['backbone']=='geocrack_imf':loss,logs=lossfn(model,b['geometry'].to(device),img,b['radius_valid'].to(device),mask_gt=b['mask'].to(device),sample_offset=offset)
        elif cfg['backbone'] in {'sit_imf_mask','hybrid_imf_mask'}:loss,logs=lossfn(model,b['mask'].to(device),img,sample_offset=offset)
        else:loss,logs=lossfn(model,b['mask'].to(device),{'y':img,'sample_offset':offset})
        offset+=bs;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train'].get('max_grad_norm',1.0));opt.step();losses.append(float(loss.detach()))
    final,final_th=evaluate(model,rast,cfg,eval_loader,device,a.seed);first=float(np.mean(losses[:max(1,min(10,len(losses)))]));last=float(np.mean(losses[-max(1,min(10,len(losses))):]));loss_ratio=last/max(first,1e-12);f1_gain=float(final['f1']-initial['f1']);finite=bool(torch.isfinite(torch.tensor(losses)).all());checks={'finite':finite,'loss_ratio_pass':loss_ratio<=a.max_loss_ratio,'f1_gain_pass':f1_gain>=a.min_f1_gain,'optional_final_f1_pass':True if a.min_final_f1 is None else final['f1']>=a.min_final_f1};passed=all(checks.values())
    out={'diagnostic':'MICRO_OVERFIT_ONLY','config':a.config,'backbone':cfg['backbone'],'subset':len(pairs),'image_size':a.image_size,'steps':a.steps,'seed':a.seed,'initial':initial,'initial_threshold':initial_th,'final':final,'final_threshold':final_th,'loss_first_window':first,'loss_last_window':last,'loss_ratio':loss_ratio,'f1_gain':f1_gain,'pass':passed,'pass_checks':checks,'pass_rule':f'finite AND loss_ratio<={a.max_loss_ratio} AND f1_gain>={a.min_f1_gain}' + ('' if a.min_final_f1 is None else f' AND final_f1>={a.min_final_f1}') + '; diagnostic gate only; no absolute final-F1 threshold by default'};path=Path(a.out) if a.out else Path(__file__).resolve().parents[1]/'reports'/f'MICRO_OVERFIT_{cfg["experiment"]}.json';Path(path).parent.mkdir(parents=True,exist_ok=True);json.dump(out,open(path,'w'),indent=2);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
