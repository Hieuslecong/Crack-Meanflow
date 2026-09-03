from __future__ import annotations
import argparse,json,os,statistics,sys,torch,yaml
from torch.utils.data import DataLoader
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crackmeanflow.common import PairedCrackDataset,discover_required_splits,evaluate_test_with_frozen_threshold,verify_checkpoint_split_provenance,config_hash
from crackmeanflow.sampler import crack_meanflow_sampler
from crackmeanflow.sit import build_sit
from crackmeanflow.adapter import CrackMeanFlowModel
from crackmeanflow.journal.models.sit_mask_baseline import MaskIMFSiTModel,HybridMaskIMFModel
from crackmeanflow.journal import build_geocrack_imf,GeometryRasterizer,evaluate_geometry_with_frozen_threshold

def _ema(model,ck): model.load_state_dict({**model.state_dict(),**ck['ema']},strict=False)
def _summary(rows,keys): return {k:{'mean':statistics.mean([r[k] for r in rows]),'std':statistics.pstdev([r[k] for r in rows])} for k in keys}
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--ckpt',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',default=None);ap.add_argument('--allow-split-mismatch',action='store_true');ap.add_argument('--allow-config-mismatch',action='store_true');a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config));device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ck=torch.load(a.ckpt,map_location='cpu',weights_only=False)
    if not a.allow_config_mismatch and ck.get('config_hash') and ck['config_hash']!=config_hash(cfg): raise RuntimeError('evaluation config does not match checkpoint config')
    sp=discover_required_splits(a.data)
    if not a.allow_split_mismatch: verify_checkpoint_split_provenance(ck,sp,keys=('val','test'))
    ds=PairedCrackDataset(sp['test'],cfg['model']['img_size'],False,False,cfg['train'].get('mask_resize_mode','nearest'));ld=DataLoader(ds,batch_size=1,shuffle=False); seeds=cfg['eval'].get('eval_seeds',[0]); th=float(ck['best_val_threshold'])
    if cfg['backbone']=='sit_mf':
        sit=build_sit(cfg['model']['img_size'],cfg['model']['patch'],cfg['model']['size'],cfg['model'].get('in_ch',1),cfg['model'].get('cond_ch',3));model=CrackMeanFlowModel(sit,T=500).to(device);_ema(model,ck);rows=[evaluate_test_with_frozen_threshold(model,ld,device,crack_meanflow_sampler,th,1,s,cfg['eval'].get('cfg_scale',1.0)) for s in seeds]; out=_summary(rows,['f1','iou','precision','recall','cldice','boundary_f1'])
    elif cfg['backbone']=='sit_imf_mask':
        model=MaskIMFSiTModel(cfg['model']['img_size'],cfg['model']['patch'],cfg['model']['size'],cfg['model'].get('background_init',-0.95)).to(device);_ema(model,ck);rows=[evaluate_test_with_frozen_threshold(model,ld,device,crack_meanflow_sampler,th,1,s,1.0) for s in seeds];out=_summary(rows,['f1','iou','precision','recall','cldice','boundary_f1'])
    elif cfg['backbone']=='hybrid_imf_mask':
        model=HybridMaskIMFModel(cfg['model']['img_size'],cfg['model']['patch'],cfg['model'].get('size','S'),cfg['model'].get('background_init',-0.95)).to(device);_ema(model,ck);rows=[evaluate_test_with_frozen_threshold(model,ld,device,crack_meanflow_sampler,th,1,s,1.0) for s in seeds];out=_summary(rows,['f1','iou','precision','recall','cldice','boundary_f1'])
    else:
        model=build_geocrack_imf(cfg['model']).to(device);_ema(model,ck);rast=GeometryRasterizer(cfg['model'].get('max_radius',16),cfg['model'].get('radius_bins',8),representation=cfg['model'].get('representation','centerline_radius'),distance_encoding=cfg['model'].get('distance_encoding','linear')).to(device); rows=[evaluate_geometry_with_frozen_threshold(model,ld,device,rast,th,s,cfg['model'].get('max_radius',16)) for s in seeds]; keys=['f1','iou','precision','recall','cldice','boundary_f1','centerline_assd_px','edt_radius_mae_px','skeleton_length_rel_error'];out=_summary(rows,keys)
    out.update({'nfe':1,'threshold':th,'threshold_source':'validation','seeds':seeds}); path=a.out or os.path.join(os.path.dirname(a.ckpt),'JOURNAL_TEST_REPORT.json');json.dump(out,open(path,'w'),indent=2);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
