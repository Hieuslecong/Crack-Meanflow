from __future__ import annotations
import argparse,json,statistics,sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, yaml
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crackmeanflow.common import (
    PairedCrackDataset,discover_required_splits,discover_evaluation_pairs,verify_source_dataset_contract,
    load_and_verify_target_lock,load_and_verify_threshold_lock,source_splits_for_config,source_tree_hash,config_hash,load_binary_mask_native,
    compute_segmentation_metrics,cldice_score,boundary_f1_score,
)
from crackmeanflow.factory import build_model_and_rasterizer
from crackmeanflow.sampler import crack_meanflow_sampler
from crackmeanflow.journal.engine.sampler import sample_geometry_one_step

def _ema(model,ck):
    ema=ck.get('ema');cur=model.state_dict()
    if not isinstance(ema,dict):raise RuntimeError('checkpoint has no EMA state')
    exp={k for k,v in cur.items() if torch.is_tensor(v) and v.dtype.is_floating_point}
    if exp!=set(ema):raise RuntimeError('EMA state keys do not exactly match model floating-point state')
    merged=dict(cur);merged.update(ema);model.load_state_dict(merged,strict=True)

def _aggregate(rows):
    tp=sum(x['tp'] for x in rows);fp=sum(x['fp'] for x in rows);fn=sum(x['fn'] for x in rows)
    if tp+fp+fn==0:pr=re=f1=iou=1.
    else:
        pr=tp/max(tp+fp,1e-12);re=tp/max(tp+fn,1e-12);f1=2*pr*re/max(pr+re,1e-12);iou=tp/max(tp+fp+fn,1e-12)
    return {'f1':f1,'iou':iou,'precision':pr,'recall':re,'f1_macro_image':float(np.mean([x['f1'] for x in rows])),'iou_macro_image':float(np.mean([x['iou'] for x in rows])),'cldice':float(np.mean([x['cldice'] for x in rows])),'boundary_f1':float(np.mean([x['boundary_f1'] for x in rows]))}

def main():
    ap=argparse.ArgumentParser(description='Diagnostic: resize model outputs back to each target image native resolution before metrics.')
    ap.add_argument('--config',required=True);ap.add_argument('--ckpt',required=True);ap.add_argument('--source-data',required=True);ap.add_argument('--data',required=True);ap.add_argument('--dataset-name',required=True);ap.add_argument('--dataset-version',required=True);ap.add_argument('--target-lock',required=True);ap.add_argument('--threshold-lock',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config));ck=torch.load(a.ckpt,map_location='cpu',weights_only=False)
    if ck.get('config_hash')!=config_hash(cfg):raise RuntimeError('config/checkpoint mismatch')
    if ck.get('source_tree_sha256')!=source_tree_hash():raise RuntimeError('code/checkpoint source-tree mismatch')
    source=source_splits_for_config(a.source_data,cfg);sp=ck.get('source_provenance') or {};source_name=sp.get('dataset_name');source_version=sp.get('dataset_version');verify_source_dataset_contract(ck,source,source_name,source_version,keys=('train','val','test'))
    preview=json.loads(Path(a.target_lock).read_text());pairs=discover_evaluation_pairs(a.data,'test',bool(preview.get('include_normal_negatives',False)));load_and_verify_target_lock(a.target_lock,pairs,a.dataset_name,a.dataset_version);_,th=load_and_verify_threshold_lock(a.threshold_lock,a.ckpt,ck,source['val'],source_name,source_version)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model,rast=build_model_and_rasterizer(cfg,device);_ema(model,ck);model.eval();ds=PairedCrackDataset(pairs,cfg['model']['img_size'],False,False,cfg['train'].get('mask_resize_mode','nearest'),cfg['train'].get('mask_binarization','auto_binary_safe'))
    seed_results=[]
    for seed in [int(x) for x in cfg['eval'].get('eval_seeds',[0])]:
        gen=torch.Generator(device='cpu').manual_seed(seed);rows=[]
        for i,(name,ip,mp) in enumerate(pairs):
            item=ds[i];img=item['crack'][None].to(device);mask_model=item['mask'][None].to(device)
            with Image.open(ip) as im:w,h=im.size
            gt=load_binary_mask_native(mp,cfg['train'].get('mask_binarization','auto_binary_safe'),fallback_hw=(h,w))[None]
            with torch.no_grad():
                if cfg['backbone']=='geocrack_imf':
                    z=torch.randn((1,2,cfg['model']['img_size'],cfg['model']['img_size']),generator=gen).to(device);_,prob=sample_geometry_one_step(model,z,img,rast);continuous=prob.detach().cpu()
                else:
                    z=torch.randn(mask_model.shape,generator=gen).to(device);pred_pm1,_=crack_meanflow_sampler(model,z,img,num_steps=1,cfg_scale=cfg['eval'].get('cfg_scale',1.0) if cfg['backbone'] in {'unet','sit_mf'} else 1.0,clamp=False);continuous=pred_pm1.detach().cpu()
            up=F.interpolate(continuous,size=(h,w),mode='bilinear',align_corners=False);pred=(up>float(th)).float();m=compute_segmentation_metrics(pred,gt);m['cldice']=cldice_score(pred,gt);m['boundary_f1']=boundary_f1_score(pred,gt);m['name']=name;rows.append(m)
        seed_results.append({'seed':seed,**_aggregate(rows)})
    keys=('f1','iou','precision','recall','f1_macro_image','iou_macro_image','cldice','boundary_f1');summary={k:{'mean':statistics.mean([r[k] for r in seed_results]),'std_over_inference_noise':statistics.pstdev([r[k] for r in seed_results])} for k in keys}
    out={'diagnostic':'NATIVE_RESOLUTION_RESIZE_BACK','scientific_validity':'DIAGNOSTIC_ONLY_NOT_HEADLINE','dataset_name':a.dataset_name,'dataset_version':a.dataset_version,'model_input_resolution':cfg['model']['img_size'],'output_resize_mode':'bilinear continuous output then frozen threshold','boundary_tolerance_note':'boundary_f1 tolerance=2 native pixels is diagnostic and not resolution-normalized' ,'threshold':float(th),'inference_noise_rows':seed_results,'metrics':summary}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
