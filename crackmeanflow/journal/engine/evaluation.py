from __future__ import annotations
import numpy as np, torch
from crackmeanflow.common.metrics import compute_segmentation_metrics,cldice_score,boundary_f1_score,geometry_centerline_metrics
from crackmeanflow.common.evaluation import _micro_threshold_sweep_from_score_gt
from crackmeanflow.journal.geometry.targets import mask_to_geometry_state,geometry_state_to_fields
from .sampler import sample_geometry_one_step

@torch.no_grad()
def collect_geometry(model,loader,device,rasterizer,seed=0):
    model.eval(); gen=torch.Generator(device='cpu').manual_seed(int(seed)); rows=[]
    for b in loader:
        img=b['crack'].to(device); gt_pm1=b['mask'].to(device); z=torch.randn((img.shape[0],2,img.shape[-2],img.shape[-1]),generator=gen).to(device)
        geom,prob=sample_geometry_one_step(model,z,img,rasterizer); rows.append((geom.detach().cpu(),prob.detach().cpu(),gt_pm1.detach().cpu()))
    return rows

def aggregate_geometry(collected,threshold,max_radius,representation,distance_encoding='linear',include_geometry=True,include_structural=True):
    tp=fp=fn=tn=0.; cls=[]; bfs=[]; geoms=[]; per=[]; per_pos=[];gt_pos=pred_pos=total_px=0.;empty_gt=empty_gt_fp=0
    for geom,prob,gt_pm1 in collected:
        gt=((gt_pm1+1)*.5); pred=(prob>float(threshold)).float(); m=compute_segmentation_metrics(pred,gt); tp+=m['tp'];fp+=m['fp'];fn+=m['fn'];tn+=m['tn']
        gt_pos+=float(gt.sum().detach().cpu());pred_pos+=float(pred.sum().detach().cpu());total_px+=float(gt.numel())
        for bi in range(gt.shape[0]):
            mi=compute_segmentation_metrics(pred[bi:bi+1],gt[bi:bi+1]);per.append(mi);
            if include_structural:cls.append(cldice_score(pred[bi:bi+1],gt[bi:bi+1]));bfs.append(boundary_f1_score(pred[bi:bi+1],gt[bi:bi+1]))
            if float(gt[bi].sum().detach().cpu())>0:per_pos.append(mi)
            else:
                empty_gt+=1
                if float(pred[bi].sum().detach().cpu())>0:empty_gt_fp+=1
        if include_geometry:
            gg,_=mask_to_geometry_state(gt_pm1,max_radius,representation,distance_encoding); c_pred,r_pred=geometry_state_to_fields(geom,max_radius,representation,distance_encoding); c_gt,r_gt=geometry_state_to_fields(gg,max_radius,representation,distance_encoding)
            geoms.append(geometry_centerline_metrics(c_pred,c_gt,r_pred,r_gt,max_radius))
    if tp+fp+fn==0: pr=re=f1=iou=1.0
    else:
        pr=tp/max(tp+fp,1e-12); re=tp/max(tp+fn,1e-12); f1=2*pr*re/max(pr+re,1e-12); iou=tp/max(tp+fp+fn,1e-12)
    macro=lambda key:float(np.mean([x[key] for x in per])) if per else float('nan')
    macro_pos=lambda key:float(np.mean([x[key] for x in per_pos])) if per_pos else float('nan')
    out={'threshold':float(threshold),'f1':f1,'dice':f1,'iou':iou,'precision':pr,'recall':re,'f1_macro_image':macro('f1'),'iou_macro_image':macro('iou'),'precision_macro_image':macro('precision'),'recall_macro_image':macro('recall'),'f1_macro_positive_image':macro_pos('f1'),'iou_macro_positive_image':macro_pos('iou'),'cldice':float(np.mean(cls)) if cls else float('nan'),'boundary_f1':float(np.mean(bfs)) if bfs else float('nan'),'gt_foreground_ratio':gt_pos/max(total_px,1.0),'pred_foreground_ratio':pred_pos/max(total_px,1.0),'empty_gt_images':empty_gt,'empty_gt_false_positive_images':empty_gt_fp,'empty_gt_false_positive_rate':empty_gt_fp/max(empty_gt,1),'metric_aggregation':'f1/iou/precision/recall are global pixel-micro; *_macro_image are arithmetic means of per-image metrics'}
    for k in geoms[0] if geoms else []: out[k]=float(np.mean([g[k] for g in geoms]))
    return out

def calibrate_geometry_threshold_on_validation(model,loader,device,rasterizer,thresholds,seed=0,max_radius=16.):
    coll=collect_geometry(model,loader,device,rasterizer,seed); score_gt=[(prob,((gt_pm1+1)*.5)) for _,prob,gt_pm1 in coll]; rows=_micro_threshold_sweep_from_score_gt(score_gt,thresholds); best=max(rows,key=lambda t:rows[t]['f1']); rows[best]=aggregate_geometry(coll,best,max_radius,rasterizer.representation,rasterizer.distance_encoding,include_geometry=True,include_structural=True); rows[best]['calibration_mode']='exact_bucketized_pixel_micro_then_full_best_threshold'; return rows,best

@torch.no_grad()
def evaluate_geometry_with_frozen_threshold(model,loader,device,rasterizer,threshold,seed=0,max_radius=16.,collect_per_image=False):
    """Streaming one-step geometry evaluation; avoids retaining OOD tensors on GPU/CPU."""
    model.eval();gen=torch.Generator(device='cpu').manual_seed(int(seed));threshold=float(threshold)
    tp=fp=fn=tn=0.;cls=[];bfs=[];geoms=[];per=[];per_pos=[];per_records=[];gt_pos=pred_pos=total_px=0.;empty_gt=empty_gt_fp=0
    for b in loader:
        img=b['crack'].to(device);gt_pm1=b['mask'].to(device);z=torch.randn((img.shape[0],2,img.shape[-2],img.shape[-1]),generator=gen).to(device)
        geom,prob=sample_geometry_one_step(model,z,img,rasterizer);geom=geom.detach().cpu();prob=prob.detach().cpu();gt_pm1=gt_pm1.detach().cpu();gt=((gt_pm1+1)*.5);pred=(prob>threshold).float()
        m=compute_segmentation_metrics(pred,gt);tp+=m['tp'];fp+=m['fp'];fn+=m['fn'];tn+=m['tn'];gt_pos+=float(gt.sum());pred_pos+=float(pred.sum());total_px+=float(gt.numel())
        gg,_=mask_to_geometry_state(gt_pm1,max_radius,rasterizer.representation,rasterizer.distance_encoding);c_pred,r_pred=geometry_state_to_fields(geom,max_radius,rasterizer.representation,rasterizer.distance_encoding);c_gt,r_gt=geometry_state_to_fields(gg,max_radius,rasterizer.representation,rasterizer.distance_encoding)
        names=b.get('name'); names=list(names) if names is not None else [str(i) for i in range(gt.shape[0])]
        for bi in range(gt.shape[0]):
            mi=compute_segmentation_metrics(pred[bi:bi+1],gt[bi:bi+1]);per.append(mi);c=cldice_score(pred[bi:bi+1],gt[bi:bi+1]);bf=boundary_f1_score(pred[bi:bi+1],gt[bi:bi+1]);gm=geometry_centerline_metrics(c_pred[bi:bi+1],c_gt[bi:bi+1],r_pred[bi:bi+1],r_gt[bi:bi+1],max_radius);cls.append(c);bfs.append(bf);geoms.append(gm)
            if collect_per_image: per_records.append({'name':str(names[bi]),**{k:float(mi[k]) for k in ('f1','iou','precision','recall','tp','fp','fn','tn')},'cldice':float(c),'boundary_f1':float(bf),**{k:float(v) for k,v in gm.items()},'gt_foreground_pixels':float(gt[bi].sum()),'pred_foreground_pixels':float(pred[bi].sum()),'total_pixels':int(gt[bi].numel())})
            if float(gt[bi].sum())>0:per_pos.append(mi)
            else:
                empty_gt+=1
                if float(pred[bi].sum())>0:empty_gt_fp+=1
    if tp+fp+fn==0:pr=re=f1=iou=1.
    else:
        pr=tp/max(tp+fp,1e-12);re=tp/max(tp+fn,1e-12);f1=2*pr*re/max(pr+re,1e-12);iou=tp/max(tp+fp+fn,1e-12)
    macro=lambda key:float(np.mean([x[key] for x in per])) if per else float('nan');macro_pos=lambda key:float(np.mean([x[key] for x in per_pos])) if per_pos else float('nan')
    out={'threshold':threshold,'f1':f1,'dice':f1,'iou':iou,'precision':pr,'recall':re,'f1_macro_image':macro('f1'),'iou_macro_image':macro('iou'),'precision_macro_image':macro('precision'),'recall_macro_image':macro('recall'),'f1_macro_positive_image':macro_pos('f1'),'iou_macro_positive_image':macro_pos('iou'),'cldice':float(np.mean(cls)) if cls else float('nan'),'boundary_f1':float(np.mean(bfs)) if bfs else float('nan'),'gt_foreground_ratio':gt_pos/max(total_px,1.0),'pred_foreground_ratio':pred_pos/max(total_px,1.0),'empty_gt_images':empty_gt,'empty_gt_false_positive_images':empty_gt_fp,'empty_gt_false_positive_rate':empty_gt_fp/max(empty_gt,1),'metric_aggregation':'f1/iou/precision/recall are global pixel-micro; *_macro_image are arithmetic means of per-image metrics','evaluation_memory_mode':'streaming','threshold_source':'validation'}
    for k in geoms[0] if geoms else []:out[k]=float(np.mean([g[k] for g in geoms]))
    if collect_per_image: out['per_image']=per_records
    return out
