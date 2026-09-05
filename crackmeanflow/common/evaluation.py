from __future__ import annotations
import numpy as np
import torch
from .metrics import compute_segmentation_metrics, cldice_score, boundary_f1_score

@torch.no_grad()
def _collect(model, loader, device, sampler, num_steps, seed, cfg_scale=1.0):
    model.eval(); gen=torch.Generator(device='cpu').manual_seed(int(seed)); outs=[]
    for batch in loader:
        img=batch['crack'].to(device); gt_pm1=batch['mask'].to(device); z=torch.randn(gt_pm1.shape,generator=gen).to(device)
        pred_pm1,_=sampler(model,z,img,num_steps=num_steps,cfg_scale=cfg_scale,clamp=False); outs.append((pred_pm1.detach().cpu(), ((gt_pm1+1)*.5).detach().cpu()))
    return outs

def _aggregate(collected, threshold, include_structural=True):
    tp=fp=fn=tn=0.; cls=[]; bfs=[]; per=[]; per_pos=[]; gt_pos=pred_pos=total_px=0.; empty_gt=empty_gt_fp=0
    for pred_pm1, gt in collected:
        pred=(pred_pm1>threshold).float(); m=compute_segmentation_metrics(pred,gt); tp+=m['tp']; fp+=m['fp']; fn+=m['fn']; tn+=m['tn']
        gt_pos += float(gt.sum().detach().cpu()); pred_pos += float(pred.sum().detach().cpu()); total_px += float(gt.numel())
        for bi in range(gt.shape[0]):
            mi=compute_segmentation_metrics(pred[bi:bi+1],gt[bi:bi+1]); per.append(mi);
            if include_structural: cls.append(cldice_score(pred[bi:bi+1],gt[bi:bi+1])); bfs.append(boundary_f1_score(pred[bi:bi+1],gt[bi:bi+1]))
            if float(gt[bi].sum().detach().cpu())>0: per_pos.append(mi)
            else:
                empty_gt += 1
                if float(pred[bi].sum().detach().cpu())>0: empty_gt_fp += 1
    if tp+fp+fn==0: pr=re=f1=iou=1.
    else:
        pr=tp/max(tp+fp,1e-12); re=tp/max(tp+fn,1e-12); f1=2*pr*re/max(pr+re,1e-12); iou=tp/max(tp+fp+fn,1e-12)
    macro=lambda key: float(np.mean([x[key] for x in per])) if per else float('nan')
    macro_pos=lambda key: float(np.mean([x[key] for x in per_pos])) if per_pos else float('nan')
    return {
        'threshold':float(threshold),'f1':f1,'dice':f1,'iou':iou,'precision':pr,'recall':re,
        'f1_macro_image':macro('f1'),'iou_macro_image':macro('iou'),'precision_macro_image':macro('precision'),'recall_macro_image':macro('recall'),
        'f1_macro_positive_image':macro_pos('f1'),'iou_macro_positive_image':macro_pos('iou'),
        'cldice':(sum(cls)/len(cls) if cls else float('nan')),'boundary_f1':(sum(bfs)/len(bfs) if bfs else float('nan')),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
        'gt_foreground_ratio':gt_pos/max(total_px,1.0),'pred_foreground_ratio':pred_pos/max(total_px,1.0),
        'empty_gt_images':empty_gt,'empty_gt_false_positive_images':empty_gt_fp,'empty_gt_false_positive_rate':empty_gt_fp/max(empty_gt,1),
        'metric_aggregation':'f1/iou/precision/recall are global pixel-micro; *_macro_image are arithmetic means of per-image metrics'
    }


def _micro_threshold_sweep_from_score_gt(score_gt_pairs, thresholds):
    """Exact pixel-micro confusion/F1 for many strict ``score > threshold`` cuts.

    Uses one bucketization pass per collected batch, avoiding a threshold x pixel
    boolean tensor and repeated full-image scans. Thresholds must be sorted and
    unique. Equality follows the evaluator's strict ``>`` semantics.
    """
    ts=[float(x) for x in thresholds]
    if not ts or ts!=sorted(ts) or len(ts)!=len(set(ts)):
        raise ValueError('thresholds must be non-empty, sorted and unique')
    b=torch.tensor(ts,dtype=torch.float64)
    pos_bins=torch.zeros(len(ts)+1,dtype=torch.float64)
    neg_bins=torch.zeros(len(ts)+1,dtype=torch.float64)
    for score,gt in score_gt_pairs:
        sc=score.detach().cpu().double().reshape(-1); gg=(gt.detach().cpu().reshape(-1)>.5)
        if sc.numel()!=gg.numel(): raise RuntimeError('score/GT size mismatch in threshold sweep')
        idx=torch.bucketize(sc,b,right=False)
        pos_bins += torch.bincount(idx[gg],minlength=len(ts)+1).double()
        neg_bins += torch.bincount(idx[~gg],minlength=len(ts)+1).double()
    pos_total=float(pos_bins.sum()); neg_total=float(neg_bins.sum())
    pos_tail=torch.flip(torch.cumsum(torch.flip(pos_bins,dims=[0]),dim=0),dims=[0])
    neg_tail=torch.flip(torch.cumsum(torch.flip(neg_bins,dims=[0]),dim=0),dims=[0])
    rows={}
    for j,t in enumerate(ts):
        tp=float(pos_tail[j+1]); fp=float(neg_tail[j+1]); fn=pos_total-tp; tn=neg_total-fp
        if tp+fp+fn==0:
            pr=re=f1=iou=1.0
        else:
            pr=tp/max(tp+fp,1e-12); re=tp/max(tp+fn,1e-12); f1=2*pr*re/max(pr+re,1e-12); iou=tp/max(tp+fp+fn,1e-12)
        rows[t]={'threshold':t,'f1':f1,'dice':f1,'iou':iou,'precision':pr,'recall':re,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'calibration_mode':'exact_bucketized_pixel_micro'}
    return rows

def calibrate_threshold_on_validation(model, loader, device, sampler, thresholds, num_steps=1, seed=0, cfg_scale=1.0):
    coll=_collect(model,loader,device,sampler,num_steps,seed,cfg_scale); rows=_micro_threshold_sweep_from_score_gt(coll,thresholds); best=max(rows,key=lambda t: rows[t]['f1']); rows[best]=_aggregate(coll,float(best),include_structural=True); rows[best]['calibration_mode']='exact_bucketized_pixel_micro_then_full_best_threshold'; return rows,best

@torch.no_grad()
def evaluate_with_threshold(model, loader, device, sampler, threshold, num_steps=1, seed=0, cfg_scale=1.0, collect_per_image=False):
    """Streaming fixed-threshold evaluation. Keeps only one batch of predictions resident."""
    model.eval(); gen=torch.Generator(device='cpu').manual_seed(int(seed)); threshold=float(threshold)
    tp=fp=fn=tn=0.; cls=[]; bfs=[]; per=[]; per_pos=[]; per_records=[]; gt_pos=pred_pos=total_px=0.; empty_gt=empty_gt_fp=0
    for batch in loader:
        img=batch['crack'].to(device); gt_pm1=batch['mask'].to(device); z=torch.randn(gt_pm1.shape,generator=gen).to(device)
        pred_pm1,_=sampler(model,z,img,num_steps=num_steps,cfg_scale=cfg_scale,clamp=False)
        pred=(pred_pm1.detach().cpu()>threshold).float(); gt=((gt_pm1.detach().cpu()+1)*.5)
        m=compute_segmentation_metrics(pred,gt); tp+=m['tp']; fp+=m['fp']; fn+=m['fn']; tn+=m['tn']; gt_pos+=float(gt.sum()); pred_pos+=float(pred.sum()); total_px+=float(gt.numel())
        names=batch.get('name'); names=list(names) if names is not None else [str(i) for i in range(gt.shape[0])]
        for bi in range(gt.shape[0]):
            mi=compute_segmentation_metrics(pred[bi:bi+1],gt[bi:bi+1]); per.append(mi); c=cldice_score(pred[bi:bi+1],gt[bi:bi+1]); bf=boundary_f1_score(pred[bi:bi+1],gt[bi:bi+1]); cls.append(c); bfs.append(bf)
            if collect_per_image: per_records.append({'name':str(names[bi]),**{k:float(mi[k]) for k in ('f1','iou','precision','recall','tp','fp','fn','tn')},'cldice':float(c),'boundary_f1':float(bf),'gt_foreground_pixels':float(gt[bi].sum()),'pred_foreground_pixels':float(pred[bi].sum()),'total_pixels':int(gt[bi].numel())})
            if float(gt[bi].sum())>0: per_pos.append(mi)
            else:
                empty_gt+=1
                if float(pred[bi].sum())>0: empty_gt_fp+=1
    if tp+fp+fn==0: pr=re=f1=iou=1.
    else:
        pr=tp/max(tp+fp,1e-12); re=tp/max(tp+fn,1e-12); f1=2*pr*re/max(pr+re,1e-12); iou=tp/max(tp+fp+fn,1e-12)
    macro=lambda key:float(np.mean([x[key] for x in per])) if per else float('nan'); macro_pos=lambda key:float(np.mean([x[key] for x in per_pos])) if per_pos else float('nan')
    out={'threshold':threshold,'f1':f1,'dice':f1,'iou':iou,'precision':pr,'recall':re,'f1_macro_image':macro('f1'),'iou_macro_image':macro('iou'),'precision_macro_image':macro('precision'),'recall_macro_image':macro('recall'),'f1_macro_positive_image':macro_pos('f1'),'iou_macro_positive_image':macro_pos('iou'),'cldice':float(np.mean(cls)) if cls else float('nan'),'boundary_f1':float(np.mean(bfs)) if bfs else float('nan'),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'gt_foreground_ratio':gt_pos/max(total_px,1.0),'pred_foreground_ratio':pred_pos/max(total_px,1.0),'empty_gt_images':empty_gt,'empty_gt_false_positive_images':empty_gt_fp,'empty_gt_false_positive_rate':empty_gt_fp/max(empty_gt,1),'metric_aggregation':'f1/iou/precision/recall are global pixel-micro; *_macro_image are arithmetic means of per-image metrics','evaluation_memory_mode':'streaming'}
    if collect_per_image: out['per_image']=per_records
    return out

def evaluate_test_with_frozen_threshold(model, loader, device, sampler, validation_threshold, num_steps=1, seed=0, cfg_scale=1.0, collect_per_image=False):
    result=evaluate_with_threshold(model,loader,device,sampler,validation_threshold,num_steps,seed,cfg_scale,collect_per_image=collect_per_image); result['threshold_source']='validation'; return result
