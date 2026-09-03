from __future__ import annotations
import torch
from .metrics import compute_segmentation_metrics, cldice_score, boundary_f1_score

@torch.no_grad()
def _collect(model, loader, device, sampler, num_steps, seed, cfg_scale=1.0):
    model.eval(); gen=torch.Generator(device='cpu').manual_seed(int(seed)); outs=[]
    for batch in loader:
        img=batch['crack'].to(device); gt_pm1=batch['mask'].to(device); z=torch.randn(gt_pm1.shape,generator=gen).to(device)
        pred_pm1,_=sampler(model,z,img,num_steps=num_steps,cfg_scale=cfg_scale,clamp=False); outs.append((pred_pm1.detach(), ((gt_pm1+1)*.5).detach()))
    return outs

def _aggregate(collected, threshold):
    tp=fp=fn=tn=0.; cls=[]; bfs=[]
    for pred_pm1, gt in collected:
        pred=(pred_pm1>threshold).float(); m=compute_segmentation_metrics(pred,gt); tp+=m['tp']; fp+=m['fp']; fn+=m['fn']; tn+=m['tn']; cls.append(cldice_score(pred,gt)); bfs.append(boundary_f1_score(pred,gt))
    if tp+fp+fn==0: pr=re=f1=iou=1.
    else:
        pr=tp/max(tp+fp,1e-12); re=tp/max(tp+fn,1e-12); f1=2*pr*re/max(pr+re,1e-12); iou=tp/max(tp+fp+fn,1e-12)
    return {'threshold':float(threshold),'f1':f1,'dice':f1,'iou':iou,'precision':pr,'recall':re,'cldice':sum(cls)/max(len(cls),1),'boundary_f1':sum(bfs)/max(len(bfs),1),'tp':tp,'fp':fp,'fn':fn,'tn':tn}

def calibrate_threshold_on_validation(model, loader, device, sampler, thresholds, num_steps=1, seed=0, cfg_scale=1.0):
    coll=_collect(model,loader,device,sampler,num_steps,seed,cfg_scale); rows={float(t):_aggregate(coll,float(t)) for t in thresholds}; best=max(rows,key=lambda t: rows[t]['f1']); return rows,best

def evaluate_with_threshold(model, loader, device, sampler, threshold, num_steps=1, seed=0, cfg_scale=1.0):
    return _aggregate(_collect(model,loader,device,sampler,num_steps,seed,cfg_scale),float(threshold))

def evaluate_test_with_frozen_threshold(model, loader, device, sampler, validation_threshold, num_steps=1, seed=0, cfg_scale=1.0):
    result=evaluate_with_threshold(model,loader,device,sampler,validation_threshold,num_steps,seed,cfg_scale); result['threshold_source']='validation'; return result
