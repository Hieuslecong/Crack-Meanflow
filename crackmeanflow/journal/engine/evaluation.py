from __future__ import annotations
import numpy as np, torch
from crackmeanflow.common.metrics import compute_segmentation_metrics,cldice_score,boundary_f1_score,geometry_centerline_metrics
from crackmeanflow.journal.geometry.targets import mask_to_geometry_state,geometry_state_to_fields
from .sampler import sample_geometry_one_step

@torch.no_grad()
def collect_geometry(model,loader,device,rasterizer,seed=0):
    model.eval(); gen=torch.Generator(device='cpu').manual_seed(int(seed)); rows=[]
    for b in loader:
        img=b['crack'].to(device); gt_pm1=b['mask'].to(device); z=torch.randn((img.shape[0],2,img.shape[-2],img.shape[-1]),generator=gen).to(device)
        geom,prob=sample_geometry_one_step(model,z,img,rasterizer); rows.append((geom.detach(),prob.detach(),gt_pm1.detach()))
    return rows

def aggregate_geometry(collected,threshold,max_radius,representation,distance_encoding='linear',include_geometry=True):
    tp=fp=fn=tn=0.; cls=[]; bfs=[]; geoms=[]
    for geom,prob,gt_pm1 in collected:
        gt=((gt_pm1+1)*.5); pred=(prob>float(threshold)).float(); m=compute_segmentation_metrics(pred,gt); tp+=m['tp'];fp+=m['fp'];fn+=m['fn'];tn+=m['tn'];cls.append(cldice_score(pred,gt));bfs.append(boundary_f1_score(pred,gt))
        if include_geometry:
            gg,_=mask_to_geometry_state(gt_pm1,max_radius,representation,distance_encoding); c_pred,r_pred=geometry_state_to_fields(geom,max_radius,representation,distance_encoding); c_gt,r_gt=geometry_state_to_fields(gg,max_radius,representation,distance_encoding); geoms.append(geometry_centerline_metrics(c_pred,c_gt,r_pred,r_gt,max_radius))
    pr=tp/max(tp+fp,1e-12); re=tp/max(tp+fn,1e-12); f1=2*pr*re/max(pr+re,1e-12); iou=tp/max(tp+fp+fn,1e-12) if tp+fp+fn else 1.0
    out={'threshold':float(threshold),'f1':f1,'dice':f1,'iou':iou,'precision':pr,'recall':re,'cldice':float(np.mean(cls)),'boundary_f1':float(np.mean(bfs))}
    for k in geoms[0] if geoms else []: out[k]=float(np.mean([g[k] for g in geoms]))
    return out

def calibrate_geometry_threshold_on_validation(model,loader,device,rasterizer,thresholds,seed=0,max_radius=16.):
    coll=collect_geometry(model,loader,device,rasterizer,seed); rows={float(t):aggregate_geometry(coll,t,max_radius,rasterizer.representation,rasterizer.distance_encoding,include_geometry=False) for t in thresholds}; best=max(rows,key=lambda t:rows[t]['f1']); rows[best]=aggregate_geometry(coll,best,max_radius,rasterizer.representation,rasterizer.distance_encoding,include_geometry=True); return rows,best

def evaluate_geometry_with_frozen_threshold(model,loader,device,rasterizer,threshold,seed=0,max_radius=16.):
    out=aggregate_geometry(collect_geometry(model,loader,device,rasterizer,seed),threshold,max_radius,rasterizer.representation,rasterizer.distance_encoding); out['threshold_source']='validation'; return out
