from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize

def _bin(x):return (x.float()>.5).float()
def _scalar(x):return float(x.detach().cpu())
def compute_segmentation_metrics(pred_binary,mask_gt,eps=1e-7):
    p=_bin(pred_binary).view(-1);g=_bin(mask_gt).view(-1);tp=(p*g).sum();fp=(p*(1-g)).sum();fn=((1-p)*g).sum();tn=((1-p)*(1-g)).sum()
    if (g.sum()==0) and (p.sum()==0):return {k:1. for k in ['iou','dice','f1','precision','recall','accuracy']}|{'tp':0.,'fp':0.,'fn':0.,'tn':_scalar(tn)}
    pr=tp/(tp+fp+eps);re=tp/(tp+fn+eps);f1=2*pr*re/(pr+re+eps);iou=tp/(tp+fp+fn+eps);acc=(tp+tn)/(tp+tn+fp+fn+eps);return {'iou':_scalar(iou),'dice':_scalar(f1),'f1':_scalar(f1),'precision':_scalar(pr),'recall':_scalar(re),'accuracy':_scalar(acc),'tp':_scalar(tp),'fp':_scalar(fp),'fn':_scalar(fn),'tn':_scalar(tn)}
def _skeleton_batch(x):
    x=_bin(x);outs=[]
    for a in x:outs.append(torch.from_numpy(skeletonize(a[0].detach().cpu().numpy().astype(bool)).astype(np.float32))[None])
    return torch.stack(outs).to(x.device)
def cldice_score(pred_binary,mask_gt,eps=1e-7):
    p=_bin(pred_binary);g=_bin(mask_gt)
    if g.sum()==0 and p.sum()==0:return 1.
    sp=_skeleton_batch(p);sg=_skeleton_batch(g);tprec=(sp*g).sum()/(sp.sum()+eps);tsens=(sg*p).sum()/(sg.sum()+eps);return _scalar(2*tprec*tsens/(tprec+tsens+eps))
def _boundary(x):
    x=_bin(x);er=1-F.max_pool2d(1-x,3,1,1);return (x-er).clamp(0,1)
def boundary_f1_score(pred_binary,mask_gt,tolerance=2,eps=1e-7):
    p=_boundary(pred_binary);g=_boundary(mask_gt)
    if p.sum()==0 and g.sum()==0:return 1.
    k=2*int(tolerance)+1;gd=(F.max_pool2d(g,k,1,tolerance)>0).float();pd=(F.max_pool2d(p,k,1,tolerance)>0).float();pr=(p*gd).sum()/(p.sum()+eps);re=(g*pd).sum()/(g.sum()+eps);return _scalar(2*pr*re/(pr+re+eps))
def skeleton_length_px(binary):
    sk=_skeleton_batch(binary).float();total=0.
    for a in sk[:,0]:
        total+=float((a[:,:-1]*a[:,1:]).sum().cpu());total+=float((a[:-1,:]*a[1:,:]).sum().cpu());total+=2**.5*float((a[:-1,:-1]*a[1:,1:]).sum().cpu());total+=2**.5*float((a[:-1,1:]*a[1:,:-1]).sum().cpu())
    return total
def geometry_centerline_metrics(pred_center_prob,gt_center_binary,pred_radius=None,gt_radius=None,max_radius=16.):
    from scipy.spatial import cKDTree
    p=_bin(pred_center_prob);g=_bin(gt_center_binary);assd=[];rerrs=[];lerrs=[]
    for bi in range(p.shape[0]):
        pyx=np.argwhere(p[bi,0].detach().cpu().numpy()>.5);gyx=np.argwhere(g[bi,0].detach().cpu().numpy()>.5);diag=float((p.shape[-2]**2+p.shape[-1]**2)**.5)
        if len(pyx)==0 and len(gyx)==0:assd.append(0.);rerrs.append(0.)
        elif len(pyx)==0 or len(gyx)==0:assd.append(diag);rerrs.append(float(max_radius) if len(gyx) else 0.)
        else:
            pt,gt=cKDTree(pyx),cKDTree(gyx);dg,idx=pt.query(gyx,k=1);dp,_=gt.query(pyx,k=1);assd.append(float((dg.mean()+dp.mean())/2))
            if pred_radius is not None and gt_radius is not None:
                pr=pred_radius[bi,0].detach().cpu().numpy();gr=gt_radius[bi,0].detach().cpu().numpy();matched=pyx[idx];rerrs.append(float(np.mean(np.abs(pr[matched[:,0],matched[:,1]]-gr[gyx[:,0],gyx[:,1]]))))
            else:rerrs.append(float('nan'))
        lp=skeleton_length_px(p[bi:bi+1]);lg=skeleton_length_px(g[bi:bi+1]);lerrs.append(abs(lp-lg)/max(lg,1.))
    return {'centerline_assd_px':float(np.mean(assd)),'edt_radius_mae_px':float(np.nanmean(rerrs)) if not np.all(np.isnan(rerrs)) else float('nan'),'skeleton_length_rel_error':float(np.mean(lerrs))}
