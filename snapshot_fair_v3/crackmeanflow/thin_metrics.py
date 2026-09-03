"""Compatibility thin-structure metrics using deterministic skeletonization and real Boundary-F1."""
import numpy as np
import torch
from skimage.morphology import skeletonize
from crackmeanflow.common.metrics import boundary_f1_score

def skeletonize_or_thin_mask(mask:torch.Tensor)->torch.Tensor:
    x=(mask.float()>0.5).float(); out=[]
    for s in x:
        sk=skeletonize(s[0].detach().cpu().numpy().astype(bool)).astype(np.float32); out.append(torch.from_numpy(sk)[None])
    return torch.stack(out).to(mask.device)

def compute_thin_crack_metrics(pred_binary,mask_gt,eps=1e-7):
    pred=(pred_binary.float()>0.5).float(); gt=(mask_gt.float()>0.5).float(); ps=skeletonize_or_thin_mask(pred); gs=skeletonize_or_thin_mask(gt)
    tp=(ps*gs).sum();fp=(ps*(1-gs)).sum();fn=((1-ps)*gs).sum()
    if ps.sum()==0 and gs.sum()==0: pr=re=f1=torch.tensor(1.,device=pred.device)
    else: pr=tp/(tp+fp+eps);re=tp/(tp+fn+eps);f1=2*pr*re/(pr+re+eps)
    bf=boundary_f1_score(pred,gt)
    return {'thin_recall':float(re.cpu()),'thin_precision':float(pr.cpu()),'thin_f1':float(f1.cpu()),'boundary_f1':float(bf),'recall_thin':float(re.cpu()),'f1_thin':float(f1.cpu()),'dice_thin':float(f1.cpu())}
