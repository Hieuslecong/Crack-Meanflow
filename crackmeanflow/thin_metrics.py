import torch
import torch.nn.functional as F


def _erode(mask: torch.Tensor) -> torch.Tensor:
    inv = 1.0 - mask.float()
    dilated_inv = F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
    return 1.0 - dilated_inv


def skeletonize_or_thin_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = (mask.float() > 0.5).float()
    try:
        from skimage.morphology import skeletonize

        out = []
        for sample in mask:
            arr = sample[0].detach().cpu().numpy() > 0
            skel = skeletonize(arr).astype("float32")
            out.append(torch.from_numpy(skel).to(mask.device).unsqueeze(0))
        return torch.stack(out, dim=0)
    except Exception:
        eroded = _erode(mask)
        thin = torch.clamp(mask - eroded, min=0.0, max=1.0)
        if thin.sum() == 0:
            thin = mask
        return thin


def compute_thin_crack_metrics(pred_binary: torch.Tensor, mask_gt: torch.Tensor, eps: float = 1e-7):
    pred = (pred_binary.float() > 0.5).float()
    gt = (mask_gt.float() > 0.5).float()
    pred_thin = skeletonize_or_thin_mask(pred)
    gt_thin = skeletonize_or_thin_mask(gt)
    tp = (pred_thin * gt_thin).sum()
    fp = (pred_thin * (1.0 - gt_thin)).sum()
    fn = ((1.0 - pred_thin) * gt_thin).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    boundary_f1 = f1
    if gt_thin.sum() == 0 and pred_thin.sum() == 0:
        precision = recall = f1 = boundary_f1 = precision.new_tensor(1.0)
    return {
        "thin_recall": float(recall.detach().cpu().item()),
        "thin_precision": float(precision.detach().cpu().item()),
        "thin_f1": float(f1.detach().cpu().item()),
        "boundary_f1": float(boundary_f1.detach().cpu().item()),
        "recall_thin": float(recall.detach().cpu().item()),
        "f1_thin": float(f1.detach().cpu().item()),
        "dice_thin": float(f1.detach().cpu().item()),
    }
