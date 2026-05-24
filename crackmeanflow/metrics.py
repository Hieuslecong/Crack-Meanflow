import torch


def _safe_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def compute_segmentation_metrics(pred_binary, mask_gt, eps=1e-7):
    pred = (pred_binary.float() > 0.5).float().view(-1)
    gt = (mask_gt.float() > 0.5).float().view(-1)
    tp = (pred * gt).sum()
    fp = (pred * (1.0 - gt)).sum()
    fn = ((1.0 - pred) * gt).sum()
    tn = ((1.0 - pred) * (1.0 - gt)).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    dice = f1
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    if gt.sum() == 0 and pred.sum() == 0:
        precision = recall = f1 = dice = iou = accuracy.new_tensor(1.0)
    return {
        "iou": _safe_float(iou),
        "dice": _safe_float(dice),
        "f1": _safe_float(f1),
        "precision": _safe_float(precision),
        "recall": _safe_float(recall),
        "accuracy": _safe_float(accuracy),
        "tp": _safe_float(tp),
        "fp": _safe_float(fp),
        "fn": _safe_float(fn),
        "tn": _safe_float(tn),
    }
