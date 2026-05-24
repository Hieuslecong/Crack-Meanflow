from pathlib import Path
from typing import Dict

import torch


def strip_prefix_if_needed(state_dict: Dict[str, torch.Tensor], prefixes=("module.", "unet.")) -> Dict[str, torch.Tensor]:
    out = dict(state_dict)
    for prefix in prefixes:
        if out and all(k.startswith(prefix) for k in out.keys()):
            out = {k[len(prefix):]: v for k, v in out.items()}
    return out


def has_nan_weights(state_dict: Dict[str, torch.Tensor]) -> bool:
    for value in state_dict.values():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            return True
    return False


def _model_state(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def save_checkpoint(path, model, optimizer=None, scheduler=None, ema=None, epoch=0, global_step=0, best_metrics=None, config=None, architecture=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": _model_state(model),
        "ema": _model_state(ema) if ema is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metrics": best_metrics or {},
        "config": config or {},
        "architecture": architecture or {},
    }
    if has_nan_weights(ckpt["model"]):
        raise RuntimeError("Refusing to save checkpoint with NaN/Inf model weights.")
    torch.save(ckpt, path)
    return str(path)


def _try_load(model, state):
    result = model.load_state_dict(state, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys)


def load_checkpoint_strict(model, ckpt_path, key="model", map_location="cpu", allow_missing=False, allow_unexpected=False):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if key not in ckpt or ckpt[key] is None:
        raise KeyError(f"Checkpoint {ckpt_path} has no key {key!r}.")
    raw_state = ckpt[key]
    if has_nan_weights(raw_state):
        raise RuntimeError(f"Checkpoint {ckpt_path}:{key} contains NaN/Inf weights.")

    candidates = [raw_state]
    module_stripped = strip_prefix_if_needed(raw_state, prefixes=("module.",))
    if module_stripped is not raw_state:
        candidates.append(module_stripped)
    fully_stripped = strip_prefix_if_needed(raw_state)
    if fully_stripped != module_stripped:
        candidates.append(fully_stripped)

    best = None
    for candidate in candidates:
        missing, unexpected = _try_load(model, candidate)
        score = len(missing) + len(unexpected)
        if best is None or score < best[0]:
            best = (score, missing, unexpected, candidate)
        if score == 0:
            break
    _, missing, unexpected, _ = best
    if (missing and not allow_missing) or (unexpected and not allow_unexpected):
        raise RuntimeError(f"Strict load failed for {ckpt_path}:{key}. missing={missing} unexpected={unexpected}")
    return ckpt, {"missing": missing, "unexpected": unexpected}


def audit_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    report = {"path": str(ckpt_path), "keys": sorted(list(ckpt.keys()))}
    for key in ("model", "ema"):
        if key in ckpt and ckpt[key] is not None:
            report[f"{key}_num_tensors"] = len(ckpt[key])
            report[f"{key}_has_nan_or_inf"] = has_nan_weights(ckpt[key])
    return report
