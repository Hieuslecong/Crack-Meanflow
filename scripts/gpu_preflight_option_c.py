from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crackmeanflow.common import EMA, environment_info, file_sha256, source_tree_hash
from crackmeanflow.common.option_c_provenance import (
    option_c_design_lock_sha256,
    validate_option_c_config,
)
from crackmeanflow.factory import build_training_components
from crackmeanflow.journal.flow.improved_meanflow import (
    _disjoint_stratified_schedule,
    _independent_stratified_mask,
)
from crackmeanflow.journal.geometry.targets import mask_to_geometry_state
from scripts.train_journal import seed_all

OPTION_C_CONFIGS = {
    'J0': 'configs/option_c_v1/j0_direct_mask.yaml',
    'J1': 'configs/option_c_v1/j1_centerline_edt_noncausal.yaml',
    'J2': 'configs/option_c_v1/j2_centerline_radius_causal.yaml',
    'J3': 'configs/option_c_v1/j3_centerline_radius_gic.yaml',
    'J4': 'configs/option_c_v1/j4_centerline_radius_gic_endpoint.yaml',
}


def _stress_offset(cfg):
    loss = cfg.get('loss') or {}
    batch = int(cfg['train']['batch_size'])
    need_gic = float(loss.get('gic_weight', 0)) > 0 and float(loss.get('gic_probability', 0)) > 0
    need_endpoint = float(loss.get('endpoint_probability', 0)) > 0
    for off in range(10000):
        gic_ok = True
        ep_ok = True
        if need_gic:
            gic = _independent_stratified_mask(
                batch,
                float(loss['gic_probability']),
                off,
                torch.device('cpu'),
                'gic',
            )
            gic_ok = bool(gic.any())
        if need_endpoint:
            if loss.get('endpoint_sampling') == 'stratified_disjoint':
                _, ep = _disjoint_stratified_schedule(
                    batch,
                    float(loss.get('data_proportion', .5)),
                    float(loss['endpoint_probability']),
                    off,
                    torch.device('cpu'),
                )
            else:
                ep = _independent_stratified_mask(
                    batch,
                    float(loss['endpoint_probability']),
                    off,
                    torch.device('cpu'),
                    'endpoint',
                )
            ep_ok = bool(ep.any())
        if gic_ok and ep_ok:
            return off
    raise RuntimeError('could not find an Option-C stress offset exercising all required stochastic branches')


def _run_variant(name, cfg, device, total_vram, max_reserved_fraction):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    seed_all(
        int(cfg['train'].get('seed', 42)),
        cfg['train'].get('deterministic', False),
        cfg['train'].get('deterministic_warn_only', False),
    )
    sample_offset = _stress_offset(cfg)
    result = {
        'variant': name,
        'backbone': cfg['backbone'],
        'image_size': int(cfg['model']['img_size']),
        'microbatch_size': int(cfg['train']['batch_size']),
        'grad_accum_steps': int(cfg['train']['grad_accum_steps']),
        'stress_sample_offset': int(sample_offset),
    }
    try:
        model, rast, lossfn = build_training_components(cfg, device)
        model.train()
        batch = int(cfg['train']['batch_size'])
        h = int(cfg['model']['img_size'])
        image = torch.rand(batch, 3, h, h, device=device)
        mask = (torch.rand(batch, 1, h, h, device=device) > .97).float() * 2 - 1
        if cfg['backbone'] == 'geocrack_imf':
            geom, radius_valid = mask_to_geometry_state(
                mask,
                cfg['model'].get('max_radius', 16),
                cfg['model'].get('representation', 'centerline_radius'),
                cfg['model'].get('distance_encoding', 'linear'),
            )
            loss, logs = lossfn(
                model,
                geom,
                image,
                radius_valid,
                mask_gt=mask,
                sample_offset=sample_offset,
            )
        elif cfg['backbone'] in {'sit_imf_mask', 'hybrid_imf_mask'}:
            loss, logs = lossfn(model, mask, image, sample_offset=sample_offset)
        else:
            loss, logs = lossfn(model, mask, {'y': image, 'sample_offset': sample_offset})

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg['train']['lr']),
            weight_decay=float(cfg['train']['weight_decay']),
        )
        ema = EMA(model, float(cfg['train']['ema_decay']))
        loss.backward()
        finite = bool(torch.isfinite(loss).item()) and all(
            p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg['train']['max_grad_norm'], error_if_nonfinite=True
        )
        opt.step()
        opt.zero_grad(set_to_none=True)
        ema.update(model)

        reserved = torch.cuda.max_memory_reserved(device)
        allocated = torch.cuda.max_memory_allocated(device)
        frac = reserved / max(total_vram, 1)
        need_gic = float((cfg.get('loss') or {}).get('gic_weight', 0)) > 0
        need_endpoint = float((cfg.get('loss') or {}).get('endpoint_probability', 0)) > 0
        gic_count = int(logs.get('gic_active_samples', 0))
        endpoint_count = int(logs.get('exact_deployment_count', logs.get('boundary_count', 0)))
        gic_ok = (not need_gic) or gic_count > 0
        endpoint_ok = (not need_endpoint) or endpoint_count > 0
        memory_ok = frac <= max_reserved_fraction
        status = 'PASS' if finite and memory_ok and h == 256 and gic_ok and endpoint_ok else 'FAIL'
        result.update(
            {
                'loss': float(loss.detach()),
                'grad_finite': finite,
                'grad_norm': float(grad_norm.detach().cpu()),
                'gic_branch_required': need_gic,
                'gic_active_samples': gic_count,
                'gic_branch_exercised': gic_ok,
                'endpoint_branch_required': need_endpoint,
                'exact_deployment_samples': endpoint_count,
                'endpoint_branch_exercised': endpoint_ok,
                'optimizer_state_allocated': True,
                'ema_state_allocated': True,
                'peak_allocated_gib': allocated / 2**30,
                'peak_reserved_gib': reserved / 2**30,
                'peak_reserved_fraction_of_vram': frac,
                'memory_headroom_pass': memory_ok,
                'status': status,
            }
        )
        del model, loss, image, mask
    except torch.cuda.OutOfMemoryError as exc:
        result.update({'status': 'FAIL_OOM', 'error': str(exc)})
    except Exception as exc:
        result.update(
            {
                'status': 'FAIL_EXCEPTION',
                'error': repr(exc),
                'traceback': traceback.format_exc(limit=8),
            }
        )
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser(
        description='Option-C V1 canonical 256x256 CUDA forward/backward/VRAM preflight for J0-J4.'
    )
    ap.add_argument('--out', default='reports/OPTION_C_V1_GPU_PREFLIGHT.json')
    ap.add_argument('--max-reserved-fraction', type=float, default=.90)
    ap.add_argument('--require-device-substring', default='RTX 3090')
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for Option-C GPU preflight')
    if not 0 < a.max_reserved_fraction < 1:
        raise ValueError('--max-reserved-fraction must lie in (0,1)')

    device = torch.device('cuda:0')
    device_name = torch.cuda.get_device_name(device)
    if a.require_device_substring and a.require_device_substring.lower() not in device_name.lower():
        raise RuntimeError(
            f'GPU mismatch: required substring={a.require_device_substring!r}, actual={device_name!r}'
        )
    total_vram = torch.cuda.get_device_properties(device).total_memory

    rows = []
    for variant, rel in OPTION_C_CONFIGS.items():
        path = ROOT / rel
        if not path.is_file():
            raise RuntimeError(f'missing Option-C config: {path}')
        cfg = yaml.safe_load(path.read_text())
        validate_option_c_config(cfg, variant)
        row = _run_variant(variant, cfg, device, total_vram, a.max_reserved_fraction)
        row.update({'config_path': rel, 'config_file_sha256': file_sha256(path)})
        rows.append(row)

    report = {
        'schema': 'CRACKMEANFLOW_OPTION_C_GPU_PREFLIGHT_V2',
        'protocol_id': 'OPTION_C_V1',
        'source_tree_sha256': source_tree_hash(),
        'design_lock_sha256': option_c_design_lock_sha256(ROOT),
        'preflight_script_sha256': file_sha256(Path(__file__)),
        'environment': environment_info(),
        'device': device_name,
        'compute_capability': list(torch.cuda.get_device_capability(device)),
        'total_vram_gib': total_vram / 2**30,
        'max_reserved_fraction_allowed': a.max_reserved_fraction,
        'variants': rows,
        'pass': all(r.get('status') == 'PASS' for r in rows),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report['pass']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
