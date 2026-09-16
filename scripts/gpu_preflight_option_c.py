from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crackmeanflow.common import environment_info, file_sha256, source_tree_hash
from crackmeanflow.common.option_c_provenance import (
    option_c_design_lock_sha256,
    validate_option_c_config,
)
from scripts.gpu_preflight import _run

OPTION_C_CONFIGS = {
    'J0': 'configs/option_c_v1/j0_direct_mask.yaml',
    'J1': 'configs/option_c_v1/j1_centerline_edt_noncausal.yaml',
    'J2': 'configs/option_c_v1/j2_centerline_radius_causal.yaml',
    'J3': 'configs/option_c_v1/j3_centerline_radius_gic.yaml',
    'J4': 'configs/option_c_v1/j4_centerline_radius_gic_endpoint.yaml',
}


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
        row = _run(variant, cfg, device, total_vram, a.max_reserved_fraction)
        row.update(
            {
                'variant': variant,
                'config_path': rel,
                'config_file_sha256': file_sha256(path),
            }
        )
        rows.append(row)

    report = {
        'schema': 'CRACKMEANFLOW_OPTION_C_GPU_PREFLIGHT_V1',
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
