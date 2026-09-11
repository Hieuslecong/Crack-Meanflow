from __future__ import annotations

"""CPU diagnostic probes for V3 scientific branch reachability.

These probes do not estimate paper metrics. They verify that the preregistered
Conference endpoint/thin branch, A2B endpoint branch, and A5 endpoint/GIC
sampling branches are actually reachable under the locked V3 schedules.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
import yaml

from crackmeanflow.conference.losses import ConferenceMeanFlowLoss
from crackmeanflow.journal.flow.improved_meanflow import ImprovedMeanFlowGeometryLoss, ImprovedMeanFlowStateLoss


class _DummyConferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self._seg = None

    def forward(self, z, r, t, y=None):
        del r, t
        self._seg = z[:, :1] * self.scale
        cond = 0.0 if y is None else y[:, :1].mean(dim=1, keepdim=True) * 0.0
        return z * self.scale + cond

    def get_seg_logits(self):
        return self._seg


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/protocol/post_repair_protocol_v3.yaml")
    ap.add_argument("--out", default="reports/BRANCH_COVERAGE_V3.json")
    args = ap.parse_args()

    protocol = _load(args.protocol)
    conf = _load(protocol["primary_arms"]["Conference"])
    a2b = _load(protocol["primary_arms"]["A2B_ENDPOINT"])
    a5 = _load(protocol["primary_arms"]["A5_ENDPOINT"])
    device = torch.device("cpu")

    # Conference: probe a batch inside the locked endpoint stage. This executes
    # the actual endpoint and thin loss branches with a JVP-compatible dummy model.
    conf_loss = ConferenceMeanFlowLoss(**conf["loss"])
    endpoint_start = int(protocol["conference_curriculum"]["stages"][-1]["optimizer_steps"][0])
    effective_batch = int(protocol["conference_curriculum"]["effective_batch_size"])
    sample_offset = endpoint_start * effective_batch
    b = 32
    x0 = torch.where(torch.rand(b, 1, 8, 8) > 0.8, torch.ones(b, 1, 8, 8), -torch.ones(b, 1, 8, 8))
    image = torch.randn(b, 3, 8, 8)
    dummy = _DummyConferenceModel()
    conf_total, conf_logs = conf_loss(dummy, x0, {"y": image, "sample_offset": sample_offset})
    conference = {
        "sample_offset": sample_offset,
        "curriculum_position": int(conf_logs["curriculum_position"]),
        "boundary_probability": float(conf_logs["boundary_prob"]),
        "boundary_count": int(conf_logs["boundary_count"]),
        "endpoint_loss": float(conf_logs["endpoint_loss"]),
        "thin_loss": float(conf_logs["thin_loss"]),
        "total_finite": bool(torch.isfinite(conf_total).all()),
    }
    conference["pass"] = bool(
        conference["boundary_probability"] > 0
        and conference["boundary_count"] > 0
        and conference["endpoint_loss"] > 0
        and conference["thin_loss"] > 0
        and conference["total_finite"]
    )

    # A2B: exact-rate disjoint endpoint sampling must be active.
    a2b_loss = ImprovedMeanFlowStateLoss(**a2b["loss"])
    _, _, fm_a2b, endpoint_a2b = a2b_loss._sample(1000, device, sample_offset=0)
    a2b_result = {
        "fm_count": int(fm_a2b.sum().item()),
        "endpoint_count": int(endpoint_a2b.sum().item()),
        "configured_endpoint_probability": float(a2b["loss"]["endpoint_probability"]),
    }
    a2b_result["pass"] = bool(a2b_result["endpoint_count"] > 0 and a2b_result["fm_count"] > 0)

    # A5: endpoint sampling and the independent GIC mask must both be active.
    a5_kwargs = dict(a5["loss"])
    a5_kwargs["rasterizer"] = None
    a5_loss = ImprovedMeanFlowGeometryLoss(**a5_kwargs)
    _, _, fm_a5, endpoint_a5 = a5_loss.sample_tr(1000, device, sample_offset=0)
    gic_a5 = a5_loss._gic_mask(1000, device, sample_offset=0)
    a5_result = {
        "fm_count": int(fm_a5.sum().item()),
        "endpoint_count": int(endpoint_a5.sum().item()),
        "gic_count": int(gic_a5.sum().item()),
        "configured_endpoint_probability": float(a5["loss"]["endpoint_probability"]),
        "configured_gic_probability": float(a5["loss"]["gic_probability"]),
    }
    a5_result["pass"] = bool(
        a5_result["endpoint_count"] > 0 and a5_result["fm_count"] > 0 and a5_result["gic_count"] > 0
    )

    report = {
        "schema": "CRACKMEANFLOW_BRANCH_COVERAGE_V3",
        "status": "PASS" if conference["pass"] and a2b_result["pass"] and a5_result["pass"] else "FAIL",
        "diagnostic_only": True,
        "research_metric_valid": False,
        "conference": conference,
        "a2b": a2b_result,
        "a5": a5_result,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if report["status"] != "PASS":
        raise RuntimeError(f"V3 branch coverage failed: {report}")
    print(json.dumps({"status": "PASS", "out": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
