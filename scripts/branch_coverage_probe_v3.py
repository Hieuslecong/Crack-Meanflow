from __future__ import annotations

"""CPU diagnostic probes for V3 scientific branch reachability.

These probes do not estimate paper metrics. They execute the preregistered
Conference endpoint/thin branch, A2B endpoint branch, and A5 endpoint/GIC loss
branches using small JVP-compatible dummy models.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
import yaml

from crackmeanflow.conference.losses import ConferenceMeanFlowLoss
from crackmeanflow.journal.flow.improved_meanflow import ImprovedMeanFlowGeometryLoss, ImprovedMeanFlowStateLoss
from crackmeanflow.journal.geometry import GeometryRasterizer


class _DummyConferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self._seg = None

    def forward(self, z, r, t, y=None):
        del r, t
        self._seg = z[:, :1] * self.scale
        cond = 0.0 if y is None else y[:, :1] * 0.0
        return z * self.scale + cond

    def get_seg_logits(self):
        return self._seg


class _DummyIMFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def flow_outputs(self, z, t, r, image):
        del t, r
        cond = image[:, :1] * 0.0
        if cond.shape[1] != z.shape[1]:
            cond = cond.expand(-1, z.shape[1], -1, -1)
        return {
            "u": z * self.scale + cond,
            "v": z * (self.scale * 0.5) + cond,
            "clean_u": torch.tanh(z * self.scale + cond),
        }


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/protocol/post_repair_protocol_v3.yaml")
    ap.add_argument("--out", default="reports/BRANCH_COVERAGE_V3.json")
    args = ap.parse_args()

    torch.manual_seed(123)
    protocol = _load(args.protocol)
    conf = _load(protocol["primary_arms"]["Conference"])
    a2b = _load(protocol["primary_arms"]["A2B_ENDPOINT"])
    a5 = _load(protocol["primary_arms"]["A5_ENDPOINT"])
    device = torch.device("cpu")

    # Conference actual endpoint/thin loss execution inside the final step stage.
    conf_loss = ConferenceMeanFlowLoss(**conf["loss"])
    endpoint_start = int(protocol["conference_curriculum"]["stages"][-1]["optimizer_steps"][0])
    effective_batch = int(protocol["conference_curriculum"]["effective_batch_size"])
    sample_offset = endpoint_start * effective_batch
    b = 32
    x0 = torch.where(torch.rand(b, 1, 8, 8) > 0.8, torch.ones(b, 1, 8, 8), -torch.ones(b, 1, 8, 8))
    image = torch.randn(b, 3, 8, 8)
    conf_total, conf_logs = conf_loss(_DummyConferenceModel(), x0, {"y": image, "sample_offset": sample_offset})
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

    # A2B actual endpoint-aware IMF loss execution.
    a2b_loss = ImprovedMeanFlowStateLoss(**a2b["loss"])
    a2b_x0 = torch.where(torch.rand(b, 1, 8, 8) > 0.8, torch.ones(b, 1, 8, 8), -torch.ones(b, 1, 8, 8))
    a2b_total, a2b_logs = a2b_loss(_DummyIMFModel(), a2b_x0, image, sample_offset=0)
    a2b_result = {
        "fm_count": int(a2b_logs["fm_count"]),
        "endpoint_count": int(a2b_logs["exact_deployment_count"]),
        "near_deployment_count": int(a2b_logs["near_deployment_count"]),
        "total_finite": bool(torch.isfinite(a2b_total).all()),
        "configured_endpoint_probability": float(a2b["loss"]["endpoint_probability"]),
    }
    a2b_result["pass"] = bool(
        a2b_result["endpoint_count"] > 0 and a2b_result["fm_count"] > 0 and a2b_result["total_finite"]
    )

    # A5 actual geometry + endpoint + GIC loss execution.
    rasterizer = GeometryRasterizer(
        max_radius=a5["model"].get("max_radius", 16.0),
        bins=a5["model"].get("radius_bins", 8),
        representation=a5["model"].get("representation", "centerline_radius"),
        distance_encoding=a5["model"].get("distance_encoding", "linear"),
    )
    a5_loss = ImprovedMeanFlowGeometryLoss(
        **a5["loss"],
        max_radius=a5["model"].get("max_radius", 16.0),
        rasterizer=rasterizer,
    )
    center = torch.where(torch.rand(b, 1, 8, 8) > 0.85, torch.ones(b, 1, 8, 8), -torch.ones(b, 1, 8, 8))
    field = torch.rand(b, 1, 8, 8) * 2.0 - 1.0
    g0 = torch.cat([center, field], dim=1)
    mask_gt = center.clone()
    radius_valid = torch.ones_like(center)
    a5_total, a5_logs = a5_loss(
        _DummyIMFModel(), g0, image, radius_valid=radius_valid, mask_gt=mask_gt, sample_offset=0
    )
    a5_result = {
        "fm_count": int(a5_logs["fm_count"]),
        "endpoint_count": int(a5_logs["exact_deployment_count"]),
        "gic_count": int(a5_logs["gic_active_samples"]),
        "gic_loss": float(a5_logs["gic_loss"]),
        "total_finite": bool(torch.isfinite(a5_total).all()),
        "configured_endpoint_probability": float(a5["loss"]["endpoint_probability"]),
        "configured_gic_probability": float(a5["loss"]["gic_probability"]),
    }
    a5_result["pass"] = bool(
        a5_result["endpoint_count"] > 0
        and a5_result["fm_count"] > 0
        and a5_result["gic_count"] > 0
        and torch.isfinite(torch.tensor(a5_result["gic_loss"]))
        and a5_result["total_finite"]
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
