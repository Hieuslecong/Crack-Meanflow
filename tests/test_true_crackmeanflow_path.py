import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crackmeanflow.adapter import CrackMeanFlowModel
from crackmeanflow.sampler import crack_meanflow_sampler


class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.calls = []

    def forward(self, x, t_int, y):
        self.calls.append((x.detach().clone(), t_int.detach().clone(), y.detach().clone()))
        velocity = x * self.weight + y[:, :1] * 0.25
        seg_logits = y[:, :1] - x
        return velocity, seg_logits


def test_adapter_uses_unet_velocity_path_and_caches_seg_logits():
    unet = TinyUNet()
    model = CrackMeanFlowModel(unet, T=10)
    x = torch.ones(2, 1, 4, 4)
    image = torch.full((2, 3, 4, 4), 4.0)

    out = model(x, r=torch.zeros(2), t=torch.ones(2), y=image)

    assert len(unet.calls) == 1
    _x_arg, t_arg, y_arg = unet.calls[0]
    assert torch.equal(t_arg, torch.full((2,), 9, dtype=torch.long))
    assert torch.equal(y_arg, image)
    assert torch.allclose(out, torch.full_like(x, 3.0))
    assert torch.count_nonzero(out).item() > 0
    assert torch.allclose(model.get_seg_logits(), torch.full_like(x, 3.0))


def test_one_step_sampler_returns_z_minus_velocity_and_seg_logits():
    model = CrackMeanFlowModel(TinyUNet(), T=10)
    z = torch.ones(1, 1, 4, 4)
    image = torch.full((1, 3, 4, 4), 4.0)

    sampled, seg_logits = crack_meanflow_sampler(model, z, image, num_steps=1, clamp=False)

    assert torch.allclose(sampled, torch.full_like(z, -2.0))
    assert torch.allclose(seg_logits, torch.full_like(z, 3.0))


def test_default_config_disables_direct_unet_primary_path():
    cfg = yaml.safe_load((ROOT / "configs" / "crackmeanflow_default.yaml").read_text())

    assert cfg["model"].get("use_direct_unet") is False
    assert "direct_unet_base" not in cfg["model"]
    assert cfg["loss"]["mode"] == "hybrid"


def test_omni_config_uses_existing_split_csv_paths_when_present():
    cfg_path = ROOT / "configs" / "crackmeanflow_omnicrack30k_mf01_l1_dice.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    for key in ["train_csv", "val_csv", "test_csv"]:
        assert key in cfg["paths"]
        assert (ROOT / cfg["paths"][key]).exists()
    assert cfg["model"].get("use_direct_unet") is False
    assert cfg["eval"]["num_steps"] == 1


if __name__ == "__main__":
    test_adapter_uses_unet_velocity_path_and_caches_seg_logits()
    test_one_step_sampler_returns_z_minus_velocity_and_seg_logits()
    test_default_config_disables_direct_unet_primary_path()
    test_omni_config_uses_existing_split_csv_paths_when_present()
    print("TRUE_CRACKMEANFLOW_PATH_TESTS_PASS")
