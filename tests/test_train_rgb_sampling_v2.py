import importlib.util
import json
from pathlib import Path
import sys
from copy import deepcopy

import pytest
import torch
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("sampling_trainer_v2", SCRIPTS / "train_rgb_sampling_v2.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("change", [("num_workers", 0), ("epochs", 3), ("samples_per_epoch", 100), ("early_stopping", True)])
def test_unequal_or_nonresumable_recipe_is_rejected(tmp_path, change):
    config = yaml.safe_load((SCRIPTS.parent / "configs/rgb_sampling_v2.yaml").read_text())
    config["training"][change[0]] = change[1]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        module.read_config(path)


def commit_fixture(directory):
    binding = {"test": "synthetic"}
    payload = {"binding": binding, "arm": "control", "promotion_eligible": False,
               "epoch": 1, "history": [{"epoch": 1}], "best_safe": {"epoch": 1, "macro_f1": .8},
               "model_state_dict": {"example": torch.tensor([1., 2.])}}
    module.write_checkpoint(directory, "checkpoint_epoch_001.pt", payload)
    digest = module.legacy.sha256(directory / "checkpoint_epoch_001.pt")
    module.legacy.atomic_json(directory / "commit.json", {"epoch": 1, "checkpoint": "checkpoint_epoch_001.pt", "sha256": digest, "best_safe_sha256": digest})
    return binding, digest


def test_committed_epoch_recovers_missing_or_corrupt_aliases(tmp_path):
    binding, digest = commit_fixture(tmp_path)
    (tmp_path / "checkpoint_latest.pt").write_bytes(b"interrupted alias")
    result = module.resume_payload(tmp_path, binding, "control")
    assert result["epoch"] == 1
    assert module.legacy.sha256(tmp_path / "checkpoint_latest.pt") == digest
    assert module.legacy.sha256(tmp_path / "checkpoint_best_safe.pt") == digest
    assert result["model_state_dict"]["example"].tolist() == [1, 2]


def test_uncommitted_epoch_is_not_treated_as_completed(tmp_path):
    module.write_checkpoint(tmp_path, "checkpoint_epoch_001.pt", {"epoch": 1})
    assert module.resume_payload(tmp_path, {}, "control") is None
    module.write_checkpoint(tmp_path, "checkpoint_epoch_001.pt", {"epoch": 1, "retry": True})
    assert len(list(tmp_path.glob("uncommitted_*_checkpoint_epoch_001.pt"))) == 1


def test_wrong_arm_or_changed_checkpoint_fail_closed(tmp_path):
    binding, _ = commit_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="Incompatible"):
        module.resume_payload(tmp_path, binding, "targeted")
    (tmp_path / "checkpoint_epoch_001.pt").write_bytes(b"tamper")
    with pytest.raises(RuntimeError, match="identity"):
        module.resume_payload(tmp_path, binding, "control")


def test_image_draws_do_not_depend_on_global_rng_or_arm(monkeypatch):
    def sampler(dataset, base, generator):
        return torch.randint(0, 11, (50,), generator=generator).tolist()
    monkeypatch.setattr(module, "build_train_sampler", sampler)
    first = module.epoch_indices("control", {}, 10, 1)
    torch.rand(999)
    assert first == module.epoch_indices("targeted", {}, 10, 1)
    assert first != module.epoch_indices("targeted", {}, 10, 2)


def test_loss_adapter_accepts_string_device_and_backpropagates():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 1)
        def forward(self, image):
            return {"logits": self.conv(image)}
    model = TinyModel()
    batch = {"image": torch.full((2, 3, 8, 8), .5), "labels": torch.zeros(2, 8, 8, dtype=torch.int64),
             "image_valid_mask": torch.ones(2, 8, 8, dtype=torch.bool), "classification_valid_mask": torch.ones(2, 8, 8, dtype=torch.bool)}
    criterion = module.legacy.SixClassFocalLoss([1.]*6, 1.5)
    loss = module.training_loss(model, batch, criterion, device="cpu")
    loss.backward()
    assert torch.isfinite(loss) and model.conv.weight.grad.abs().sum() > 0
