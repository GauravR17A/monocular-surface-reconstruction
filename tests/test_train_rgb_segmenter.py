from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("test_rgb_trainer_module", ROOT / "scripts/train_rgb_segmenter.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_focal_matches_fixed_v3_weighted_pixel_mean():
    logits = torch.randn(2, 6, 4, 5, requires_grad=True)
    labels = torch.randint(6, (2, 4, 5))
    valid = torch.ones_like(labels, dtype=torch.bool)
    weights = [0.7, 0.6, 3.0, 0.7, 0.6, 0.4]
    loss = MODULE.SixClassFocalLoss(weights, 1.5)(logits, labels, valid, valid)
    ce = torch.nn.functional.cross_entropy(logits, labels, weight=torch.tensor(weights), reduction="none")
    p = logits.softmax(1).gather(1, labels[:, None])[:, 0]
    assert torch.allclose(loss, (ce * (1 - p).pow(1.5)).mean())
    loss.backward()
    assert logits.grad.abs().sum() > 0


def test_independent_masks_ignore_unknown_and_invalid_logits():
    logits = torch.randn(1, 6, 2, 2, requires_grad=True)
    logits.data[:, :, 0, 0] = float("nan")
    labels = torch.tensor([[[255, 2], [4, 5]]])
    image_valid = torch.tensor([[[False, True], [True, True]]])
    class_valid = torch.tensor([[[False, True], [False, True]]])
    result = MODULE.SixClassFocalLoss([1.] * 6, 0)(logits, labels, image_valid, class_valid)
    assert torch.isfinite(result)
    result.backward()
    assert torch.count_nonzero(logits.grad[:, :, 0, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, :, 1, 0]) == 0


def test_valid_unknown_is_rejected_and_empty_batch_is_differentiable():
    criterion = MODULE.SixClassFocalLoss([1.] * 6, 1.5)
    logits = torch.randn(1, 6, 2, 2, requires_grad=True)
    labels = torch.full((1, 2, 2), 255)
    valid = torch.ones_like(labels, dtype=torch.bool)
    with pytest.raises(ValueError, match="indices"):
        criterion(logits, labels, valid, valid)
    loss = criterion(logits, labels, valid, ~valid)
    assert loss == 0
    loss.backward()
    assert logits.grad is not None


def test_lr_warmup_and_cosine_boundaries():
    f = MODULE.learning_rate_factor
    assert f(0, 1000, 100, .1) == .01
    assert f(99, 1000, 100, .1) == 1
    assert f(100, 1000, 100, .1) == 1
    assert f(1000, 1000, 100, .1) == pytest.approx(.1)
    assert f(1200, 1000, 100, .1) == pytest.approx(.1)


def test_rng_restore_including_loader_generator():
    MODULE.seed_everything(45)
    generator = torch.Generator().manual_seed(7)
    state = MODULE.rng_state(generator)
    expected = (random.random(), np.random.rand(), torch.rand(2), torch.rand(2, generator=generator))
    MODULE.restore_rng(state, generator)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(2), expected[2])
    assert torch.equal(torch.rand(2, generator=generator), expected[3])


def test_atomic_replace_retries_windows_sharing_error(monkeypatch, tmp_path):
    source, target = tmp_path / "new", tmp_path / "old"
    source.write_text("new")
    target.write_text("old")
    actual = MODULE.os.replace
    calls = []
    def sometimes(a, b):
        calls.append(1)
        if len(calls) < 3:
            error = PermissionError("locked")
            error.winerror = 32
            raise error
        return actual(a, b)
    monkeypatch.setattr(MODULE.os, "replace", sometimes)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    MODULE.atomic_replace(source, target)
    assert target.read_text() == "new" and len(calls) == 3


def test_atomic_json_and_history_are_valid_and_forbid_nan(tmp_path):
    target = tmp_path / "status.json"
    MODULE.atomic_json(target, {"stage": "training"})
    assert MODULE.json.loads(target.read_text()) == {"stage": "training"}
    with pytest.raises(ValueError):
        MODULE.atomic_json(target, {"loss": float("nan")})
    assert MODULE.json.loads(target.read_text()) == {"stage": "training"}
    MODULE.write_history(tmp_path / "metrics.jsonl", [{"epoch": 1}, {"epoch": 2}])
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2


def test_config_keeps_recipe_isolated_and_full_native():
    config = MODULE.load_config(ROOT / "configs/gamus_rgb_segmenter_v1.yaml")
    assert config["protocol"]["auto_promotion"] is False
    assert config["model"]["pretrained_repository"] == "nvidia/mit-b0"
    assert config["data"]["validation_patch_size"] == 1024
    assert config["training"]["batch_size"] == 4
    assert config["training"]["epochs"] == 8


def test_resume_rejects_non_latest_or_other_experiment(tmp_path):
    with pytest.raises(ValueError, match="checkpoint_latest"):
        MODULE.verify_resume(tmp_path / "checkpoint_best_development.pt", {}, {})


def test_protected_pointer_utf8_bom_preserves_exact_hash_and_target(tmp_path):
    checkpoint = tmp_path / "protected.pt"
    checkpoint.write_bytes(b"protected checkpoint fixture")
    pointer = tmp_path / "showcase_checkpoint.txt"
    pointer.write_text(str(checkpoint) + "\r\n", encoding="utf-8-sig")
    pointer_bytes = pointer.read_bytes()
    assert pointer_bytes.startswith(b"\xef\xbb\xbf")
    protocol = {"protected_checkpoint": str(checkpoint), "protected_checkpoint_sha256": MODULE.sha256(checkpoint),
                "live_pointer_file": str(pointer), "live_pointer_file_sha256": MODULE.sha256(pointer)}
    verified = MODULE.verify_protected({"protocol": protocol})
    assert verified["pointer_sha256"] == protocol["live_pointer_file_sha256"]
    assert pointer.read_bytes() == pointer_bytes


def test_proofs_fail_closed_on_stale_source_binding(tmp_path):
    report = {"passes": True, "binding": {"source": "old"}}
    MODULE.atomic_json(tmp_path / "proof.json", report)
    config = {"proof": {"overfit_report": str(tmp_path / "proof.json")}}
    with pytest.raises(RuntimeError, match="stale"):
        MODULE.verify_proofs(config, {"source": "new"})


def test_source_manifest_covers_rgb_model_data_and_evaluation():
    manifest = MODULE.source_manifest()
    assert len(manifest) >= 7
    assert all(len(value) == 64 for value in manifest.values())


def test_tiny_epoch_status_score_checkpoint_and_resume(monkeypatch, tmp_path):
    from transformers import SegformerConfig
    from msr.models.rgb_segmenter import RgbSegformer
    from types import SimpleNamespace

    class TinyData(torch.utils.data.Dataset):
        records = [SimpleNamespace(sample_id="DC_tiny"), SimpleNamespace(sample_id="PHL_tiny")]
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return {"image": torch.ones(3, 64, 64) * .5,
                    "labels": torch.full((64, 64), index, dtype=torch.long),
                    "image_valid_mask": torch.ones(64, 64, dtype=torch.bool),
                    "classification_valid_mask": torch.ones(64, 64, dtype=torch.bool),
                    "sample_id": self.records[index].sample_id}

    config = MODULE.load_config(ROOT / "configs/gamus_rgb_segmenter_v1.yaml")
    config["experiment"]["output_root"] = str(tmp_path / "experiments")
    config["data"]["num_workers"] = 0
    config["training"]["epochs"] = 2
    config["training"]["batch_size"] = 2
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}")
    config["protocol"]["fixed_v3_independent_replay"] = str(baseline)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(MODULE.yaml.safe_dump(config))
    binding = {"config_sha256": MODULE.sha256(config_path), "source_code_sha256": {}}
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "verify_proofs", lambda *args: None)
    monkeypatch.setattr(MODULE, "verify_protected", lambda *args: {"unchanged": True})
    monkeypatch.setattr(MODULE, "verify_pretrained", lambda *args: None)
    monkeypatch.setattr(MODULE, "assert_sources", lambda *args: None)
    monkeypatch.setattr(MODULE, "verify_data_unchanged", lambda *args: None)
    def snapshot(experiment, config_path, binding):
        MODULE.shutil.copy2(config_path, experiment / "config.yaml")
        MODULE.atomic_json(experiment / "binding.json", binding)
    monkeypatch.setattr(MODULE, "snapshot_sources", snapshot)
    def new_model(config, device):
        return RgbSegformer(SegformerConfig(hidden_sizes=[8, 16, 24, 32], depths=[1] * 4,
                            num_attention_heads=[1, 2, 3, 4], sr_ratios=[8, 4, 2, 1],
                            decoder_hidden_size=16)).to(device)
    monkeypatch.setattr(MODULE, "make_model", new_model)
    validations = []
    def evaluate(model, loader, device, **kwargs):
        kwargs["progress_callback"]({"completed_batches": 1, "total_batches": 2})
        validations.append(1)
        if len(validations) == 2:
            raise RuntimeError("simulated interruption")
        return {"overall": {"six_class_identification": {"macro_f1": .7}}}
    monkeypatch.setattr(MODULE, "dataset_tools", lambda: (
        None, lambda dataset, config, generator: torch.utils.data.SequentialSampler(dataset),
        evaluate, lambda *args: {"passes": False}, None))
    with pytest.raises(RuntimeError, match="simulated interruption"):
        MODULE.train(config, config_path, TinyData(), TinyData(), binding, torch.device("cpu"), None)
    pointer = MODULE.json.loads((tmp_path / "outputs/orchestration/rgb_segmenter_v1_active.json").read_text())
    experiment = Path(pointer["experiment_dir"])
    assert pointer["experiment_path"] == pointer["experiment_dir"]
    latest = experiment / "checkpoint_latest.pt"
    payload = torch.load(latest, weights_only=False)
    assert payload["epoch"] == 1
    assert payload["history"][0]["validation"]["overall"]["six_class_identification"]["macro_f1"] == .7
    assert (experiment / "checkpoint_best_development.pt").exists()
    assert not (experiment / "checkpoint_best_guarded.pt").exists()
    assert MODULE.json.loads((experiment / "status.json").read_text())["stage"] == "failed"
    # Resumes only the completed epoch. Candidate architecture/state must load
    # without downloading an encoder or recreating the existing experiment.
    resumed = MODULE.train(config, config_path, TinyData(), TinyData(), binding, torch.device("cpu"), latest)
    assert resumed == experiment
    summary = MODULE.json.loads((experiment / "training_summary.json").read_text())
    assert summary["completed_epochs"] == 2 and summary["stage"] == "completed"
    assert summary["promotion_performed"] is False
    assert len((experiment / "metrics.jsonl").read_text().splitlines()) == 2
