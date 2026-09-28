"""Small CPU fixtures for the independent replay; no real checkpoint/data/GPU reads."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("independent_rgb_replay_test_module", ROOT / "scripts/evaluate_rgb_segmenter_checkpoint.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TinyRgbModel(torch.nn.Module):
    def forward(self, image):
        prediction = (image[:, 0] * 5).round().long()
        return {"logits": torch.nn.functional.one_hot(prediction, 6).permute(0, 3, 1, 2).float() * 10}


def batch(sample_id="DC_a"):
    labels = torch.tensor([[[0, 1, 2], [3, 4, 5], [4, 1, 2]]])
    return {"image": (labels[:, None].float() / 5).repeat(1, 3, 1, 1), "labels": labels,
        "image_valid_mask": torch.ones_like(labels, dtype=torch.bool),
        "classification_valid_mask": torch.ones_like(labels, dtype=torch.bool),
        "dark_pixel_proxy_mask": labels == 0, "sample_id": [sample_id], "city": [sample_id.split("_")[0]]}


def evaluate(batches=None, **kwargs):
    batches = batches or [batch("DC_a"), batch("PHL_a")]
    return MODULE.independent_evaluate(TinyRgbModel(), batches, "cpu", expected_ids=[b["sample_id"][0] for b in batches],
                                       native_size=3, **kwargs)


def test_independent_counts_scores_and_low_vegetation_error_directions():
    truth = np.array([0, 0, 1, 4, 4, 5, 255, np.nan])
    pred = np.array([0, 1, 1, 4, 5, 4, 255, np.nan])
    valid = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    matrix = MODULE.independent_confusion(pred, truth, valid)
    result = MODULE.independent_scores(matrix)
    assert result["total_valid_pixels"] == 6
    assert result["correct_pixels"] == 3
    assert result["accuracy"] == .5
    assert result["per_class"]["ground"]["precision"] == 1
    assert result["per_class"]["ground"]["recall"] == .5
    assert result["per_class"]["ground"]["f1"] == pytest.approx(2 / 3)
    assert result["per_class"]["low_vegetation"]["f1"] == .5
    assert result["per_class"]["low_vegetation"]["iou"] == pytest.approx(1 / 3)
    assert result["per_class"]["water"]["f1"] == 0
    errors = MODULE.low_vegetation_confusions(matrix)
    assert errors["reference_low_vegetation_predicted_as"]["trees"] == 1
    assert errors["predicted_low_vegetation_actual_reference"]["trees"] == 1
    assert errors["false_positive_pixels"] == errors["false_negative_pixels"] == 1


def test_loop_does_not_use_saved_or_trainer_multiclass_implementation(monkeypatch):
    import msr.evaluation.classification_metrics as metrics
    import msr.evaluation.rgb_segmentation as trainer_metrics
    def forbidden(*args, **kwargs):
        raise AssertionError("Trainer multiclass computation/evaluator must not run")
    monkeypatch.setattr(metrics, "compute_multiclass_metrics", forbidden)
    monkeypatch.setattr(metrics.StreamingMulticlassMetrics, "update", forbidden)
    monkeypatch.setattr(trainer_metrics, "evaluate_rgb_segmentation", forbidden)
    result = evaluate()
    assert result["evaluated_sample_count"] == 2
    assert result["overall"]["six_class_identification"]["total_valid_pixels"] == 18
    assert result["overall"]["six_class_identification"]["macro_f1"] == 1
    assert set(result["by_city"]) == {"DC", "PHL"}


def test_independent_validity_unknown_and_reference_hash(tmp_path):
    a, b = batch("DC_a"), batch("PHL_a")
    a["image_valid_mask"][0, 0, 0] = False
    a["dark_pixel_proxy_mask"][0, 0, 0] = False
    b["classification_valid_mask"][0, 1, 1] = False
    b["labels"][0, 1, 1] = 255
    result = evaluate([a, b], output=tmp_path)
    assert result["overall"]["six_class_identification"]["total_valid_pixels"] == 16
    records = [json.loads(line) for line in (tmp_path / "per_image_metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert all(row["masks"]["ignored_pixels"] == 1 for row in records)
    support = [{"sample_id": row["sample_id"], **{key: row["masks"][key] for key in ("valid_mask_sha256", "reference_labels_sha256")}} for row in records]
    assert result["reference_grid_binding_sha256"] == MODULE.canonical_sha256(support)
    assert result["overall"]["water_dark_pixel_proxy"]["is_shadow_proxy_not_ground_truth"]


def test_invalid_valid_classes_and_bad_grid_fail_closed():
    with pytest.raises(ValueError, match="indices"):
        MODULE.independent_confusion(np.array([1]), np.array([255]), np.array([True]))
    with pytest.raises(ValueError, match="indices"):
        MODULE.independent_confusion(np.array([.5]), np.array([1]), np.array([True]))
    a = batch()
    a["labels"][0, 0, 0] = 255
    with pytest.raises(ValueError, match="Classification-valid"):
        evaluate([a])
    a = batch()
    a["dark_pixel_proxy_mask"][0, 0, 0] = True
    a["image_valid_mask"][0, 0, 0] = False
    with pytest.raises(ValueError, match="Dark-pixel"):
        evaluate([a])


def test_exact_validation_order_batch_size_and_geography():
    with pytest.raises(ValueError, match="fixed order"):
        MODULE.independent_evaluate(TinyRgbModel(), [batch("DC_b")], "cpu", expected_ids=["DC_a"], native_size=3)
    with pytest.raises(ValueError, match="DC/PHL"):
        MODULE.independent_evaluate(TinyRgbModel(), [], "cpu", expected_ids=["NYC_a"], native_size=3)
    with pytest.raises(ValueError, match="full-native"):
        MODULE.independent_evaluate(TinyRgbModel(), [batch()], "cpu", expected_ids=["DC_a"])
    a = batch()
    a["sample_id"] = ["DC_a", "DC_b"]
    a["city"] = ["DC", "DC"]
    with pytest.raises(ValueError, match="batch size one"):
        MODULE.independent_evaluate(TinyRgbModel(), [a], "cpu", expected_ids=["DC_a", "DC_b"], native_size=3)


def test_preview_selection_has_no_scores_and_preserves_all_classes(tmp_path):
    ids = [f"{city}_{i}" for city in ("PHL", "DC") for i in range(5)]
    selected = MODULE.select_preview_ids(ids[::-1])
    assert selected == ["DC_0", "DC_2", "DC_4", "PHL_0", "PHL_2", "PHL_4"]
    assert selected == MODULE.select_preview_ids(ids)
    batches = [batch(s) for s in ids]
    for a in batches:
        a["classification_valid_mask"][0, 2, 0] = False
        a["labels"][0, 2, 0] = 255
    evaluate(batches, output=tmp_path, selected_ids=selected)
    manifest = json.loads((tmp_path / "preview_manifest.json").read_text())
    assert len(manifest["items"]) == 6
    assert manifest["selection"]["selected_ids"] == selected
    assert len({row["color"] for row in manifest["classes"]}) == 6
    item = manifest["items"][0]
    predicted = np.asarray(Image.open(tmp_path / item["prediction"]))
    reference = np.asarray(Image.open(tmp_path / item["reference"]))
    overlay = np.asarray(Image.open(tmp_path / item["prediction_overlay"]))
    ref_overlay = np.asarray(Image.open(tmp_path / item["reference_overlay"]))
    assert set(np.unique(predicted)) == set(range(6))
    assert set(np.unique(reference)) == {*range(6), 255}
    assert overlay[0, 0, 3] == 255  # Ground is visible; gallery controls opacity.
    assert ref_overlay[2, 0, 3] == 0  # Unknown reference is transparent.
    assert Image.open(tmp_path / item["rgb"]).size == (3, 3)
    metadata = json.loads((tmp_path / item["metadata"]).read_text())
    assert metadata["six_class_identification"]["total_valid_pixels"] == 8
    assert not manifest["promotion_performed"]


@pytest.mark.parametrize("change", ["count", "score", "support", "ids", "grid", "boundary", "dark"])
def test_saved_epoch_comparison_rejects_drift(change):
    result = evaluate()
    assert MODULE.compare_saved_epoch(result, result)["exact_match"]
    other = deepcopy(result)
    if change == "count":
        other["overall"]["six_class_identification"]["confusion_matrix"][0][1] += 1
        other["overall"]["six_class_identification"]["confusion_matrix"][0][0] -= 1
    elif change == "score":
        other["overall"]["six_class_identification"]["macro_f1"] -= .001
    elif change == "support":
        other["by_city"]["DC"]["six_class_identification"]["confusion_matrix"][0][0] += 1
    elif change == "ids":
        other["evaluated_ordered_ids_sha256"] = "changed"
    elif change == "grid":
        other["reference_grid_binding_sha256"] = "changed"
    elif change == "boundary":
        other["overall"]["road_boundary_quality"]["reference_boundary_pixels"] += 1
    else:
        other["by_city"]["DC"]["water_dark_pixel_proxy"]["dark_non_water_pixels"] += 1
    comparison = MODULE.compare_saved_epoch(other, result)
    assert not comparison["passes"]
    assert not comparison["exact_match"]
    assert comparison["failed_checks"]


def test_small_scalar_roundoff_explicitly_reported():
    result = evaluate()
    other = deepcopy(result)
    other["overall"]["six_class_identification"]["macro_f1"] -= 1e-13
    comparison = MODULE.compare_saved_epoch(other, result)
    assert comparison["passes"] and not comparison["exact_match"]
    assert comparison["counts_match"]
    assert comparison["differences"]["overall"]["score_deltas"]["macro_f1"] != 0


def test_checkpoint_path_and_hash_authenticate_before_pickle(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(MODULE.torch, "load", lambda *a, **k: calls.append((a, k)))
    with pytest.raises(ValueError, match="Only the exact"):
        MODULE.load_known_checkpoint(tmp_path)
    monkeypatch.setattr(MODULE, "KNOWN_EXPERIMENT", tmp_path)
    checkpoint = tmp_path / "checkpoint_best_guarded.pt"
    checkpoint.write_bytes(b"untrusted pickle bytes")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        MODULE.load_known_checkpoint(tmp_path)
    assert calls == []


def test_readonly_file_identity_and_changed_bytes_detection(tmp_path):
    protected = tmp_path / "protected.pt"
    pointer = tmp_path / "pointer.txt"
    protected.write_bytes(b"protected unchanged model")
    pointer.write_text(str(protected))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (protected, pointer)}
    identities = MODULE.capture_file_identities([protected, pointer])
    MODULE.verify_file_identities(identities)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (protected, pointer)}
    protected.write_bytes(b"modified model")
    with pytest.raises(RuntimeError, match="identity changed"):
        MODULE.verify_file_identities(identities)


def test_existing_output_fails_before_any_checkpoint_or_data_read(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    output = tmp_path / "outputs/evaluation/already_there"
    output.mkdir(parents=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("No checkpoint or data reads after output collision")
    monkeypatch.setattr(MODULE, "load_known_checkpoint", forbidden)
    with pytest.raises(FileExistsError):
        MODULE.main(["--experiment", str(tmp_path), "--output", str(output)])


def test_model_training_state_is_restored():
    model = TinyRgbModel().train()
    MODULE.independent_evaluate(model, [batch()], "cpu", expected_ids=["DC_a"], native_size=3)
    assert model.training
