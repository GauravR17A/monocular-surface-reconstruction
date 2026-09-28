"""Independent, read-only replay of the completed RGB SegFormer guarded checkpoint.

Only the named, SHA-256-pinned local checkpoint is eligible for pickle loading.
The production CLI evaluates all 859 native DC/PHL validation tiles with batch
size one and bf16 CUDA inference. It never trains, accesses test/NYC, or changes
an app pointer. All new artifacts belong to a fresh outputs/evaluation child.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msr.data.gamus_rgb_segmentation import (
    ALLOWED_CITIES, GamusRgbSegmentationDataset, authenticate_validation_content,
    canonical_sha256, source_metadata_snapshot,
)
from msr.evaluation.classification_metrics import (
    StreamingClassBoundaryMetrics, StreamingWaterDarkPixelProxy,
)
from msr.evaluation.rgb_segmentation import classification_acceptance
from msr.models.rgb_segmenter import CLASS_NAMES, RgbSegformer

KNOWN_EXPERIMENT = ROOT / "experiments/20260915T140116654177Z_gamus_rgb_segmenter_v1"
KNOWN_CHECKPOINT_SHA256 = "4940b2d4ae8817370384ccb54da9f73aee1b6bf3614435d8c077bf3e3363d021"
KNOWN_EPOCH = 8
KNOWN_MACRO_F1 = 0.8382524600400737
COLORS = ("#D4A259", "#F45B69", "#24A9F2", "#A78BFA", "#C9E84B", "#168A61")
SCORE_TOLERANCE = 1e-12
COUNT_FRACTION_TOLERANCE = 0.0


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_hash(path, expected):
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected)) or sha256(path) != expected:
        raise RuntimeError(f"Sealed file identity mismatch: {path}")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def capture_file_identities(paths):
    """Read file bytes and metadata; callers can prove all sealed inputs stayed put."""
    result = {}
    for path in sorted({Path(p).resolve() for p in paths}):
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"File changed while recording identity: {path}")
        result[str(path)] = {"sha256": digest, "size_bytes": after.st_size,
                             "modified_time_ns": after.st_mtime_ns}
    return result


def verify_file_identities(expected):
    if capture_file_identities(expected) != expected:
        raise RuntimeError("A sealed input or protected app identity changed during replay")


def load_known_checkpoint(experiment, checkpoint=None):
    experiment = resolve(experiment)
    checkpoint = resolve(checkpoint) if checkpoint is not None else experiment / "checkpoint_best_guarded.pt"
    if experiment != KNOWN_EXPERIMENT.resolve() or checkpoint != experiment / "checkpoint_best_guarded.pt":
        raise ValueError("Only the exact completed run's checkpoint_best_guarded.pt is trusted")
    assert_hash(checkpoint, KNOWN_CHECKPOINT_SHA256)
    # Hash and exact-path authentication MUST precede unsafe pickle loading.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_type") != RgbSegformer.model_type or payload.get("epoch") != KNOWN_EPOCH:
        raise RuntimeError("Wrong model type or completed epoch")
    if payload.get("promotion_eligible") is not False:
        raise RuntimeError("Expected a development-only standalone checkpoint")
    if payload.get("input_contract") != "raw_rgb_01_imagenet_normalized_in_model":
        raise RuntimeError("Unexpected model input contract")
    history = payload.get("history", [])
    if [row.get("epoch") for row in history] != list(range(1, KNOWN_EPOCH + 1)):
        raise RuntimeError("Incomplete or reordered checkpoint epoch history")
    expected_best = {"epoch": KNOWN_EPOCH, "macro_f1": KNOWN_MACRO_F1}
    if payload.get("best_guarded") != expected_best:
        raise RuntimeError("Checkpoint is not the known guarded epoch-eight selection")
    if history[-1]["validation"]["overall"]["six_class_identification"]["macro_f1"] != KNOWN_MACRO_F1:
        raise RuntimeError("Checkpoint's saved epoch score differs from known completed run")
    if history[-1]["gate"].get("passes") is not True:
        raise RuntimeError("Saved guarded checkpoint did not pass its declared gate")
    return payload


def validate_sealed_run(experiment, payload):
    """Authenticate run metadata and live source bytes without invoking trainer preflight."""
    binding = json.loads((experiment / "binding.json").read_text(encoding="utf-8"))
    if payload.get("binding") != binding:
        raise RuntimeError("Checkpoint and run binding disagree")
    paths = [experiment / name for name in ("binding.json", "config.yaml", "metrics.jsonl",
             "training_summary.json", "status.json", "checkpoint_best_guarded.pt",
             "overfit_report.json", "validation_smoke_report.json")]
    assert_hash(experiment / "config.yaml", binding["config_sha256"])
    config = yaml.safe_load((experiment / "config.yaml").read_text(encoding="utf-8"))
    current_config = ROOT / "configs/gamus_rgb_segmenter_v1.yaml"
    assert_hash(current_config, binding["config_sha256"])
    paths.append(current_config)
    source_names = binding["source_code_sha256"]
    if set(source_names) != {"scripts/train_rgb_segmenter.py",
            "src/msr/models/rgb_segmenter.py", "src/msr/data/gamus_rgb_segmentation.py",
            "src/msr/evaluation/rgb_segmentation.py", "src/msr/evaluation/classification_metrics.py",
            "src/msr/data/gamus_dataset.py", "src/msr/data/raster_dataset.py"}:
        raise RuntimeError("Unexpected sealed source set")
    for name, expected in source_names.items():
        for path in (ROOT / name, experiment / "source_snapshot" / name):
            assert_hash(path, expected)
            paths.append(path)
    current_software = {"python": sys.version.split()[0], **{
        package: importlib.metadata.version(package) for package in binding["software"] if package != "python"}}
    if current_software != binding["software"]:
        raise RuntimeError(f"Replay software differs from the sealed run: {current_software}")
    history = [json.loads(line) for line in (experiment / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if history != payload["history"]:
        raise RuntimeError("Saved metrics log differs from checkpoint's authenticated history")
    summary = json.loads((experiment / "training_summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != "completed" or summary.get("completed_epochs") != KNOWN_EPOCH or summary.get("best_guarded") != payload["best_guarded"]:
        raise RuntimeError("Experiment is not the expected completed guarded run")
    for name in ("overfit_report", "validation_smoke_report"):
        proof = json.loads((experiment / f"{name}.json").read_text(encoding="utf-8"))
        if proof.get("passes") is not True or proof.get("binding") != binding:
            raise RuntimeError(f"Sealed {name} proof is stale or failed")
    protocol, data = config["protocol"], config["data"]
    if protocol.get("auto_promotion") is not False or protocol.get("development_only") is not True:
        raise RuntimeError("Replay must remain development-only")
    if protocol.get("development_validation_cities") != ["DC", "PHL"] or protocol.get("learning_cities") != ["DC", "PHL"]:
        raise RuntimeError("Only sealed DC/PHL development geography is permitted")
    if data["validation_patch_size"] != 1024 or config["training"]["precision"] != "bf16" or data["validation_radiometric_policy"] != "raw":
        raise RuntimeError("Replay requires the original 1024-pixel raw RGB bf16 protocol")
    for key in ("protected_checkpoint", "live_pointer_file", "fixed_v3_independent_replay"):
        path = resolve(protocol[key])
        assert_hash(path, protocol[f"{key}_sha256"])
        paths.append(path)
    pointer = resolve(protocol["live_pointer_file"]).read_text(encoding="utf-8-sig").strip()
    if resolve(pointer) != resolve(protocol["protected_checkpoint"]):
        raise RuntimeError("App pointer target differs from the protected checkpoint")
    protected = {"checkpoint_sha256": protocol["protected_checkpoint_sha256"],
                 "pointer_sha256": protocol["live_pointer_file_sha256"]}
    if protected != binding["protected"] or summary.get("protected") != protected:
        raise RuntimeError("Protected checkpoint binding differs")
    index = resolve(data["approved_index_path"])
    assert_hash(index, data["approved_index_file_sha256"])
    paths.append(index)
    contract = dict(binding["data_contract"])
    contract_sha = contract.pop("contract_sha256")
    if canonical_sha256(contract) != contract_sha or contract["validation_count"] != 859 or contract["class_names"] != list(CLASS_NAMES):
        raise RuntimeError("Sealed data contract identity differs")
    return config, binding, capture_file_identities(paths)


def validation_dataset(config, binding):
    data = config["data"]
    dataset = GamusRgbSegmentationDataset(resolve(data["root"]), "val",
        approved_index_path=resolve(data["approved_index_path"]),
        approved_index_sha256=data["approved_index_file_sha256"], patch_size=1024,
        random_crop=False, augment=False, rgb_scale=float(data["rgb_scale"]),
        dark_pixel_threshold=float(data["dark_pixel_threshold"]), native_shape=1024)
    contract = binding["data_contract"]
    if len(dataset) != 859 or canonical_sha256(list(dataset.sample_ids)) != contract["validation_ids_sha256"]:
        raise RuntimeError("Validation IDs/order differ from sealed checkpoint")
    if source_metadata_snapshot(dataset) != contract["source_metadata"]["val"]:
        raise RuntimeError("Validation source metadata differs from sealed checkpoint")
    return dataset


def independent_confusion(prediction, reference, valid):
    prediction, reference, valid = np.asarray(prediction), np.asarray(reference), np.asarray(valid, dtype=bool)
    if prediction.shape != reference.shape or reference.shape != valid.shape:
        raise ValueError("Prediction, reference and validity shapes must match")
    for name, values in (("prediction", prediction[valid]), ("reference", reference[valid])):
        if np.any(~np.isfinite(values) | (values < 0) | (values >= 6) | (values != np.rint(values))):
            raise ValueError(f"Valid {name} must contain integer class indices 0..5")
    encoded = reference[valid].astype(np.int64) * 6 + prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=36).reshape(6, 6)


def independent_scores(matrix):
    """Explicit scalar score formulas; no trainer multiclass accumulator/helper."""
    matrix = np.asarray(matrix)
    if matrix.shape != (6, 6) or matrix.dtype.kind not in "iu" or np.any(matrix < 0):
        raise ValueError("Expected a nonnegative integer 6x6 confusion matrix")
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        tp, support, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        fp, fn = predicted - tp, support - tp
        per_class[name] = {"precision": tp / predicted if predicted else 0.,
            "recall": tp / support if support else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.,
            "support_pixels": support, "predicted_pixels": predicted, "true_positive_pixels": tp,
            "false_positive_pixels": fp, "false_negative_pixels": fn}
    macro = {key: float(np.mean([row[key] for row in per_class.values()]))
             for key in ("precision", "recall", "f1", "iou")}
    total, correct = int(matrix.sum()), int(np.trace(matrix))
    accuracy = correct / total if total else 0.
    return {"class_names": list(CLASS_NAMES), "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": {"rows": "reference", "columns": "predicted"},
        "total_valid_pixels": total, "correct_pixels": correct, "accuracy": accuracy,
        "error_rate": 1. - accuracy if total else 0., "macro": macro,
        **{f"macro_{key}": value for key, value in macro.items()}, "per_class": per_class}


class MetricGroup:
    def __init__(self):
        self.matrix = np.zeros((6, 6), dtype=np.int64)
        self.road = StreamingClassBoundaryMetrics(CLASS_NAMES.index("roads"), tolerance_pixels=2)
        self.water = StreamingWaterDarkPixelProxy(CLASS_NAMES.index("water"))

    def update(self, matrix, prediction, reference, dark, valid):
        self.matrix += matrix
        self.road.update(prediction, reference, valid)
        self.water.update(prediction, reference, dark, valid)

    def compute(self):
        return {"six_class_identification": independent_scores(self.matrix),
            "road_boundary_quality": self.road.compute(),
            "water_dark_pixel_proxy": {**self.water.compute(), "water_class_index": CLASS_NAMES.index("water")}}


def select_preview_ids(sample_ids):
    """Pre-score selection: first, middle, last lexicographic ID in each city."""
    if len(sample_ids) != len(set(sample_ids)) or any(not re.fullmatch(r"(?:DC|PHL)_[A-Za-z0-9_]+", s) for s in sample_ids):
        raise ValueError("Preview IDs must be unique safe DC/PHL development IDs")
    result = []
    for city in ALLOWED_CITIES:
        ids = sorted(s for s in sample_ids if s.startswith(city + "_"))
        if len(ids) < 3:
            raise ValueError("Three distinct validation IDs per city are required for previews")
        result.extend(ids[i] for i in (0, (len(ids) - 1) // 2, len(ids) - 1))
    return result


def mask_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.uint8).tobytes()).hexdigest()


def low_vegetation_confusions(matrix):
    i = CLASS_NAMES.index("low_vegetation")
    return {"reference_low_vegetation_predicted_as": {name: int(matrix[i, j]) for j, name in enumerate(CLASS_NAMES)},
        "predicted_low_vegetation_actual_reference": {name: int(matrix[j, i]) for j, name in enumerate(CLASS_NAMES)},
        "false_positive_pixels": int(matrix[:, i].sum() - matrix[i, i]),
        "false_negative_pixels": int(matrix[i].sum() - matrix[i, i])}


def export_preview(output, record, image, prediction, reference, image_valid, valid):
    from PIL import Image
    sample_id = record["sample_id"]
    directory = output / "previews" / sample_id
    directory.mkdir(parents=True, exist_ok=False)
    palette = [0] * (256 * 3)
    for index, color in enumerate(COLORS):
        palette[index * 3:index * 3 + 3] = bytes.fromhex(color[1:])
    palette[255 * 3:255 * 3 + 3] = [55, 65, 81]
    rgb = np.rint(np.moveaxis(image, 0, -1) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(directory / "rgb.png")
    for name, values, mask in (("prediction", prediction, image_valid), ("reference", reference, valid)):
        indexed = np.where(mask, values, 255).astype(np.uint8)
        categorical = Image.fromarray(indexed).convert("P")
        categorical.putpalette(palette)
        categorical.save(directory / f"{name}.png")
        rgba = np.zeros((*indexed.shape, 4), dtype=np.uint8)
        colors = np.asarray(palette, dtype=np.uint8).reshape(256, 3)
        rgba[:, :, :3] = colors[indexed]
        rgba[:, :, 3] = np.where(mask, 255, 0)
        Image.fromarray(rgba).save(directory / f"{name}_overlay.png")
        np.save(directory / f"{name}.npy", indexed, allow_pickle=False)
    metadata = {**record, "native_size": list(rgb.shape[:2]),
        "rgb_export": "Original raw RGB input multiplied by 255 and rounded to PNG bytes; no resizing or synthesis",
        "prediction_ignore": "255 only where RGB invalid", "reference_ignore": "255 where RGB or classification invalid",
        "selection": "Pre-score first/middle/last lexicographic validation ID per city"}
    atomic_json(directory / "metadata.json", metadata)
    result = {"sample_id": sample_id, "city": record["city"]}
    for key, name in (("rgb", "rgb.png"), ("prediction", "prediction.png"), ("reference", "reference.png"),
                      ("prediction_overlay", "prediction_overlay.png"), ("reference_overlay", "reference_overlay.png"),
                      ("metadata", "metadata.json")):
        result[key] = (directory / name).relative_to(output).as_posix()
    result["sha256"] = {key: sha256(output / path) for key, path in result.items() if key not in {"sample_id", "city"}}
    return result


@torch.inference_mode()
def independent_evaluate(model, loader, device, *, expected_ids, precision="bf16", native_size=1024,
                         output=None, selected_ids=(), progress_callback=None):
    device = torch.device(device)
    expected_ids = tuple(expected_ids)
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Unique expected validation IDs are required")
    if any(not re.fullmatch(r"(?:DC|PHL)_[A-Za-z0-9_]+", s) for s in expected_ids):
        raise ValueError("Only DC/PHL development validation is permitted")
    if precision not in {"bf16", "fp32"}:
        raise ValueError("Only bf16 or fp32 inference is supported")
    if set(selected_ids) - set(expected_ids) or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Preview selection must be unique and belong to sealed validation IDs")
    groups, overall = {city: MetricGroup() for city in ALLOWED_CITIES}, MetricGroup()
    observed, support_rows, preview_items = [], [], []
    was_training = model.training
    model.eval()
    stream = (Path(output) / "per_image_metrics.jsonl").open("x", encoding="utf-8", newline="\n") if output is not None else None
    try:
        for batch in loader:
            sample_ids, cities = list(batch["sample_id"]), list(batch["city"])
            if len(sample_ids) != 1 or len(cities) != 1 or len(observed) >= len(expected_ids) or sample_ids[0] != expected_ids[len(observed)]:
                raise ValueError("Validation must use batch size one and cover every sealed ID exactly once in fixed order")
            sample_id, city = sample_ids[0], cities[0]
            if city != sample_id.split("_", 1)[0]:
                raise ValueError("Sample and city identities disagree")
            image_cpu = batch["image"]
            if image_cpu.shape != (1, 3, native_size, native_size):
                raise ValueError("Inference requires unresized full-native RGB tiles")
            if not torch.isfinite(image_cpu).all() or torch.any((image_cpu < 0) | (image_cpu > 1)):
                raise ValueError("Model input must be finite raw RGB in [0,1]")
            image = image_cpu.to(device, non_blocking=True)
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and precision == "bf16" else nullcontext()
            with amp:
                result = model(image)
            logits = result["logits"] if isinstance(result, dict) else result
            if logits.shape != (1, 6, native_size, native_size) or not torch.isfinite(logits).all():
                raise ValueError("Model must return finite six-class full-native logits")
            prediction = logits.argmax(1)[0].cpu().numpy()
            reference = batch["labels"][0].cpu().numpy()
            image_valid = batch["image_valid_mask"][0].cpu().numpy().astype(bool)
            class_valid = batch["classification_valid_mask"][0].cpu().numpy().astype(bool)
            dark = batch["dark_pixel_proxy_mask"][0].cpu().numpy().astype(bool)
            if not all(values.shape == prediction.shape for values in (reference, image_valid, class_valid, dark)):
                raise ValueError("All independent masks must match full-native prediction")
            if np.any(class_valid & (~np.isfinite(reference) | (reference < 0) | (reference >= 6) | (reference != np.rint(reference)))):
                raise ValueError("Classification-valid references must be six-class indices")
            if np.any(dark & ~image_valid):
                raise ValueError("Dark-pixel proxy includes invalid RGB")
            valid = image_valid & class_valid
            matrix = independent_confusion(prediction, reference, valid)
            overall.update(matrix, prediction, reference, dark, valid)
            groups[city].update(matrix, prediction, reference, dark, valid)
            support = {"sample_id": sample_id, "valid_mask_sha256": mask_hash(valid),
                       "reference_labels_sha256": mask_hash(np.where(valid, reference, 255))}
            support_rows.append(support)
            record = {"sample_id": sample_id, "city": city, "six_class_identification": independent_scores(matrix),
                "low_vegetation_confusions": low_vegetation_confusions(matrix),
                "masks": {"image_valid_pixels": int(image_valid.sum()), "classification_valid_pixels": int(class_valid.sum()),
                          "joint_valid_pixels": int(valid.sum()), "ignored_pixels": int(valid.size - valid.sum()),
                          "dark_non_water_pixels": int((dark & valid & (reference != CLASS_NAMES.index("water"))).sum()),
                          **{key: value for key, value in support.items() if key != "sample_id"}},
                "prediction_on_valid_sha256": mask_hash(np.where(valid, prediction, 255))}
            if stream is not None:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
            if sample_id in selected_ids and output is not None:
                preview_items.append(export_preview(Path(output), record, image_cpu[0].numpy(), prediction, reference, image_valid, valid))
            observed.append(sample_id)
            if progress_callback is not None:
                progress_callback({"completed_batches": len(observed), "total_batches": len(expected_ids), "sample_id": sample_id})
            del logits, result, image
    finally:
        if stream is not None:
            stream.close()
        model.train(was_training)
    if tuple(observed) != expected_ids or int(overall.matrix.sum()) <= 0:
        raise ValueError("Validation must cover every sealed ID exactly once with nonempty reference support")
    evaluation = {"schema": "msr.gamus_rgb_segmentation_independent_evaluation.v1",
        "interpretation": "DC+PHL development validation, not system-unseen or official test",
        "height_evaluated": False, "model_inputs": ["RGB"], "validation_native_size": native_size,
        "evaluated_sample_count": len(observed), "evaluated_samples_by_city": dict(sorted(Counter(s.split("_", 1)[0] for s in observed).items())),
        "evaluated_ids_sha256": canonical_sha256(sorted(observed)), "evaluated_ordered_ids_sha256": canonical_sha256(observed),
        "evaluated_ids_by_city_sha256": {city: canonical_sha256(sorted(s for s in observed if s.startswith(city + "_"))) for city in ALLOWED_CITIES},
        "reference_grid_binding_sha256": canonical_sha256(support_rows),
        "overall": overall.compute(), "by_city": {city: group.compute() for city, group in groups.items()}}
    if output is not None:
        by_id = {row["sample_id"]: row for row in preview_items}
        if set(by_id) != set(selected_ids):
            raise RuntimeError("Not every preselected preview was exported")
        atomic_json(Path(output) / "preview_manifest.json", {
            "schema": "msr.rgb_segmenter_independent_previews.v1", "native_size": native_size,
            "selection": {"policy": "first_middle_last_lexicographic_id_per_city_before_scoring", "selected_ids": list(selected_ids), "per_city_count": 3},
            "classes": [{"index": i, "name": name, "color": COLORS[i]} for i, name in enumerate(CLASS_NAMES)],
            "ignore": {"index": 255, "color": "#374151", "label": "Unknown / invalid reference; excluded from metrics"},
            "overlay_alpha": 255, "items": [by_id[s] for s in selected_ids], "promotion_performed": False})
    return evaluation


def compare_saved_epoch(actual, saved):
    failures, differences, exact, counts_match = [], {}, True, True
    identity_keys = ("evaluated_sample_count", "evaluated_samples_by_city", "evaluated_ids_sha256",
        "evaluated_ordered_ids_sha256", "evaluated_ids_by_city_sha256", "reference_grid_binding_sha256", "validation_native_size")
    for key in identity_keys:
        if actual.get(key) != saved.get(key) or key not in actual:
            failures.append(f"Exact validation identity mismatch: {key}")
    for name in ("overall", *ALLOWED_CITIES):
        current = actual["overall"] if name == "overall" else actual["by_city"][name]
        reference = saved["overall"] if name == "overall" else saved["by_city"][name]
        a, b = current["six_class_identification"], reference["six_class_identification"]
        matrix_a, matrix_b = np.asarray(a["confusion_matrix"], dtype=np.int64), np.asarray(b["confusion_matrix"], dtype=np.int64)
        count_l1 = int(np.abs(matrix_a - matrix_b).sum())
        counts_match = counts_match and count_l1 == 0
        total = int(matrix_b.sum())
        budget = int(total * COUNT_FRACTION_TOLERANCE)
        score_deltas = {"macro_f1": a["macro_f1"] - b["macro_f1"], "macro_iou": a["macro_iou"] - b["macro_iou"]}
        for class_name in CLASS_NAMES:
            for key in ("precision", "recall", "f1", "iou"):
                score_deltas[f"{class_name}.{key}"] = a["per_class"][class_name][key] - b["per_class"][class_name][key]
        if not np.array_equal(matrix_a.sum(1), matrix_b.sum(1)):
            failures.append(f"{name}: exact reference class supports changed")
        if count_l1 > budget:
            failures.append(f"{name}: confusion count L1 drift {count_l1} exceeds {budget}")
        if any(not np.isfinite(delta) or abs(delta) > SCORE_TOLERANCE for delta in score_deltas.values()):
            failures.append(f"{name}: independent class scores exceed {SCORE_TOLERANCE} absolute drift")
        auxiliary = {}
        for metric, support_key in (("road_boundary_quality", "reference_boundary_pixels"),
                                    ("water_dark_pixel_proxy", "dark_non_water_pixels")):
            if current[metric][support_key] != reference[metric][support_key]:
                failures.append(f"{name}.{metric}: exact {support_key} changed")
            for key, value in reference[metric].items():
                observed = current[metric].get(key)
                if observed == value:
                    continue
                exact = False
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    failures.append(f"{name}.{metric}.{key}: protocol metadata differs")
                elif isinstance(value, int):
                    delta = int(observed) - value
                    counts_match = counts_match and delta == 0
                    auxiliary[f"{metric}.{key}"] = delta
                    if abs(delta) > int(max(1, abs(value)) * COUNT_FRACTION_TOLERANCE):
                        failures.append(f"{name}.{metric}.{key}: count drift exceeds tolerance")
                else:
                    delta = float(observed) - value
                    auxiliary[f"{metric}.{key}"] = delta
                    if not np.isfinite(delta) or abs(delta) > SCORE_TOLERANCE:
                        failures.append(f"{name}.{metric}.{key}: score drift exceeds tolerance")
        exact = exact and count_l1 == 0 and all(delta == 0 for delta in score_deltas.values())
        differences[name] = {"confusion_matrix_l1_difference": count_l1, "confusion_matrix_l1_budget": budget,
                             "score_deltas": score_deltas, "auxiliary_deltas": auxiliary}
    return {"passes": not failures, "exact_match": exact and not failures, "counts_match": counts_match, "failed_checks": failures,
        "tolerance": {"score_absolute": SCORE_TOLERANCE, "confusion_l1_fraction_of_valid_pixels": COUNT_FRACTION_TOLERANCE,
            "auxiliary_count_fraction": COUNT_FRACTION_TOLERANCE,
            "explanation": "All integer confusion/auxiliary counts, IDs, reference-grid hashes and reference supports must match exactly. Only scalar scores permit 1e-12 absolute numeric rounding drift; any changed prediction counts fail verification."},
        "differences": differences}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    experiment, output = resolve(args.experiment), resolve(args.output)
    allowed = (ROOT / "outputs/evaluation").resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("Output must be a fresh child below outputs/evaluation")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def status(stage, **fields):
        atomic_json(output / "status.json", {"stage": stage, "updated_at_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started, "promotion_performed": False, **fields})
    try:
        status("authenticating_checkpoint")
        payload = load_known_checkpoint(experiment, args.checkpoint)
        config, binding, file_identities = validate_sealed_run(experiment, payload)
        dataset = validation_dataset(config, binding)
        selected = select_preview_ids(dataset.sample_ids)
        atomic_json(output / "preview_selection.json", {"selected_at_utc": utc_now(),
            "policy": "first_middle_last_lexicographic_id_per_city_before_scoring", "selected_ids": selected})
        atomic_json(output / "input_file_identities.json", file_identities)
        shutil.copy2(Path(__file__), output / "evaluator_source.py")
        evaluator_sha = sha256(Path(__file__))
        status("authenticating_validation_content", validation_count=len(dataset), source_file_count=1718)
        protocol = config["protocol"]
        content_identity = authenticate_validation_content(dataset, resolve(protocol["fixed_v3_independent_replay"]),
            protocol["fixed_v3_independent_replay_sha256"], cache_path=None)
        atomic_json(output / "validation_content_identity.json", content_identity)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This production replay requires CUDA with native bf16 support")
        device = torch.device(args.device)
        status("loading_fresh_model", checkpoint_epoch=KNOWN_EPOCH)
        torch.manual_seed(config["experiment"]["seed"])
        model = RgbSegformer.from_architecture(payload["architecture"])
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.requires_grad_(False).eval().to(device)
        saved = payload["history"][-1]["validation"]
        architecture = payload["architecture"]
        del payload
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config["data"]["num_workers"], pin_memory=True)
        def progress(update):
            done = update["completed_batches"]
            if done == 1 or done % 10 == 0 or done == len(dataset):
                status("evaluating", **update)
            if done == 1 or done % 100 == 0 or done == len(dataset):
                print(f"Independent RGB replay {done}/{len(dataset)}", flush=True)
        status("evaluating", completed_batches=0, total_batches=len(dataset))
        evaluation = independent_evaluate(model, loader, device, expected_ids=dataset.sample_ids,
            precision="bf16", native_size=1024, output=output, selected_ids=selected, progress_callback=progress)
        del model, loader
        torch.cuda.empty_cache()
        status("verifying_results")
        comparison = compare_saved_epoch(evaluation, saved)
        baseline = json.loads(resolve(protocol["fixed_v3_independent_replay"]).read_text(encoding="utf-8"))
        acceptance = classification_acceptance(evaluation, baseline, protocol["classification_acceptance"],
                                             protocol["classification_acceptance_by_city"])
        if source_metadata_snapshot(dataset) != binding["data_contract"]["source_metadata"]["val"]:
            raise RuntimeError("Validation source metadata changed during replay")
        verify_file_identities(file_identities)
        assert_hash(Path(__file__), evaluator_sha)
        report = {"schema": "msr.rgb_segmenter_checkpoint_independent_replay.v1",
            "passes": comparison["passes"] and acceptance["passes"], "replay_verified": comparison["passes"], "created_at_utc": utc_now(),
            "experiment": str(experiment), "checkpoint": {"path": str(experiment / "checkpoint_best_guarded.pt"),
                "sha256": KNOWN_CHECKPOINT_SHA256, "epoch": KNOWN_EPOCH, "saved_macro_f1": KNOWN_MACRO_F1,
                "fresh_cpu_load": True, "weights_only": False, "strict_state_dict_load": True},
            "binding": binding, "run_config": config, "architecture": architecture,
            "provenance": {"evaluator_source_sha256": evaluator_sha, "evaluator_snapshot": "evaluator_source.py",
                "multiclass_metrics": "Independent NumPy bincount confusion and explicit scalar precision/recall/F1/IoU formulas",
                "auxiliary_metrics": "Reused sealed road boundary and water dark-pixel proxy accumulators",
                "device": str(device), "device_name": torch.cuda.get_device_name(device), "precision": "bf16",
                "batch_size": 1, "native_size": 1024, "validation_content": content_identity,
                "protected_and_sealed_file_identities_unchanged": True, "input_identities_file": "input_file_identities.json",
                "training_rasters_read": False, "height_or_prior_files_read": False,
                "official_test_or_nyc_read": False, "external_geography_read": False, "network_model_download": False},
            "evaluation": evaluation, "saved_epoch_comparison": comparison, "acceptance": acceptance,
            "artifacts": {"per_image_metrics": "per_image_metrics.jsonl", "preview_manifest": "preview_manifest.json"},
            "elapsed_seconds": time.monotonic() - started, "development_only": True,
            "promotion_eligible": False, "promotion_performed": False}
        atomic_json(output / "report.json", report)
        status("completed" if report["passes"] else "failed", passes=report["passes"],
               macro_f1=evaluation["overall"]["six_class_identification"]["macro_f1"],
               exact_saved_epoch_match=comparison["exact_match"], report="report.json")
        print(json.dumps({"output": str(output), "passes": report["passes"], "exact_match": comparison["exact_match"]}), flush=True)
        return 0 if report["passes"] else 1
    except BaseException as error:
        status("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
