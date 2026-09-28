"""Compare cached native validation with the real app's tiled inference.

All 160 HighBuild and 120 OpenCanopy scenes, both corrected HighBuild masks,
and every reference pixel remain fixed. This is an inference-protocol audit,
not a trained-model improvement. The one alternative is predeclared as the
app's 512-pixel tiles, 128 overlap, zero padding and CUDA fp16 autocast.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_corrected_legacy_triplet as legacy
from msr.data.raster_dataset import IMAGENET_MEAN, IMAGENET_STD
from msr.inference.predict import load_predictor, predict_height

NATIVE_REPORT = "outputs/evaluation/corrected_legacy_triplet_v1/report.json"
NATIVE_SHA = "0977bc90e034746df74dc6241c6676f15a6ea5f36dc0b422008a75c6767ab351"
PLAN = "configs/corrected_legacy_triplet_v1.yaml"
OUTPUT = "outputs/evaluation/protected_inference_protocol_v1/report.json"
STRICT = "highbuild_measured_coco_intersection_v1"
INCLUSIVE = "highbuild_all_annotated_positive_v1"


def require_same_normalized_rgb(raw: np.ndarray, sample: dict) -> None:
    """Refuse any raw-source/preprocessed-input mismatch before inference."""
    normalized = (raw.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    if not np.array_equal(normalized, sample["image"].numpy()):
        raise legacy.CorrectedTripletError("Raw RGB differs from authenticated model input")


def metric_deltas(candidate: dict, baseline: dict) -> dict:
    result = {}
    for key, value in candidate.items():
        if isinstance(value, dict) and isinstance(baseline.get(key), dict):
            result[key] = metric_deltas(value, baseline[key])
        elif key in legacy.METRIC_KEYS and value is not None and baseline.get(key) is not None:
            result[key] = value - baseline[key]
    return result


def prepare() -> tuple[dict, dict, dict, dict]:
    plan = legacy.load_plan(legacy._resolve(PLAN))
    native_identity = legacy._check_file(NATIVE_REPORT, NATIVE_SHA, "frozen native result")
    native = legacy._load_json(Path(native_identity["path"]))
    if native.get("official_test_used") is not False or native.get("split") != "validation":
        raise legacy.CorrectedTripletError("Only frozen development validation may be replayed")
    protected = plan["models"]["protected"]
    checkpoint = legacy._check_file(protected["checkpoint"], protected["checkpoint_sha256"], "protected checkpoint")
    pointer = legacy._check_file(plan["protocol"]["live_pointer"], plan["protocol"]["live_pointer_sha256"], "live pointer")
    if legacy._pointer_target(Path(pointer["path"])) != Path(checkpoint["path"]):
        raise legacy.CorrectedTripletError("Live pointer no longer selects protected checkpoint")
    data = legacy.authenticate_data(plan)
    identities = {"native_report": native_identity, "checkpoint": checkpoint, "pointer": pointer}
    return plan, native, data, identities


def run(*, output: Path, integrity_only: bool = False) -> dict:
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    plan, native, data_auth, identities = prepare()
    inference_sources = [
        "src/msr/inference/predict.py", "src/msr/inference/tiling.py",
        "src/msr/models/domain_surface_net.py", "src/msr/models/height_net.py",
        "src/msr/data/surface_dataset.py", "scripts/evaluate_corrected_legacy_triplet.py",
        "scripts/audit_protected_inference_protocol_v1.py",
    ]
    source_hashes = {p: legacy.file_sha256(legacy._resolve(p)) for p in inference_sources}
    data = plan["data"]
    protocol_maps = {
        name: {r.sample_id: r for r in records if r.target_kind == "building_height"}
        for name, records in data_auth["protocol_records"].items()
    }
    highbuild = data_auth["highbuild_records"]
    inclusive_records = [protocol_maps[INCLUSIVE][r.sample_id] for r in highbuild]
    strict_records = [protocol_maps[STRICT][r.sample_id] for r in highbuild]
    open_records = data_auth["open_canopy_records"]
    for record in inclusive_records:
        legacy._require_native_shape(record, 1024)
    for record in open_records:
        legacy._require_native_shape(record, 384)
    datasets = {
        "inclusive": legacy._dataset(inclusive_records, patch_size=1024, data=data),
        "strict": legacy._dataset(strict_records, patch_size=1024, data=data),
        "open": legacy._dataset(open_records, patch_size=384, data=data),
    }
    if integrity_only:
        return {"status": "integrity_passed", "sample_counts": {"highbuild": len(highbuild), "open_canopy": len(open_records)}, "identities": identities, "source_hashes": source_hashes, "official_test_used": False}
    if not torch.cuda.is_available():
        raise RuntimeError("This predeclared deployed-fp16 audit requires CUDA")
    model, metadata = load_predictor(identities["checkpoint"]["path"], device="cuda")
    expected = plan["models"]["protected"]
    if metadata["model_type"] != expected["model_type"] or metadata["epoch"] != expected["epoch"]:
        raise legacy.CorrectedTripletError("Protected model metadata mismatch")
    model.eval()
    books = {p: {scope: legacy.MetricScopes() for scope in ("combined", "highbuild", "open_canopy")} for p in (INCLUSIVE, STRICT)}
    model_digest = hashlib.sha256()
    supervision_digest = hashlib.sha256()
    scene_order = []
    scene_results = []
    work = [("highbuild", index, record) for index, record in enumerate(inclusive_records)]
    work += [("open_canopy", index, record) for index, record in enumerate(open_records)]
    for suite, index, record in tqdm(work, desc="Protected app512 vs native", unit="scene"):
        sample = datasets["inclusive" if suite == "highbuild" else "open"][index]
        scene_order.append(record.sample_id)
        model_digest.update(record.sample_id.encode("utf-8") + b"\0")
        for field in ("image", "relative_prior", "image_valid_mask"):
            legacy._hash_array(model_digest, field, sample[field])
        if suite == "highbuild":
            strict = datasets["strict"][index]
            legacy._assert_shared_highbuild_samples(sample, strict, record.sample_id)
            fields = (("inclusive_target", sample["height"]), ("inclusive_regression", sample["regression_mask"]), ("strict_regression", strict["regression_mask"]), ("domain_target", sample["domain_target"]), ("domain_valid", sample["domain_valid_mask"]))
            masks = {INCLUSIVE: sample["regression_mask"][0].numpy(), STRICT: strict["regression_mask"][0].numpy()}
        else:
            fields = (("open_target", sample["height"]), ("open_regression", sample["regression_mask"]), ("open_domain_target", sample["domain_target"]), ("open_domain_valid", sample["domain_valid_mask"]))
            masks = {p: sample["regression_mask"][0].numpy() for p in (INCLUSIVE, STRICT)}
        for field, value in fields:
            legacy._hash_array(supervision_digest, field, value)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(record.rgb_path) as source:
                raw = source.read((1, 2, 3))
        require_same_normalized_rgb(raw, sample)
        result = predict_height(raw, model=model, device="cuda", tile_size=512, overlap=128, rgb_scale=255.0, amp=True, model_metadata=metadata, relative_prior=sample["relative_prior"][0].numpy().copy(), valid_mask=sample["image_valid_mask"][0].numpy(), padding_policy="fixed_tile")
        target = sample["height"][0].numpy()
        if result.height_map.shape != target.shape:
            raise legacy.CorrectedTripletError("Tiled output changed reference grid")
        if not np.isfinite(result.height_map[masks[STRICT]]).all():
            raise legacy.CorrectedTripletError("Nonfinite predictions on scored support")
        for protocol, mask in masks.items():
            for scope in ("combined", suite):
                books[protocol][scope].update(result.height_map, target, mask, landscape=record.landscape, region=record.region, domain_target=sample["domain_target"].numpy())
        metric = legacy.StreamingRegressionMetrics()
        metric.update(result.height_map, target, masks[STRICT])
        scene_results.append({"sample_id": record.sample_id, "suite": suite, "metrics": metric.compute() if np.any(masks[STRICT]) else None, "strict_valid_pixels": int(np.count_nonzero(masks[STRICT]))})
    identity = {"ordered_sample_ids_sha256": legacy.canonical_json_sha256(scene_order), "model_input_tensors_sha256": model_digest.hexdigest(), "supervision_tensors_sha256": supervision_digest.hexdigest()}
    for key, value in identity.items():
        if native["identical_input_identity"][key] != value:
            raise legacy.CorrectedTripletError(f"Full-scene identity changed: {key}")
    native_metrics = native["models"]["protected"]["protocol_metrics"]
    protocol_results = {}
    for protocol in (INCLUSIVE, STRICT):
        scopes = {scope: book.compute() for scope, book in books[protocol].items()}
        for scope, metrics in scopes.items():
            if metrics["pixel_count"] != native_metrics[protocol][scope]["pixel_count"]:
                raise legacy.CorrectedTripletError(f"Pixel support changed for {protocol}/{scope}")
        protocol_results[protocol] = {"app_tiled": scopes, "native": {s: native_metrics[protocol][s] for s in scopes}, "app_minus_native": {s: metric_deltas(scopes[s], native_metrics[protocol][s]) for s in scopes}}
    for role in ("checkpoint", "pointer"):
        legacy._check_file(identities[role]["path"], identities[role]["sha256"], role)
    if source_hashes != {p: legacy.file_sha256(legacy._resolve(p)) for p in inference_sources}:
        raise legacy.CorrectedTripletError("Inference implementation changed during audit")
    report = {"schema": "msr.protected_inference_protocol.v1", "created_utc": datetime.now(timezone.utc).isoformat(), "status": "completed", "interpretation": "Same model, images and full-scene scored pixels. Tiling, zero padding and deployed fp16 versus native bf16 change inference; this is not a learned accuracy improvement.", "official_test_used": False, "external_holdout_used": False, "model_changed": False, "promotion_performed": False, "protocol": {"alternative_count": 1, "native_precision": "bf16", "app_precision": "fp16", "tile_size": 512, "overlap": 128, "prior": "same cached stored_01", "highbuild_count": 160, "open_canopy_count": 120}, "identities": identities, "source_hashes": source_hashes, "input_identity": identity, "same_full_pixel_support": True, "protocols": protocol_results, "per_scene": scene_results}
    legacy._atomic_json(output, report, replace=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=legacy._resolve(OUTPUT))
    parser.add_argument("--integrity-only", action="store_true")
    args = parser.parse_args()
    result = run(output=args.output.resolve(), integrity_only=args.integrity_only)
    print(result["status"])
    if not args.integrity_only:
        print(result["protocols"][STRICT]["app_tiled"]["combined"])


if __name__ == "__main__":
    main()
