"""Paired fp16 fixed512 versus minimal384 inference on all 120 OC val scenes.

Only padding changes. The model, original source RGB, cached DAV2 prior,
independent reference and all scored pixels are shared by both calls.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import torch
from tqdm import tqdm

import audit_protected_inference_protocol_v1 as common
from msr.inference.predict import load_predictor, predict_height

OUTPUT = "outputs/evaluation/open_canopy_padding_v1/report.json"
FIXED_REPORT = "outputs/evaluation/protected_inference_protocol_v1/report.json"
FIXED_SHA = "b50f62b652c5b43608a54d2561fd179f41666bae2fd9f217e4dd04e9d15bd0e4"


def main() -> None:
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    legacy = common.legacy
    output = legacy._resolve(OUTPUT)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    plan, native, data_auth, identities = common.prepare()
    fixed_identity = legacy._check_file(FIXED_REPORT, FIXED_SHA, "previous actual app result")
    fixed_report = legacy._load_json(Path(fixed_identity["path"]))
    paths = ["src/msr/inference/predict.py", "src/msr/inference/tiling.py", "src/msr/models/domain_surface_net.py", "src/msr/models/height_net.py", "src/msr/data/surface_dataset.py", "scripts/audit_open_canopy_padding_v1.py"]
    implementation = {p: legacy.file_sha256(legacy._resolve(p)) for p in paths}
    records = data_auth["open_canopy_records"]
    if len(records) != 120:
        raise ValueError("Expected the complete 120-scene OpenCanopy validation set")
    for record in records:
        legacy._require_native_shape(record, 384)
    dataset = legacy._dataset(records, patch_size=384, data=plan["data"])
    if not torch.cuda.is_available():
        raise RuntimeError("The paired fp16 control requires CUDA")
    model, metadata = load_predictor(identities["checkpoint"]["path"], device="cuda")
    model.eval()
    books = {policy: legacy.MetricScopes() for policy in ("fixed_tile", "minimal")}
    digest = hashlib.sha256()
    scenes = []
    for index, record in enumerate(tqdm(records, desc="OC same fp16 padding A/B", unit="scene")):
        sample = dataset[index]
        with rasterio.open(record.rgb_path) as source:
            raw = source.read((1, 2, 3))
        common.require_same_normalized_rgb(raw, sample)
        digest.update(record.sample_id.encode("utf-8") + b"\0")
        for field in ("image", "relative_prior", "image_valid_mask", "height", "regression_mask", "domain_target", "domain_valid_mask"):
            legacy._hash_array(digest, field, sample[field])
        target = sample["height"][0].numpy()
        mask = sample["regression_mask"][0].numpy()
        item = {"sample_id": record.sample_id, "region": record.region, "rgb_path": str(record.rgb_path), "reference_path": str(record.surface_path), "valid_pixels": int(mask.sum()), "results": {}}
        for policy, book in books.items():
            predicted = predict_height(raw.copy(), model=model, device="cuda", tile_size=512, overlap=128, rgb_scale=255.0, amp=True, model_metadata=metadata, relative_prior=sample["relative_prior"][0].numpy().copy(), valid_mask=sample["image_valid_mask"][0].numpy(), padding_policy=policy)
            if predicted.height_map.shape != target.shape or not np.isfinite(predicted.height_map[mask]).all():
                raise ValueError("Prediction grid/support changed")
            book.update(predicted.height_map, target, mask, landscape=record.landscape, region=record.region, domain_target=sample["domain_target"].numpy())
            metric = legacy.StreamingRegressionMetrics()
            metric.update(predicted.height_map, target, mask)
            item["results"][policy] = metric.compute()
        scenes.append(item)
    metrics = {policy: book.compute() for policy, book in books.items()}
    cached = fixed_report["protocols"][common.STRICT]["app_tiled"]["open_canopy"]
    if metrics["fixed_tile"]["pixel_count"] != metrics["minimal"]["pixel_count"] or metrics["fixed_tile"]["pixel_count"] != cached["pixel_count"]:
        raise ValueError("Paired scored pixel support differs")
    for field in ("rmse_m", "mae_m", "bias_m", "correlation", "r2"):
        if abs(metrics["fixed_tile"][field] - cached[field]) > 1e-6:
            raise ValueError(f"Fixed-policy reproduction disagrees at {field}")
    for role in ("checkpoint", "pointer"):
        legacy._check_file(identities[role]["path"], identities[role]["sha256"], role)
    if implementation != {p: legacy.file_sha256(legacy._resolve(p)) for p in paths}:
        raise ValueError("Inference implementation changed during audit")
    report = {"schema": "msr.open_canopy_padding.v1", "created_utc": datetime.now(timezone.utc).isoformat(), "status": "completed", "protocol": {"split": "validation", "scene_count": 120, "same_pixel_support": True, "precision_both": "fp16", "tile_size_both": 512, "overlap_both": 128, "fixed_network_input": [512, 512], "minimal_network_input": [384, 384], "reference_grid_both": [384, 384], "official_test_used": False, "external_holdout_used": False, "changed_factor": "padding_only", "model_changed": False, "promotion_performed": False}, "identities": identities, "fixed_report": fixed_identity, "implementation": implementation, "input_and_supervision_sha256": digest.hexdigest(), "metrics": metrics, "minimal_minus_fixed": common.metric_deltas(metrics["minimal"], metrics["fixed_tile"]), "native_bf16_context": native["models"]["protected"]["protocol_metrics"][common.STRICT]["open_canopy"], "per_scene": scenes, "interpretation": "Development evidence for an inference-padding change only. Full HighBuild1024 scenes contain only complete512 tiles and are unchanged by the new policy; CPU tests prove identical tiling outputs for that geometry. Small urban images remain a separate unmeasured domain."}
    legacy._atomic_json(output, report, replace=False)
    for policy, metric in metrics.items():
        print(policy, {key: metric[key] for key in ("pixel_count", "rmse_m", "mae_m", "bias_m", "correlation", "r2")})
        print("domains", {name: value["rmse_m"] for name, value in metric["domains"].items()})


if __name__ == "__main__":
    main()
