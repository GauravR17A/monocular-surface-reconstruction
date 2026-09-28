"""One-shot, frozen RGB V1 external transfer evaluation; never promotes a model."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import rasterio
import torch

from evaluate_rgb_segmenter_checkpoint import (
    KNOWN_EXPERIMENT, KNOWN_CHECKPOINT_SHA256, ROOT, RgbSegformer, MetricGroup,
    independent_confusion, independent_scores, load_known_checkpoint,
    validate_sealed_run, verify_file_identities, sha256, atomic_json,
)

DATA = Path("D:/MSRData/external_holdouts/oem_christchurch_rgb_v1")
OUTPUT = ROOT / "outputs/evaluation/christchurch_rgb_segmenter_v1"
PROTOCOL = ROOT / "docs/RGB_SEGMENTER_V1_CHRISTCHURCH_PROTOCOL.md"
CROSSWALK = np.array([255, 0, 4, 0, 3, 5, 2, 255, 1], dtype=np.uint8)


def remap(reference, agriculture=False):
    reference = np.asarray(reference)
    if reference.dtype.kind not in "iu" or np.any((reference < 0) | (reference > 8)):
        raise ValueError("Expected original OEM integer IDs 0..8")
    mapped = CROSSWALK[reference]
    if agriculture:
        mapped = mapped.copy()
        mapped[reference == 7] = 4
    return mapped


def native_confusion(prediction, source, valid):
    if prediction.shape != source.shape or source.shape != valid.shape:
        raise ValueError("Native-grid arrays differ")
    valid = valid & (source >= 1) & (source <= 8)
    if np.any((prediction[valid] < 0) | (prediction[valid] >= 6)):
        raise ValueError("Unexpected prediction")
    return np.bincount((source[valid].astype(np.int64)-1)*6+prediction[valid], minlength=48).reshape(8, 6)


def summarize(matrix, tiles):
    result = independent_scores(matrix)
    for index, row in enumerate(result["per_class"].values()):
        row["support_tiles"] = int(sum(np.asarray(tile)[index].sum() > 0 for tile in tiles))
        row["reference_class_present"] = row["support_pixels"] > 0
        row["coverage_sufficient"] = row["support_tiles"] >= 5 and row["support_pixels"] >= 10000
    result["comprehensive_six_class_coverage"] = all(row["coverage_sufficient"] for row in result["per_class"].values())
    result["undefined_ratio_policy"] = "Zero for undefined ratio; absent reference classes explicitly flagged, not perfect. Macro always includes six classes."
    tp = int(matrix[4:6, 4:6].sum())
    denominator = int(matrix[4:6].sum()+matrix[:, 4:6].sum())
    result["diagnostic_combined_vegetation_f1"] = 2*tp/denominator if denominator else None
    return result


def bootstrap(tiles, resamples=2000):
    values = np.asarray(tiles, dtype=np.int64)
    if values.ndim != 3 or values.shape[1:] != (6, 6) or len(values) == 0:
        raise ValueError("Require nonempty whole-tile confusion matrices")
    rng = np.random.default_rng(20260915)
    samples = []
    for _ in range(resamples):
        matrix = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        tp = np.diag(matrix).astype(float)
        support, predicted = matrix.sum(1), matrix.sum(0)
        f1 = np.divide(2*tp, support+predicted, out=np.zeros(6), where=(support+predicted)>0)
        iou = np.divide(tp, support+predicted-tp, out=np.zeros(6), where=(support+predicted-tp)>0)
        samples.append([*f1, f1.mean(), iou.mean()])
    interval = np.quantile(samples, [.025, .975], axis=0)
    return {"seed": 20260915, "resamples": resamples, "unit": "whole released tile",
            "warning": "Descriptive within-city interval; adjacent tiles can be correlated. Not global geographic uncertainty.",
            "f1_per_class_95_percentile": interval[:, :6].T.tolist(),
            "macro_f1_95_percentile": interval[:, 6].tolist(),
            "macro_iou_95_percentile": interval[:, 7].tolist()}


def checked_files(manifest):
    groups = {"images": {}, "labels": {}}
    for row in manifest["files"]:
        path = (DATA / row["path"]).resolve()
        if not path.is_relative_to(DATA.resolve()) or path.parent.name not in groups or path.suffix != ".tif":
            raise ValueError("Invalid staged path")
        if sha256(path) != row["sha256"] or path.stat().st_size != row["source"]["bytes"] or row["crc32_verified"] is not True:
            raise RuntimeError("Staged image/label identity mismatch")
        if path.stem in groups[path.parent.name]:
            raise ValueError("Duplicate tile")
        groups[path.parent.name][path.stem] = path
    if set(groups["images"]) != set(groups["labels"]) or len(groups["images"]) != manifest["pair_count"]:
        raise RuntimeError("Incomplete paired manifest")
    return groups


def validate_raster_headers(image, label):
    # Metadata only: fail before consumption for unsupported formats/grids.
    if image.count != 3 or image.dtypes != ("uint8",)*3 or label.count != 1 or image.shape != label.shape:
        raise RuntimeError("Unsupported radiometry or paired dimensions")
    if min(image.shape) < 32 or max(image.shape) > 1024 or image.crs != label.crs or not image.transform.almost_equals(label.transform):
        raise RuntimeError("Unexpected native size or inconsistent geospatial grids")


def main():
    if OUTPUT.exists() or (DATA / "CONSUMED.json").exists():
        raise RuntimeError("Existing evaluation/consumption; no undisclosed rerun permitted")
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != sha256(PROTOCOL) or manifest["contract_sha256"] != sha256(DATA / "acquisition_contract.json"):
        raise RuntimeError("Acquisition protocol or contract changed")
    contract = json.loads((DATA / "acquisition_contract.json").read_text(encoding="utf-8"))
    if len(manifest["files"]) != len(contract["entries"]) or sorted(row["source"]["name"] for row in manifest["files"]) != sorted(row["name"] for row in contract["entries"]):
        raise RuntimeError("Manifest does not cover the complete acquisition contract")
    groups = checked_files(manifest)
    for stem in sorted(groups["images"]):
        with rasterio.open(groups["images"][stem]) as image, rasterio.open(groups["labels"][stem]) as label:
            validate_raster_headers(image, label)
    payload = load_known_checkpoint(KNOWN_EXPERIMENT)
    config, binding, protected = validate_sealed_run(KNOWN_EXPERIMENT, payload)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Frozen inference requires CUDA bf16")
    torch.set_num_threads(4)
    torch.manual_seed(config["experiment"]["seed"])
    model = RgbSegformer.from_architecture(payload["architecture"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        smoke = model(torch.zeros(1, 3, 128, 128, device="cuda"))["logits"]
    if smoke.shape != (1, 6, 128, 128) or not torch.isfinite(smoke).all():
        raise RuntimeError("Synthetic inference failed before holdout consumption")
    sources = [Path(__file__), ROOT / "scripts/evaluate_rgb_segmenter_checkpoint.py",
               PROTOCOL, ROOT / "src/msr/models/rgb_segmenter.py",
               ROOT / "src/msr/evaluation/classification_metrics.py"]
    seal = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": KNOWN_CHECKPOINT_SHA256,
            "manifest_sha256": sha256(manifest_path),
            "sources": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
            "software": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "rasterio", "transformers", "scipy")},
            "python": sys.version, "crosswalk": CROSSWALK.tolist(),
            "preview_order": sorted(groups["images"], key=lambda name: hashlib.sha256(name.encode()).hexdigest())[:6],
            "tile_order": sorted(groups["images"]), "run_config": config,
            "local_pre_evaluation_commitment_only": True, "height_inputs": False,
            "precision": "bf16", "batch_size": 1, "dark_mean_rgb_threshold": .15}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    atomic_json(OUTPUT / "seal.json", seal)
    marker = {"first_pixel_decode_authorized_utc": datetime.now(timezone.utc).isoformat(),
              "seal_sha256": sha256(OUTPUT / "seal.json"), "output": str(OUTPUT),
              "warning": "Any failure after this marker counts as consumption"}
    with (DATA / "CONSUMED.json").open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2)
    try:
        overall = MetricGroup()
        native = np.zeros((8, 6), dtype=np.int64)
        agriculture = np.zeros((6, 6), dtype=np.int64)
        tiles, sensitivity_tiles = [], []
        total_pixels = ignored = 0
        (OUTPUT / "predictions").mkdir()
        with (OUTPUT / "per_image_metrics.jsonl").open("x", encoding="utf-8") as stream:
            for index, stem in enumerate(seal["tile_order"], 1):
                with rasterio.open(groups["images"][stem]) as image, rasterio.open(groups["labels"][stem]) as label:
                    validate_raster_headers(image, label)
                    rgb = image.read()
                    original = label.read(1)
                    image_valid = np.all(image.read_masks() > 0, axis=0)
                    label_valid = label.read_masks(1) > 0
                target = remap(original)
                valid = image_valid & label_valid & (target != 255)
                source_valid = image_valid & label_valid
                input_rgb = torch.from_numpy(rgb.astype(np.float32)/255).unsqueeze(0).cuda()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_rgb)["logits"]
                if logits.shape != (1, 6, *original.shape) or not torch.isfinite(logits).all():
                    raise RuntimeError("Invalid native prediction")
                prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                matrix = independent_confusion(prediction, target, valid)
                dark = image_valid & (rgb.astype(np.float32).mean(0)/255 <= .15)
                overall.update(matrix, prediction, target, dark, valid)
                tiles.append(matrix)
                native += native_confusion(prediction, original, source_valid)
                sensitive_target = remap(original, agriculture=True)
                sensitive = independent_confusion(prediction, sensitive_target, source_valid & (sensitive_target != 255))
                agriculture += sensitive
                sensitivity_tiles.append(sensitive)
                total_pixels += valid.size
                ignored += int((~valid).sum())
                np.save(OUTPUT / "predictions" / f"{stem}.npy", prediction, allow_pickle=False)
                record = {"sample_id": stem, "shape": list(original.shape), "six_class_identification": independent_scores(matrix),
                          "ignored_pixels": int((~valid).sum()), "image_valid_pixels": int(image_valid.sum()),
                          "label_valid_pixels": int(label_valid.sum()),
                          "prediction_sha256": sha256(OUTPUT / "predictions" / f"{stem}.npy")}
                stream.write(json.dumps(record)+"\n")
                stream.flush()
                atomic_json(OUTPUT / "status.json", {"stage": "evaluating", "completed": index, "total": len(groups["images"])})
                if index % 10 == 0 or index == len(groups["images"]):
                    print(f"External test {index}/{len(groups['images'])} tiles", flush=True)
        verify_file_identities(protected)
        checked_files(manifest)
        if any(sha256(ROOT / name) != expected for name, expected in seal["sources"].items()) or sha256(manifest_path) != seal["manifest_sha256"]:
            raise RuntimeError("Inputs or evaluator changed during test")
        report = {"stage": "completed", "scope": "One-city transfer under fixed OEM crosswalk; not height accuracy or global generalisation",
                  "checkpoint_sha256": KNOWN_CHECKPOINT_SHA256, "tile_count": len(tiles),
                  "primary": summarize(overall.matrix, tiles),
                  "agriculture_as_low_vegetation_sensitivity": summarize(agriculture, sensitivity_tiles),
                  "native_oem_8_by_6": native.tolist(), "native_rows": list(range(1, 9)),
                  "ignored_share": ignored/total_pixels, "bootstrap": bootstrap(tiles),
                  "road_boundary_quality": overall.road.compute(), "dark_pixel_water_proxy": overall.water.compute(),
                  "protected_model_and_pointer_unchanged": True, "app_model_changed": False,
                  "seal_sha256": sha256(OUTPUT / "seal.json"), "excluded_no_released_label": contract["excluded_no_released_label"]}
        atomic_json(OUTPUT / "report.json", report)
        atomic_json(OUTPUT / "status.json", {"stage": "completed", "completed": len(tiles), "total": len(tiles), "report_sha256": sha256(OUTPUT / "report.json")})
        print(json.dumps({"macro_f1": report["primary"]["macro_f1"], "macro_iou": report["primary"]["macro_iou"], "tile_count": len(tiles)}), flush=True)
    except Exception as error:
        atomic_json(OUTPUT / "status.json", {"stage": "failed_after_consumption", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
