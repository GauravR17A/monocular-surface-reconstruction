"""Frozen six-class RGB inference on the existing HighBuild validation subset.

HighBuild polygons are positive-only building labels, NOT exhaustive land-cover
truth. Report annotated-building coverage/recall, never fabricated precision,
six-class F1, counting accuracy, or height accuracy. All outputs are offline.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from PIL import Image
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from scipy.ndimage import binary_erosion
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from evaluate_rgb_segmenter_checkpoint import (
    KNOWN_EXPERIMENT, KNOWN_CHECKPOINT_SHA256, CLASS_NAMES, COLORS,
    RgbSegformer, load_known_checkpoint, sha256, atomic_json, assert_hash,
    capture_file_identities, verify_file_identities,
)
from evaluate_highbuild_instances import _CocoShardStore, _load_and_verify_contract, _read_csv
from msr.data.highbuild import annotation_masks
from msr.evaluation.highbuild_instances import parse_highbuild_instances


def summarize_building_support(prediction, footprints, image_valid):
    prediction, footprints, image_valid = map(np.asarray, (prediction, footprints, image_valid))
    if prediction.ndim != 2 or prediction.shape != footprints.shape or prediction.shape != image_valid.shape:
        raise ValueError("Prediction, footprints and optical validity must share a 2D grid")
    if prediction.dtype.kind not in "iu" or np.any((prediction[image_valid] < 0) | (prediction[image_valid] > 5)):
        raise ValueError("Expected six-class integer predictions on image-valid pixels")
    support = footprints.astype(bool) & image_valid
    counts = np.bincount(prediction[support].astype(np.int64), minlength=6)
    interior = binary_erosion(footprints.astype(bool), structure=np.ones((5, 5), bool)) & image_valid
    edge = support & ~interior
    return {
        "building_reference_pixels": int(support.sum()),
        "building_correct_pixels": int(counts[1]),
        "building_reference_predicted_as_counts": counts.tolist(),
        "building_pixel_recall": float(counts[1] / counts.sum()) if counts.sum() else None,
        "interior_reference_pixels": int(interior.sum()),
        "interior_correct_pixels": int(((prediction == 1) & interior).sum()),
        "edge_reference_pixels": int(edge.sum()),
        "edge_correct_pixels": int(((prediction == 1) & edge).sum()),
        "predicted_class_pixel_counts": np.bincount(prediction[image_valid].astype(np.int64), minlength=6).tolist(),
        "image_valid_pixels": int(image_valid.sum()),
        "building_precision": None, "building_f1": None, "building_iou": None,
        "six_class_macro_f1": None,
        "unavailable_reason": "Outside annotated buildings is unknown, not a verified negative class; other five classes lack truth.",
    }


def aggregate(rows):
    keys = ("building_reference_pixels", "building_correct_pixels", "interior_reference_pixels",
            "interior_correct_pixels", "edge_reference_pixels", "edge_correct_pixels", "image_valid_pixels")
    result = {key: sum(row[key] for row in rows) for key in keys}
    result["images"] = len(rows)
    for name, numerator, denominator in (
        ("building_pixel_recall", "building_correct_pixels", "building_reference_pixels"),
        ("interior_recall", "interior_correct_pixels", "interior_reference_pixels"),
        ("edge_recall", "edge_correct_pixels", "edge_reference_pixels"),
    ):
        result[name] = result[numerator] / result[denominator] if result[denominator] else None
    for key in ("building_reference_predicted_as_counts", "predicted_class_pixel_counts"):
        result[key] = np.sum([row[key] for row in rows], axis=0, dtype=np.int64).tolist() if rows else [0]*6
    support = result["building_reference_pixels"]
    result["building_confusion_fraction"] = {name: count/support if support else None
        for name, count in zip(CLASS_NAMES, result["building_reference_predicted_as_counts"])}
    result["building_precision"] = result["building_f1"] = result["building_iou"] = result["six_class_macro_f1"] = None
    return result


def object_coverage(prediction, instances, valid):
    rows = []
    for instance in instances:
        support = instance.mask & valid
        count = int(support.sum())
        if not count:
            continue
        correct = int(((prediction == 1) & support).sum())
        rows.append({"annotation_id": instance.annotation_id, "reference_pixels": count,
                     "building_fraction": correct/count, "correct_pixels": correct,
                     "size_bin_pixels": "small_<256" if count < 256 else "medium_256_4095" if count < 4096 else "large_4096_plus"})
    return rows


def summarize_objects(objects):
    values = np.asarray([row["building_fraction"] for row in objects])
    return {"annotated_objects_with_valid_pixels": len(objects),
            "mean_reference_footprint_coverage": float(values.mean()) if len(values) else None,
            "fraction_at_least_half_covered": float((values >= .5).mean()) if len(values) else None,
            "fraction_at_least_80_percent_covered": float((values >= .8).mean()) if len(values) else None,
            "fraction_less_than_10_percent_covered": float((values < .1).mean()) if len(values) else None,
            "note": "Diagnostic coverage of supplied outlines, not matched-object detection recall or count accuracy."}


def palette():
    return np.array([[int(color[i:i+2], 16) for i in (1, 3, 5)] for color in COLORS], dtype=np.uint8)


def save_preview(output, sample_id, rgb, prediction, footprint, recall):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors = palette()[prediction]
    evidence = rgb.copy()
    correct, missed = footprint & (prediction == 1), footprint & (prediction != 1)
    evidence[correct] = (evidence[correct]*.4 + np.array([38, 220, 151])*.6).astype(np.uint8)
    evidence[missed] = (evidence[missed]*.4 + np.array([248, 75, 91])*.6).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor="#101924")
    for axis, image, title in zip(axes, (rgb, colors, evidence),
                                 ("Original RGB", "Six predicted categories", "Known buildings: green found / red missed")):
        axis.imshow(image); axis.set_title(title, color="white", fontsize=11); axis.axis("off")
    score = "no building support" if recall is None else f"annotated-building pixel recall {recall:.1%}"
    fig.suptitle(f"{sample_id}\n{score} | not overall classification accuracy", color="white", fontsize=12)
    fig.legend(handles=[Patch(color=color, label=name.replace("_", " ")) for color, name in zip(COLORS, CLASS_NAMES)],
               loc="lower center", ncol=6, facecolor="#101924", labelcolor="white", frameon=False)
    fig.subplots_adjust(top=.82, bottom=.12, left=.015, right=.985, wspace=.035)
    fig.savefig(output / "previews" / f"{sample_id}.jpg", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_gallery(output, rows, model_label="Frozen RGB SegFormer V1"):
    legend = " ".join(f'<span style="color:{c}">■ {html.escape(n.replace("_", " "))}</span>' for c,n in zip(COLORS, CLASS_NAMES))
    cards = []
    # Fixed city/sample order. Do not hide failures or rank only attractive scenes.
    for row in rows:
        sid = row["sample_id"]
        score = row["building_pixel_recall"]
        label = "N/A" if score is None else f"{score:.1%}"
        cards.append(f'<article><h3>{html.escape(sid)}</h3><p>Annotated-building pixel recall: {label}</p>'
                     f'<div class="pair"><img loading="lazy" src="rgb/{sid}.jpg" alt="Original RGB">'
                     f'<img loading="lazy" src="classes/{sid}_colours.png" alt="Predicted categories"></div></article>')
    page = '<!doctype html><meta charset="utf-8"><title>Monocular Surface Reconstruction | HighBuild classification audit</title>'
    page += '<style>body{background:#101924;color:#eaf1fa;font:16px system-ui;margin:32px auto;max-width:1300px;padding:0 20px}h1{color:#74e6c2}p{line-height:1.6}article{background:#192636;border:1px solid #3a4a61;border-radius:12px;padding:18px;margin:24px 0}img{width:100%;min-width:0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}span{margin-right:16px}</style>'
    page += f'<h1>Monocular Surface Reconstruction — HighBuild classification check</h1><p>{html.escape(model_label)} · 160 development images · 4 cities · full native-resolution inference. No training or default-model change.</p>'
    page += '<p><strong>Read these honestly:</strong> building recall counts known building pixels that were labelled building. It does not measure false positives. HighBuild does not supply complete six-category truth; tree/grass/road/water/ground colours below are predictions, not verified accuracy. All tested scenes are shown, including failures.</p>'
    page += f'<p>{legend}</p>' + ''.join(cards)
    (output / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-v3", action="store_true", help="Audit the pinned experimental V3 epoch-1 classifier; never promote it")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; existing evidence is preserved")
    output.mkdir(parents=True)
    for name in ("classes", "rgb", "previews"):
        (output / name).mkdir()
    started = time.monotonic()
    status = lambda stage, **details: atomic_json(output / "status.json", {"stage": stage, **details})
    try:
        config_path = ROOT / "configs/corrected_legacy_triplet_v1.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data = config["data"]
        protocol = data["highbuild_protocols"]["highbuild_measured_coco_intersection_v1"]
        manifest, contract = Path(protocol["manifest"]), Path(protocol["contract_report"])
        assert_hash(manifest, protocol["manifest_sha256"])
        assert_hash(contract, protocol["contract_report_sha256"])
        assert_hash(Path(data["highbuild_source_index"]), data["highbuild_source_index_sha256"])
        _load_and_verify_contract(report_path=contract, manifest_path=manifest, split="validation")
        rows = sorted([row for row in _read_csv(manifest) if row["target_kind"] == "building_height"], key=lambda row:row["sample_id"])
        if len(rows) != 160 or len({r["sample_id"] for r in rows}) != 160 or sorted(Counter(r["region"] for r in rows).values()) != [40]*4:
            raise RuntimeError("Expected the complete existing four-city/160-scene HighBuild subset")
        protected = capture_file_identities([ROOT / config["protocol"]["live_pointer"],
            ROOT / config["models"]["protected"]["checkpoint"], KNOWN_EXPERIMENT / "checkpoint_best_guarded.pt"])
        atomic_json(output / "protected_before.json", protected)
        candidate_path = KNOWN_EXPERIMENT / "checkpoint_best_guarded.pt"
        candidate_sha = KNOWN_CHECKPOINT_SHA256
        model_label = "Frozen RGB SegFormer V1"
        if args.candidate_v3:
            from msr.inference.rgb_preview import CHECKPOINT, CHECKPOINT_SHA256
            candidate_path, candidate_sha = CHECKPOINT, CHECKPOINT_SHA256
            assert_hash(candidate_path, candidate_sha)
            payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
            model_label = "Experimental RGB multi-region V3 / epoch 1"
            # Architecture implementation is inherited unchanged from the pinned V1.
            original_payload = load_known_checkpoint(KNOWN_EXPERIMENT)
            implementation_hash = original_payload["binding"]["source_code_sha256"]["src/msr/models/rgb_segmenter.py"]
            del original_payload
        else:
            payload = load_known_checkpoint(KNOWN_EXPERIMENT)
            implementation_hash = payload["binding"]["source_code_sha256"]["src/msr/models/rgb_segmenter.py"]
        assert_hash(ROOT / "src/msr/models/rgb_segmenter.py", implementation_hash)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native CUDA bf16 required for the fixed inference recipe")
        torch.set_num_threads(4)
        torch.manual_seed(20260919)
        model = RgbSegformer.from_architecture(payload["architecture"])
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.requires_grad_(False).eval().to("cuda")
        del payload
        # These representative previews are selected BEFORE inference; all
        # 160 predictions remain available in the gallery and raw class maps.
        preview_ids = {r["sample_id"] for region in sorted({r["region"] for r in rows})
                       for r in [x for x in rows if x["region"] == region][::13]}
        atomic_json(output / "protocol.json", {"checkpoint_sha256": candidate_sha, "model_label": model_label,
            "manifest_sha256": sha256(manifest), "contract_sha256": sha256(contract),
            "source_index_sha256": data["highbuild_source_index_sha256"],
            "source_script_sha256": sha256(Path(__file__)), "scene_ids": [r["sample_id"] for r in rows],
            "preview_ids_selected_before_inference": sorted(preview_ids),
            "inference": "raw RGB/255; normalization inside frozen model; native grid; CUDA bf16; argmax; no TTA or postprocessing",
            "reference_policy": "All COCO building outlines, regardless of height-label availability. Unknown background ignored.",
            "data_scope": "Existing development/regression subset. Cities excluded from standalone DC/PHL training, not system-untouched geography.",
            "training": False, "promotion": False, "heights_evaluated": False})
        results, objects = [], []
        with _CocoShardStore(source_index=Path(data["highbuild_source_index"]),
                split_root=Path(data["highbuild_split_root"]), expected_split="validation") as store:
            for index, row in enumerate(rows):
                sid = row["sample_id"]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", NotGeoreferencedWarning)
                    with rasterio.open(row["rgb_path"]) as source:
                        if source.count != 3 or source.dtypes != ("uint8",)*3:
                            raise RuntimeError("Expected native uint8 RGB, not normalized or resized imagery")
                        rgb = source.read().transpose(1, 2, 0)
                        valid = np.all(source.read_masks() > 0, axis=0)
                    with rasterio.open(row["building_mask_path"]) as source:
                        footprint = source.read(1).astype(bool)
                height, width = rgb.shape[:2]
                coco = store.load(sid)
                expected, _, _ = annotation_masks(coco, height=height, width=width)
                if not np.array_equal(footprint, expected.astype(bool)):
                    raise RuntimeError(f"COCO and corrected semantic mask differ: {sid}")
                tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255).unsqueeze(0).to("cuda")
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(tensor)["logits"].argmax(1)[0].cpu().numpy().astype(np.uint8)
                metrics = summarize_building_support(prediction, footprint, valid)
                instances = parse_highbuild_instances(coco, height=height, width=width)
                object_rows = object_coverage(prediction, instances, valid)
                objects.extend({**item, "sample_id": sid, "region": row["region"]} for item in object_rows)
                result = {"sample_id": sid, "region": row["region"], "shape": [height, width],
                          "rgb_path": row["rgb_path"], "rgb_sha256": sha256(Path(row["rgb_path"])),
                          "footprint_sha256": sha256(Path(row["building_mask_path"])),
                          "coco_sha256": hashlib.sha256(json.dumps(coco, sort_keys=True).encode()).hexdigest(),
                          "objects": summarize_objects(object_rows), **metrics}
                Image.fromarray(prediction).save(output / "classes" / f"{sid}_ids.png")
                Image.fromarray(palette()[prediction]).save(output / "classes" / f"{sid}_colours.png")
                Image.fromarray(rgb).resize((512, 512)).save(output / "rgb" / f"{sid}.jpg", quality=90)
                if sid in preview_ids:
                    save_preview(output, sid, rgb, prediction, footprint & valid, metrics["building_pixel_recall"])
                results.append(result)
                with (output / "per_image.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                if (index+1) % 20 == 0 or index == 0:
                    status("inference", completed=index+1, total=len(rows), elapsed_seconds=time.monotonic()-started)
                    print(f"HighBuild classifier: {index+1}/{len(rows)} | building recall so far {aggregate(results)['building_pixel_recall']:.1%}", flush=True)
        del model
        torch.cuda.empty_cache()
        verify_file_identities(protected)
        report = {"schema": "msr.highbuild_rgb_classification_diagnostic.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(candidate_path), "checkpoint_sha256": candidate_sha,
            "overall": aggregate(results),
            "by_city": {region: aggregate([r for r in results if r["region"] == region]) for region in sorted({r["region"] for r in results})},
            "object_coverage": summarize_objects(objects),
            "object_size_slices": {size:summarize_objects([r for r in objects if r["size_bin_pixels"] == size]) for size in sorted({r["size_bin_pixels"] for r in objects})},
            "limitations": ["Positive-only building labels: no precision, F1, IoU, six-class accuracy or count accuracy claimed.",
                "Other classes are model predictions, not ground truth; visual assessment is qualitative.",
                "This is existing project development data, not an untouched final benchmark or all HighBuild images.",
                "No height predictions evaluated; V3 is an opt-in identification preview, not a replacement height model."],
            "application_and_checkpoints_unchanged": True, "elapsed_seconds": time.monotonic()-started}
        atomic_json(output / "report.json", report)
        atomic_json(output / "per_object.json", objects)
        write_gallery(output, results, model_label)
        status("completed", completed=len(results), total=len(rows), elapsed_seconds=time.monotonic()-started)
        print(json.dumps({"output": str(output), "overall": report["overall"], "objects": report["object_coverage"]}), flush=True)
    except BaseException as error:
        status("failed", error=str(error), error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
