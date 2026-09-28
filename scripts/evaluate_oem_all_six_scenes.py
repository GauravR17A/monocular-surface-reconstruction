"""Score frozen RGB V1 on a label-selected all-six-class external challenge set."""
from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from pathlib import Path
import time
import warnings

import numpy as np
from PIL import Image
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import torch

from stage_oem_all_six_scenes import DESTINATION, digest, CROSSWALK, ATTRIBUTION, MIN_CLASS_PIXELS
from evaluate_rgb_segmenter_checkpoint import (
    ROOT, KNOWN_EXPERIMENT, KNOWN_CHECKPOINT_SHA256, CLASS_NAMES, COLORS,
    RgbSegformer, load_known_checkpoint, independent_confusion, independent_scores,
    MetricGroup, atomic_json, assert_hash, capture_file_identities, verify_file_identities,
)
from evaluate_christchurch_rgb_v1 import remap, native_confusion, summarize, validate_raster_headers


def oem_descriptions(value):
    """Keep reused helper prose accurate about this run's reference dataset."""
    if isinstance(value, dict):
        return {key:oem_descriptions(item) for key,item in value.items()}
    if isinstance(value, list):
        return [oem_descriptions(item) for item in value]
    return value.replace("GAMUS", "OpenEarthMap") if isinstance(value, str) else value


def export_scene(output, row, rgb, reference, prediction, valid, image_valid, scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    stem = row["sample_id"]
    target = output / "scenes" / stem
    target.mkdir(parents=True)
    colors = np.full((256, 3), 55, np.uint8)
    for i, color in enumerate(COLORS):
        colors[i] = [int(color[j:j+2], 16) for j in (1, 3, 5)]
    reference_ids = np.where(valid, reference, 255).astype(np.uint8)
    predicted_ids = np.where(image_valid, prediction, 255).astype(np.uint8)
    Image.fromarray(rgb).save(target / "original.png")
    Image.fromarray(reference_ids).save(target / "reference_ids.png")
    Image.fromarray(predicted_ids).save(target / "prediction_ids.png")
    Image.fromarray(colors[reference_ids]).save(target / "reference_colours.png")
    Image.fromarray(colors[predicted_ids]).save(target / "prediction_colours.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.9), facecolor="#101924")
    for ax, values, title in zip(axes, (rgb, colors[reference_ids], colors[predicted_ids]),
                               ("Original RGB", "Independent reference labels", "Frozen model prediction")):
        ax.imshow(values); ax.axis("off"); ax.set_title(title, color="white", fontsize=11)
    fig.suptitle(f"{stem} | all six reference classes\nMacro F1 {scores['macro_f1']:.1%} · Pixel accuracy {scores['accuracy']:.1%}", color="white", fontsize=13)
    fig.legend(handles=[Patch(color=c, label=n.replace("_", " ")) for c,n in zip(COLORS, CLASS_NAMES)],
               loc="lower center", ncol=6, facecolor="#101924", labelcolor="white", frameon=False)
    fig.subplots_adjust(top=.82, bottom=.12, left=.015, right=.985, wspace=.035)
    fig.savefig(target / "comparison.jpg", dpi=120, facecolor=fig.get_facecolor()); plt.close(fig)
    atomic_json(target / "metrics.json", scores)


def gallery(output, report, rows):
    scores = report["primary"]
    class_rows = ''.join(f'<tr><td>{html.escape(name.replace("_", " "))}</td>' +
        ''.join(f'<td>{values[key]:.1%}</td>' for key in ("precision", "recall", "f1", "iou")) + '</tr>'
        for name, values in scores["per_class"].items())
    cards = ''.join(f'<article><h2>{html.escape(row["sample_id"])}</h2>'
        f'<p>Macro F1 {row["scores"]["macro_f1"]:.1%} · Pixel accuracy {row["scores"]["accuracy"]:.1%}</p>'
        f'<a href="scenes/{row["sample_id"]}/original.png">Original image</a> · '
        f'<a href="scenes/{row["sample_id"]}/reference_colours.png">Reference labels</a> · '
        f'<a href="scenes/{row["sample_id"]}/prediction_colours.png">Prediction</a>'
        f'<img loading="lazy" src="scenes/{row["sample_id"]}/comparison.jpg" alt="Original, reference and prediction"></article>' for row in rows)
    page = '<!doctype html><meta charset="utf-8"><title>Monocular Surface Reconstruction | All six classes</title>'
    page += '<style>body{background:#101924;color:#edf4fa;font:17px system-ui;max-width:1400px;margin:36px auto;padding:0 20px}p{line-height:1.6}h1,a{color:#71e5c0}article{background:#1b293a;margin:28px 0;padding:20px;border-radius:12px}img{display:block;width:100%;margin-top:14px}td,th{padding:10px 24px;text-align:left;border-bottom:1px solid #45566a}table{border-collapse:collapse}.note{color:#ffc47d}</style>'
    page += f'<h1>All six classes — independent reference comparison</h1><p>{report["tile_count"]} images · {len(report["by_region"])} regions · frozen RGB SegFormer V1</p>'
    page += f'<h2>Macro F1: {scores["macro_f1"]:.1%} · Pixel accuracy: {scores["accuracy"]:.1%}</h2>'
    page += '<p class="note">Small label-selected mixed-scene challenge, not global accuracy. Selection required all six classes and was made before model inference; every selected image is shown. These are classification labels, not LiDAR or height measurements.</p>'
    page += '<table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>IoU</th></tr>'+class_rows+'</table>'
    page += '<p>Precision: how often a predicted class is right. Recall: how much of the true class is found. F1 balances both. IoU measures region overlap. Gray reference pixels are ignored, not ground.</p>'
    page += cards
    page += '<h2>Sources and limitations</h2><p>OpenEarthMap v1; GeoNRW / North Rhine-Westphalia (DL-DE-BY-2.0), AIGEO Center (CC BY 4.0), UR Field Lab Chiang Mai (CC BY 4.0). '
    page += '<a href="https://open-earth-map.org/attribution.html">Official source attribution</a>. '
    page += 'Bareland and developed space map to ground; rangeland maps to low vegetation; agriculture and unknown are ignored. See report.json for native-label confusion and agriculture sensitivity. No training or app-model replacement was performed.</p>'
    (output / "index.html").write_text(page, encoding="utf-8")


def main():
    data = DESTINATION
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract_path = data / "selection_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert_hash(contract_path, manifest["selection_contract_sha256"])
    if contract["checkpoint_sha256"] != KNOWN_CHECKPOINT_SHA256 or contract["crosswalk"] != CROSSWALK.tolist():
        raise RuntimeError("Model or crosswalk differs from pre-selection contract")
    if (data / "EVALUATION_STARTED.json").exists() or (data / "results").exists():
        raise RuntimeError("Preserve prior attempt; no undisclosed rerun or overwrite")
    pairs = manifest["pairs"]
    if not pairs or len({r["sample_id"] for r in pairs}) != len(pairs):
        raise RuntimeError("Unique nonempty image list required")
    paths = [manifest_path, contract_path, ROOT / "outputs/runtime/showcase_checkpoint.txt",
        ROOT / "experiments/20260829T203146Z_multidomain_surface_v2_guarded_vegetation/checkpoint_best_guarded_vegetation.pt",
        KNOWN_EXPERIMENT / "checkpoint_best_guarded.pt", Path(__file__),
        ROOT / "scripts/stage_oem_all_six_scenes.py", ROOT / "scripts/evaluate_christchurch_rgb_v1.py",
        ROOT / "scripts/evaluate_rgb_segmenter_checkpoint.py", ROOT / "src/msr/models/rgb_segmenter.py"]
    for row in pairs:
        for name in ("image", "label"):
            path = Path(row[f"{name}_path"])
            if not path.resolve().is_relative_to(data.resolve()):
                raise RuntimeError("Input path escaped selected dataset")
            assert_hash(path, row[f"{name}_receipt"]["sha256"])
            paths.append(path)
    protected = capture_file_identities(paths)
    payload = load_known_checkpoint(KNOWN_EXPERIMENT)
    assert_hash(ROOT / "src/msr/models/rgb_segmenter.py", payload["binding"]["source_code_sha256"]["src/msr/models/rgb_segmenter.py"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bf16 required by frozen inference recipe")
    torch.set_num_threads(4); torch.manual_seed(20260919)
    model = RgbSegformer.from_architecture(payload["architecture"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.requires_grad_(False).eval().to("cuda")
    del payload
    output = data / "results"
    output.mkdir()
    atomic_json(output / "seal.json", {"file_identities":protected, "selected_ids":[r["sample_id"] for r in pairs],
        "created_utc":datetime.now(timezone.utc).isoformat(), "checkpoint_sha256":KNOWN_CHECKPOINT_SHA256,
        "input":"raw RGB/255; internal ImageNet normalization; native resolution; no augmentation/postprocessing",
        "precision":"CUDA bf16", "selection":"All six classes >=256 labelled pixels, hash-order per region before inference",
        "scope":"Label-selected mixed-scene diagnostic; not representative global accuracy; now consumed for this model",
        "crosswalk":CROSSWALK.tolist(), "height_evaluation":False, "app_promotion":False})
    atomic_json(data / "EVALUATION_STARTED.json", {"seal_sha256":digest(output / "seal.json"), "warning":"Any inference attempt consumes this diagnostic sample; no longer untouched"})
    started = time.monotonic()
    results, matrices = [], []
    overall, groups = MetricGroup(), {}
    native = np.zeros((8, 6), np.int64)
    sensitive = np.zeros((6, 6), np.int64)
    total, ignored = 0, 0
    try:
        for index, row in enumerate(pairs, 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with rasterio.open(row["image_path"]) as image, rasterio.open(row["label_path"]) as label:
                    validate_raster_headers(image, label)
                    rgb = image.read()
                    labels = label.read(1)
                    image_valid = np.all(image.read_masks()>0, axis=0)
                    source_valid = image_valid & (label.read_masks(1)>0)
            reference = remap(labels)
            valid = source_valid & (reference != 255)
            input_rgb = torch.from_numpy(rgb.astype(np.float32)/255).unsqueeze(0).cuda()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_rgb)["logits"]
            if logits.shape != (1, 6, *labels.shape) or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid output grid or nonfinite logits")
            prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            matrix = independent_confusion(prediction, reference, valid)
            dark = image_valid & (rgb.astype(np.float32).mean(0)/255 <= .15)
            overall.update(matrix, prediction, reference, dark, valid)
            region = row["region"]
            groups.setdefault(region, MetricGroup()).update(matrix, prediction, reference, dark, valid)
            native += native_confusion(prediction, labels, source_valid)
            sensitivity = remap(labels, agriculture=True)
            sensitive += independent_confusion(prediction, sensitivity, source_valid & (sensitivity != 255))
            matrices.append(matrix)
            scores = independent_scores(matrix)
            record = {"sample_id":row["sample_id"], "region":region, "scores":scores,
                "all_six_on_joint_validity":bool(np.all(matrix.sum(1) >= MIN_CLASS_PIXELS)),
                "ignored_pixels":int((~valid).sum()), "shape":list(labels.shape)}
            results.append(record)
            with (output / "per_image.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False)+"\n")
            export_scene(output, row, rgb.transpose(1, 2, 0), reference, prediction, valid, image_valid, scores)
            total += valid.size; ignored += int((~valid).sum())
            atomic_json(output / "status.json", {"stage":"evaluating", "completed":index, "total":len(pairs)})
            print(f"All-six test {index}/{len(pairs)} | {row['sample_id']} | macro F1 {scores['macro_f1']:.1%}", flush=True)
        del model
        torch.cuda.empty_cache()
        verify_file_identities(protected)
        report = {"schema":"msr.oem_all_six_mixed_scene_diagnostic.v1", "stage":"completed",
            "checkpoint_sha256":KNOWN_CHECKPOINT_SHA256, "tile_count":len(results),
            "primary":summarize(overall.matrix, matrices), "by_region":{name:group.compute() for name,group in groups.items()},
            "agriculture_as_low_vegetation_sensitivity":independent_scores(sensitive),
            "native_oem_8_by_6":native.tolist(), "native_reference_ids":list(range(1,9)),
            "ignored_pixel_share":ignored/total, "all_selected_scenes_have_six_classes":all(r["all_six_on_joint_validity"] for r in results),
            "road_boundary_quality":overall.road.compute(), "dark_non_water_predicted_water":overall.water.compute(),
            "scope":"Small label-enriched all-six-present challenge set, NOT unbiased city/global accuracy or height accuracy",
            "attribution":ATTRIBUTION, "app_and_model_files_unchanged":True,
            "elapsed_seconds":time.monotonic()-started, "selection_contract_sha256":digest(contract_path)}
        report = oem_descriptions(report)
        atomic_json(output / "report.json", report)
        gallery(output, report, results)
        atomic_json(output / "status.json", {"stage":"completed", "completed":len(results), "total":len(pairs)})
        atomic_json(data / "EVALUATED.json", {"report_sha256":digest(output / "report.json"), "tile_count":len(pairs),
            "no_longer_untouched":True, "training":False, "app_model_changed":False})
        print(json.dumps({"output":str(output), "macro_f1":report["primary"]["macro_f1"], "accuracy":report["primary"]["accuracy"]}), flush=True)
    except BaseException as error:
        atomic_json(output / "status.json", {"stage":"failed_after_consumption", "error":str(error)})
        raise


if __name__ == "__main__":
    main()
