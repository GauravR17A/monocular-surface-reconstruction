"""Pinned, experimental RGB classification. Never supplies heights or mesh masks."""
from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image

from msr.io.raster import read_rgb_raster
from msr.models.rgb_segmenter import CLASS_NAMES, RgbSegformer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = PROJECT_ROOT / "models/release/classifier_model.pt"
CHECKPOINT_SHA256 = "742d02b34dacbde82c983b46c742847d761478104a4e33a790470988ea724289"
PALETTE = ((186, 161, 127), (255, 158, 72), (61, 166, 255), (193, 200, 218), (174, 224, 91), (30, 178, 124))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_preview_model(device: str):
    # Verify the fixed local artifact before deserializing its optimizer/RNG payload.
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Classifier checkpoint integrity check failed; preview is disabled.")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if payload.get("model_type") != RgbSegformer.model_type or payload.get("input_contract") != "raw_rgb_01_imagenet_normalized_in_model":
        raise ValueError("Unsupported classifier input contract")
    model = RgbSegformer.from_architecture(payload["architecture"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def tile_starts(size: int, tile: int, overlap: int) -> list[int]:
    if size <= 0 or tile <= 0 or not 0 <= overlap < tile:
        raise ValueError("Invalid tiling dimensions")
    return sorted(set([*range(0, max(1, size - tile + 1), tile - overlap), max(0, size - tile)]))


@torch.inference_mode()
def predict_classes(rgb, valid, model, device="cpu", *, tile=1024, overlap=128):
    """Native-resolution tiles, averaged probabilities in overlaps; no height inputs."""
    if rgb.ndim != 3 or rgb.shape[0] != 3 or valid.shape != rgb.shape[1:]:
        raise ValueError("RGB and image-validity grids must match")
    if not valid.any():
        raise ValueError("Image contains no valid RGB pixels")
    clean = np.where(valid[None], rgb, 0).astype(np.float32)
    if not np.isfinite(clean).all() or clean.min() < 0 or clean.max() > 255:
        raise ValueError("Preview requires RGB intensities in the 0â€“255 range")
    h, w = valid.shape
    scores = np.zeros((6, h, w), np.float32)
    counts = np.zeros((h, w), np.float32)
    target = torch.device(device)
    for y in tile_starts(h, tile, overlap):
        for x in tile_starts(w, tile, overlap):
            image = torch.from_numpy(clean[:, y:y+tile, x:x+tile].copy())[None].to(target) / 255
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if target.type == "cuda" else nullcontext()
            with amp:
                logits = model(image)["logits"]
            probability = logits.float().softmax(1)[0].cpu().numpy()
            if probability.shape != (6, *image.shape[-2:]) or not np.isfinite(probability).all():
                raise ValueError("Classifier returned invalid probabilities")
            th, tw = probability.shape[1:]
            scores[:, y:y+th, x:x+tw] += probability
            counts[y:y+th, x:x+tw] += 1
    labels = (scores / counts[None]).argmax(0).astype(np.uint8)
    labels[~valid] = 255
    return labels


def coverage_report(labels: np.ndarray, profile: dict) -> dict:
    valid = labels < 6
    count = int(valid.sum())
    pixel_area = None
    crs = profile.get("crs")
    transform = profile.get("transform")
    # Only a projected CRS with known linear units supports an area claim here.
    if crs is not None and transform is not None:
        crs = rasterio.crs.CRS.from_user_input(crs)
        if crs.is_projected:
            factor = crs.linear_units_factor[1]
            area = abs(transform.a * transform.e - transform.b * transform.d) * factor**2
            if np.isfinite(area) and area > 0:
                pixel_area = float(area)
    classes = []
    for index, name in enumerate(CLASS_NAMES):
        pixels = int((labels == index).sum())
        classes.append({"id": index, "name": name, "color": list(PALETTE[index]), "pixels": pixels,
                        "coverage_percent": pixels / count * 100 if count else 0,
                        "area_m2": pixels * pixel_area if pixel_area is not None else None})
    return {"classes": classes, "valid_pixels": count, "ignored_pixels": int(labels.size-count),
            "pixel_area_m2": pixel_area, "area_basis": "projected map area" if pixel_area is not None else "unknown; pixel coverage only"}


def run_preview(input_path: Path, output_dir: Path, device: str) -> dict:
    source = read_rgb_raster(input_path, max_pixels=3072**2, max_dimension=3072)
    if source.profile.get("dtype") != "uint8":
        raise ValueError("Experimental classifier currently requires 8-bit RGB. High-bit-depth height uploads still work; classification radiometry needs separate validation.")
    model = load_preview_model(device)
    try:
        labels = predict_classes(source.rgb, source.valid_mask, model, device)
    finally:
        del model
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = {key: value for key, value in source.profile.items() if key in {"crs", "transform"}}
    profile.update(driver="GTiff", count=1, dtype="uint8", width=labels.shape[1], height=labels.shape[0], nodata=255, compress="deflate")
    with rasterio.open(output_dir / "classes.tif", "w", **profile) as raster:
        raster.write(labels, 1)
        raster.update_tags(MSR_PRODUCT="experimental_six_class_identification",
                           MSR_CLASSES=json.dumps(dict(enumerate(CLASS_NAMES))),
                           MSR_CHECKPOINT_SHA256=CHECKPOINT_SHA256)
    Image.fromarray(labels).save(output_dir / "classes.png")
    (output_dir / "classes.bin").write_bytes(labels.tobytes())
    report = {"version": "rgb_multiregion_v3_epoch1_preview", "experimental": True,
              "checkpoint_sha256": CHECKPOINT_SHA256, "source_sha256": file_sha256(input_path),
              "width": labels.shape[1], "height": labels.shape[0], "resampled": source.resampled,
              "height_pipeline_changed": False, "input_contract": "raw_rgb_01_imagenet_normalized_in_model",
              "inference": "native tiles up to 1024 px; 128 px overlap; mean probabilities; no TTA",
              "release_status": "Development improvement; regression checks failed; not approved as default.",
              "warnings": ["Classes do not establish object heights or object counts.",
                           "No calibrated confidence probabilities are provided.",
                           "This model targets high-resolution aerial RGB; coarse satellite imagery is out of its validated domain."],
              **coverage_report(labels, source.profile)}
    (output_dir / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
