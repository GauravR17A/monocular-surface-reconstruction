"""CPU-only development error audit; no model, training, or external data reads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = ROOT / "outputs/evaluation/gamus_rgb_segmenter_v1_independent_replay"
CLASSES = ("ground", "buildings", "water", "roads", "low_vegetation", "trees")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def summarize(rows):
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Require nonempty unique development records")
    matrix = np.zeros((6, 6), dtype=np.int64)
    for row in rows:
        current = np.asarray(row["six_class_identification"]["confusion_matrix"])
        if current.shape != (6, 6) or current.dtype.kind not in "iu" or np.any(current < 0):
            raise ValueError("Malformed confusion counts")
        matrix += current
    support = int(matrix[4].sum())
    predicted = int(matrix[:, 4].sum())
    tp = int(matrix[4, 4])
    per_image = []
    for row in rows:
        counts = np.asarray(row["six_class_identification"]["confusion_matrix"])
        n, p, hit = int(counts[4].sum()), int(counts[:, 4].sum()), int(counts[4, 4])
        per_image.append({"sample_id": row["sample_id"], "reference_pixels": n,
            "predicted_pixels": p, "missed_pixels": n-hit,
            "f1": 2*hit/(n+p) if n+p else None, "recall": hit/n if n else None})
    return {"confusion_matrix": matrix.tolist(), "image_count": len(rows), "reference_low_vegetation_pixels": support,
        "total_reference_pixels": int(matrix.sum()),
        "reference_low_vegetation_share": support/int(matrix.sum()) if matrix.sum() else None,
        "f1": 2*tp/(support+predicted) if support+predicted else None,
        "precision": tp/predicted if predicted else None,
        "recall": tp/support if support else None,
        "reference_low_vegetation_predicted_as": {
            name: {"pixels": int(matrix[4, i]), "fraction": int(matrix[4, i])/support if support else None}
            for i, name in enumerate(CLASSES)},
        "highest_missed_pixel_cases": sorted(per_image, key=lambda r: (-r["missed_pixels"], r["sample_id"]))[:20],
        "per_image": per_image}


def structure(reference, prediction):
    """Thin/interior breakdown of preselected examples; not a city-wide estimate."""
    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError("Reference and prediction must be aligned 2D arrays")
    if not all(np.isin(a, [0, 1, 2, 3, 4, 5, 255]).all() for a in (reference, prediction)):
        raise ValueError("Unexpected class indices")
    low = reference == 4
    total = int(low.sum())
    result = {}
    for radius in (1, 2, 4, 8):
        interior = binary_erosion(low, structure=np.ones((3, 3), bool), iterations=radius, border_value=0)
        edge = low & ~interior
        n = int(interior.sum())
        result[str(radius)] = {"reference_interior_pixels": n,
            "reference_edge_fraction": int(edge.sum())/total if total else None,
            "interior_recall": int(((prediction == 4) & interior).sum())/n if n else None,
            "edge_recall": int(((prediction == 4) & edge).sum())/int(edge.sum()) if edge.any() else None}
    return {"reference_low_vegetation_pixels": total, "erosion_radius_pixels": result,
        "interpretation": "Chebyshev pixel erosion, not metres; six fixed development previews only."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs/diagnostics") or output.exists():
        raise ValueError("Use a fresh outputs/diagnostics directory")
    replay = DEFAULT_REPLAY
    report = json.loads((replay / "report.json").read_text(encoding="utf-8"))
    if not report.get("replay_verified") or not report["saved_epoch_comparison"]["counts_match"]:
        raise ValueError("Require independently verified replay")
    rows = [json.loads(line) for line in (replay / "per_image_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 859 or any(r["city"] not in {"DC", "PHL"} for r in rows):
        raise ValueError("Only the complete fixed development replay is permitted")
    result = {"schema": "msr.rgb_vegetation_development_audit.v1",
        "checkpoint_sha256": report["checkpoint"]["sha256"],
        "source_report_sha256": digest(replay / "report.json"),
        "source_per_image_sha256": digest(replay / "per_image_metrics.jsonl"),
        "scope": "Development diagnosis only; no external evaluation or model edits",
        "by_city": {}, "fixed_preview_structure": []}
    for city in ("DC", "PHL"):
        subset = [r for r in rows if r["city"] == city]
        result["by_city"][city] = summarize(subset)
        expected = report["evaluation"]["by_city"][city]["six_class_identification"]["per_class"]["low_vegetation"]
        if result["by_city"][city]["confusion_matrix"] != report["evaluation"]["by_city"][city]["six_class_identification"]["confusion_matrix"]:
            raise ValueError("Per-image confusion counts do not match replay")
        if abs(result["by_city"][city]["f1"]-expected["f1"]) > 1e-12:
            raise ValueError("Per-image counts disagree with authenticated pooled report")
    manifest = json.loads((replay / "preview_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["items"]:
        relative = Path(item["reference"]).parent
        directory = (replay / relative).resolve()
        if not directory.is_relative_to(replay.resolve()):
            raise ValueError("Preview escaped source directory")
        ref = np.load(directory / "reference.npy", allow_pickle=False)
        pred = np.load(directory / "prediction.npy", allow_pickle=False)
        for key, array in (("reference", ref), ("prediction", pred)):
            png_path = (replay / item[key]).resolve()
            if not png_path.is_relative_to(replay.resolve()) or digest(png_path) != item["sha256"][key]:
                raise ValueError("Preview PNG identity mismatch")
            with Image.open(png_path) as png:
                if not np.array_equal(array, np.asarray(png)):
                    raise ValueError("Preview raw labels disagree with sealed PNG")
        result["fixed_preview_structure"].append({"sample_id": item["sample_id"], "city": item["city"], **structure(ref, pred)})
    result["limitations"] = ["Boundary concentration is not proof of annotation error.",
        "Top missed-pixel cases are explicitly error-selected, never representative accuracy examples.",
        "Do not use external holdouts to choose fixes; production height pipeline unchanged."]
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({city: {k: result["by_city"][city][k] for k in ("image_count", "f1", "recall", "reference_low_vegetation_share")} for city in ("DC", "PHL")}))


if __name__ == "__main__":
    main()
