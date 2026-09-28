"""Prepare positive-only roof-plane/section pilot masks; no training or height changes."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio

from msr.data.roof3d import roof_instance_targets

ROOT = Path("D:/MSRData/data/roof3d_training_pilot_v1")


def main():
    output = ROOT / "prepared_positive_only_v1"
    output.mkdir(exist_ok=True)
    report = {"schema": "msr.roof3d_positive_only_pilot.v1", "training_started": False,
              "height_supervision_enabled": False, "background_policy": "unknown, not a supervised negative",
              "geographic_split_ready": False, "reason": "Released numeric chip names lack source location/type metadata; resolve before long training.", "kinds": {}}
    for kind in ("plane", "sec"):
        labels = json.loads((ROOT / "train" / f"pilot_annotation_{kind}.json").read_text())
        by_image = defaultdict(list)
        for ann in labels["annotations"]:
            by_image[ann["image_id"]].append(ann)
        rows = []
        for im in labels["images"]:
            name = Path(im["file_name"]).name
            with rasterio.open(ROOT / "train/rgb" / name) as rgb:
                if (rgb.height, rgb.width) != (im["height"], im["width"]) or rgb.count != 3:
                    raise ValueError("RGB/annotation dimensions mismatch")
                valid = np.all(rgb.read_masks() > 0, axis=0)
                source_grid = {"crs": str(rgb.crs), "transform": list(rgb.transform)}
                shape = (rgb.height, rgb.width)
            with rasterio.open(ROOT / "train/dsm" / name) as dsm:
                dsm_grid = {"crs": str(dsm.crs), "transform": list(dsm.transform)}
                matching_grid = dsm_grid == source_grid and (dsm.height, dsm.width) == shape
            targets = roof_instance_targets(by_image[im["id"]], valid)
            destination = output / f"{kind}_{Path(name).stem}.npz"
            np.savez_compressed(destination, **targets)
            rows.append({"image_id": im["id"], "rgb": str(ROOT / "train/rgb" / name), "targets": str(destination),
                         "annotations": len(by_image[im["id"]]), "classification_valid_pixels": int(targets["classification_valid"].sum()),
                         "boundary_valid_pixels": int(targets["boundary_valid"].sum()), "boundary_pixels": int(targets["boundary"].sum()),
                         "overlap_pixels": int(targets["ambiguous_overlap"].sum()), "image_valid_pixels": int(valid.sum()),
                         "rgb_dsm_exact_grid_match": matching_grid, "rgb_grid": source_grid, "dsm_grid": dsm_grid})
        report["kinds"][kind] = rows
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prepared": str(output), "rows": {k:len(v) for k,v in report["kinds"].items()}, "training_started": False}))


if __name__ == "__main__":
    main()
