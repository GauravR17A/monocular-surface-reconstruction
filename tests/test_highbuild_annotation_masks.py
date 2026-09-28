from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import rasterio
from affine import Affine


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "data"
    / "build_highbuild_annotation_masks.py"
)
SPEC = importlib.util.spec_from_file_location("highbuild_annotation_masks_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _annotation(identifier: int, polygon: list[float], *, height: float, estimated: bool):
    return {
        "id": identifier,
        "image_id": 1,
        "category_id": 1,
        "segmentation": [polygon],
        "iscrowd": 0,
        "attributes": {
            "height": height,
            "is_estimated_height": estimated,
        },
    }


def test_annotation_masks_keep_semantics_but_exclude_estimated_height() -> None:
    coco = {
        "images": [{"id": 1, "width": 8, "height": 8}],
        "annotations": [
            _annotation(1, [1, 1, 4, 1, 4, 4, 1, 4], height=1.5, estimated=False),
            _annotation(2, [4, 4, 7, 4, 7, 7, 4, 7], height=9.0, estimated=True),
        ],
        "categories": [{"id": 1, "name": "building"}],
    }
    all_mask, trusted, counts = MODULE.annotation_masks(coco, height=8, width=8)

    assert all_mask.dtype == np.uint8
    assert trusted.dtype == np.uint8
    assert int(all_mask.sum()) > int(trusted.sum()) > 0
    assert counts == {
        "annotations": 2,
        "estimated_annotations": 1,
        "rejected_nonpositive_or_missing_height_annotations": 0,
        "trusted_annotations": 1,
    }


def test_annotation_mask_rejects_wrong_image_shape() -> None:
    coco = {"images": [{"id": 1, "width": 9, "height": 8}], "annotations": []}
    try:
        MODULE.annotation_masks(coco, height=8, width=8)
    except ValueError as error:
        assert "dimensions differ" in str(error)
    else:
        raise AssertionError("shape mismatch must fail closed")


def test_named_height_protocols_keep_strict_and_inclusive_support_separate() -> None:
    coco = {
        "images": [{"id": 1, "width": 8, "height": 8}],
        "annotations": [
            _annotation(1, [1, 1, 4, 1, 4, 4, 1, 4], height=4.0, estimated=False),
            _annotation(2, [4, 4, 7, 4, 7, 7, 4, 7], height=9.0, estimated=True),
        ],
    }
    all_mask, measured_mask, _ = MODULE.annotation_masks(coco, height=8, width=8)
    target = np.full((8, 8), 10.0, dtype=np.float32)
    strict = MODULE.regression_support_mask(
        all_footprints=all_mask,
        measured_footprints=measured_mask,
        target=target,
        nodata=None,
        protocol="strict_measured",
    )
    inclusive = MODULE.regression_support_mask(
        all_footprints=all_mask,
        measured_footprints=measured_mask,
        target=target,
        nodata=None,
        protocol="inclusive_annotated",
    )

    assert int(strict.sum()) == int(measured_mask.sum())
    assert int(inclusive.sum()) == int(all_mask.sum())
    assert int(inclusive.sum()) > int(strict.sum())


def test_read_csv_accepts_explicit_provenance_key(tmp_path: Path) -> None:
    provenance = tmp_path / "train.csv"
    provenance.write_text(
        "webdataset_key,webdataset_shard,webdataset_json_member\n"
        "city_tile,shard.tar,city_tile.json\n",
        encoding="utf-8",
    )

    fields, rows = MODULE._read_csv(
        provenance, required_field="webdataset_key"
    )

    assert fields[0] == "webdataset_key"
    assert rows == [
        {
            "webdataset_key": "city_tile",
            "webdataset_shard": "shard.tar",
            "webdataset_json_member": "city_tile.json",
        }
    ]


def _write_surface(path: Path) -> None:
    values = np.zeros((8, 8), dtype=np.float32)
    values[1:4, 1:4] = 4.0
    values[4:7, 4:7] = 9.0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=Affine.translation(500000, 3000000) * Affine.scale(1, -1),
    ) as destination:
        destination.write(values, 1)


def _write_contract_inputs(tmp_path: Path) -> tuple[Path, Path]:
    manifests = tmp_path / "manifests"
    split_root = tmp_path / "splits"
    manifests.mkdir()
    (split_root / "shards").mkdir(parents=True)
    fields = [
        "sample_id",
        "region",
        "landscape",
        "rgb_path",
        "surface_path",
        "target_kind",
        "dtm_path",
        "building_mask_path",
        "vegetation_mask_path",
        "valid_mask_path",
        "relative_prior_path",
        "gsd_m",
    ]
    coco = {
        "images": [{"id": 1, "width": 8, "height": 8}],
        "annotations": [
            _annotation(1, [1, 1, 4, 1, 4, 4, 1, 4], height=4.0, estimated=False),
            _annotation(2, [4, 4, 7, 4, 7, 7, 4, 7], height=9.0, estimated=True),
        ],
    }
    annotation_bytes = json.dumps(coco).encode("utf-8")
    for split in MODULE.SPLITS:
        sample_id = f"{split}-tile"
        sample_dir = tmp_path / split
        sample_dir.mkdir()
        surface = sample_dir / "building_height_m.tif"
        _write_surface(surface)
        with (manifests / f"{split}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "region": f"region-{split}",
                    "landscape": "urban",
                    "rgb_path": str(sample_dir / "rgb.jpg"),
                    "surface_path": str(surface),
                    "target_kind": "ndsm",
                    "dtm_path": "",
                    "building_mask_path": "",
                    "vegetation_mask_path": "",
                    "valid_mask_path": "",
                    "relative_prior_path": "",
                    "gsd_m": "",
                }
            )
        shard_relative = Path("shards") / f"{split}.tar"
        with tarfile.open(split_root / shard_relative, "w") as archive:
            info = tarfile.TarInfo(f"{sample_id}.json")
            info.size = len(annotation_bytes)
            archive.addfile(info, io.BytesIO(annotation_bytes))
        with (split_root / f"{split}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "webdataset_key",
                    "msr_shard",
                    "webdataset_json_member",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "webdataset_key": sample_id,
                    "msr_shard": str(shard_relative),
                    "webdataset_json_member": f"{sample_id}.json",
                }
            )
    return manifests, split_root


def test_versioned_builder_reproduces_strict_and_inclusive_protocols(
    tmp_path: Path,
) -> None:
    manifests, split_root = _write_contract_inputs(tmp_path)
    strict_root = tmp_path / "strict"
    inclusive_root = tmp_path / "inclusive"

    strict_report = MODULE.build_contract(
        source_manifest_dir=manifests,
        split_root=split_root,
        output_root=strict_root,
        height_protocol="strict_measured",
    )
    inclusive_report = MODULE.build_contract(
        source_manifest_dir=manifests,
        split_root=split_root,
        output_root=inclusive_root,
        height_protocol="inclusive_annotated",
    )

    assert strict_report["height_protocol"] == "strict_measured"
    assert inclusive_report["height_protocol"] == "inclusive_annotated"
    assert strict_report["splits"]["validation"]["regression_height_pixels"] < (
        inclusive_report["splits"]["validation"]["regression_height_pixels"]
    )
    for root, validity_name in (
        (strict_root, "measured_height_valid.tif"),
        (inclusive_root, "annotated_height_valid.tif"),
    ):
        row = next(csv.DictReader((root / "manifests" / "train.csv").open()))
        assert row["target_kind"] == "building_height"
        assert Path(row["building_mask_path"]).name == "building_footprints.tif"
        assert Path(row["valid_mask_path"]).name == validity_name
        assert row["building_mask_path"] != row["valid_mask_path"]
