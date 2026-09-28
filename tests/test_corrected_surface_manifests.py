from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "data"
    / "version_corrected_surface_manifests.py"
)
SPEC = importlib.util.spec_from_file_location("corrected_surface_manifests_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


FIELDS = [
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


def _write_split(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(sample_id: str, *, urban: bool) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "region": sample_id,
        "landscape": "urban" if urban else "forest",
        "rgb_path": f"{sample_id}/rgb.tif",
        "surface_path": (
            f"{sample_id}/building_height_m.tif"
            if urban
            else f"{sample_id}/canopy_height_m.tif"
        ),
        "target_kind": "ndsm",
        "dtm_path": "",
        "building_mask_path": "",
        "vegetation_mask_path": "" if urban else f"{sample_id}/vegetation.tif",
        "valid_mask_path": "" if urban else f"{sample_id}/valid.tif",
        "relative_prior_path": "",
        "gsd_m": "",
    }


def test_versions_highbuild_as_building_only_and_preserves_forest(tmp_path: Path) -> None:
    source = tmp_path / "old"
    source.mkdir()
    for index, split in enumerate(MODULE.SPLITS):
        _write_split(
            source / f"{split}.csv",
            [_row(f"urban-{index}", urban=True), _row(f"forest-{index}", urban=False)],
        )
    destination = tmp_path / "corrected"
    summary = MODULE.build_versioned_manifests(source, destination)

    rows = list(csv.DictReader((destination / "validation.csv").open(encoding="utf-8")))
    assert rows[0]["target_kind"] == "building_height"
    assert rows[0]["building_mask_path"] == rows[0]["surface_path"]
    assert rows[1]["target_kind"] == "ndsm"
    assert summary["splits"]["validation"]["changed_rows"] == 1
    assert summary["raster_content_opened"] is False
    assert summary["official_test_content_opened"] is False


def test_refuses_to_overwrite_versioned_output(tmp_path: Path) -> None:
    source = tmp_path / "old"
    source.mkdir()
    for index, split in enumerate(MODULE.SPLITS):
        _write_split(source / f"{split}.csv", [_row(f"urban-{index}", urban=True)])
    destination = tmp_path / "corrected"
    MODULE.build_versioned_manifests(source, destination)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.build_versioned_manifests(source, destination)
