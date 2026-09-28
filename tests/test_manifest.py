from pathlib import Path

import pytest

from msr.data.manifest import SampleRecord, assert_disjoint_regions, load_manifest


def _record(sample_id: str, region: str) -> SampleRecord:
    placeholder = Path("placeholder.tif")
    return SampleRecord(sample_id, region, placeholder, placeholder)


def test_manifest_resolves_relative_paths(tmp_path):
    (tmp_path / "rgb.tif").touch()
    (tmp_path / "height.tif").touch()
    manifest = tmp_path / "samples.csv"
    manifest.write_text(
        "sample_id,region,rgb_path,height_path,mask_path,gsd_m\n"
        "one,city_a,rgb.tif,height.tif,,0.5\n",
        encoding="utf-8",
    )

    records = load_manifest(manifest)

    assert records[0].rgb_path == (tmp_path / "rgb.tif").resolve()
    assert records[0].gsd_m == 0.5


def test_geographic_leakage_is_rejected():
    train = [_record("a", "city_a")]
    validation = [_record("b", "city_a")]

    with pytest.raises(ValueError, match="Geographic leakage"):
        assert_disjoint_regions(train, validation)

