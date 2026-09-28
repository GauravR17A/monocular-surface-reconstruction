import pytest

from msr.data.open_canopy_holdout import (
    connected_components,
    plan_holdout,
    split_overlap_audit,
)


def _record(
    sample_id: str,
    split: str,
    region: str,
    image_name: str,
    image_date: str,
    lidar_date: str,
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "split": split,
        "region": region,
        "image_name": image_name,
        "imagery_acquisition_date": image_date,
        "lidar_acquisition_date": lidar_date,
    }


def test_source_and_region_constraints_are_transitive() -> None:
    records = [
        _record("a", "train", "r1", "scene-a", "2021-01-01", "2021-02-01"),
        _record("b", "train", "r2", "scene-a", "2021-01-01", "2021-02-02"),
        _record("c", "train", "r2", "scene-b", "2021-01-02", "2021-02-03"),
        _record("d", "train", "r3", "scene-c", "2021-01-03", "2021-02-04"),
    ]

    components = connected_components(records, ["region", "image_name"])

    assert components == [["a", "b", "c"], ["d"]]


def test_current_split_audit_exposes_shared_source_scene() -> None:
    records = [
        _record("a", "train", "r1", "scene-a", "2021-01-01", "2021-02-01"),
        _record("b", "val", "r2", "scene-a", "2021-01-01", "2021-02-02"),
    ]

    audit = split_overlap_audit(records, ["region", "image_name"])

    assert audit["passes"] is False
    assert audit["pairs"]["train__val"]["region"] == []
    assert audit["pairs"]["train__val"]["image_name"] == ["scene-a"]


def test_plan_is_deterministic_and_verified_disjoint() -> None:
    records = [
        _record(str(index), "train", f"r{index}", f"s{index}", f"2021-01-{index + 1:02d}", f"2021-02-{index + 1:02d}")
        for index in range(10)
    ]

    first = plan_holdout(records, ["region", "image_name"], holdout_fraction=0.2, seed="fixed")
    second = plan_holdout(records, ["region", "image_name"], holdout_fraction=0.2, seed="fixed")

    assert first == second
    assert first["holdout_samples"] == 2
    assert first["verification"]["passes"] is True


def test_missing_required_metadata_fails_closed() -> None:
    records = [{"sample_id": "a", "split": "train", "region": "r1"}]
    with pytest.raises(ValueError, match="missing required isolation field"):
        connected_components(records, ["image_name"])
