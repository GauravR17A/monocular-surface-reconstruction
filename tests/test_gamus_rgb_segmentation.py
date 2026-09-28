import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from msr.data.gamus_rgb_segmentation import (
    GamusRgbSegmentationDataset, authenticate_validation_content,
    build_train_sampler, canonical_sha256, file_sha256,
)


def _fixture(tmp_path, size=4):
    split_ids = {"train": ["DC_train", "PHL_train"], "val": ["DC_val", "PHL_val"]}
    index = {"schema": "msr.gamus.approved_samples.v1", "dataset": "GAMUS", "splits": {}}
    for split, ids in split_ids.items():
        index["splits"][split] = {"approved_count": len(ids), "approved_sample_ids": ids,
                                    "semantic_eligible_sample_ids": ids}
        for sample_id in ids:
            for role, suffix, value in (("images", "RGB", np.full((size, size, 3), 51, dtype=np.float32)),
                                        ("classes", "CLS", (np.arange(size*size).reshape(size, size) % 7).astype(np.float32))):
                path = tmp_path / role / split / f"{sample_id}_{suffix}.h5"
                path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("image", data=value)
    path = tmp_path / "approved.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    return {"root": tmp_path, "approved_index_path": path,
            "approved_index_sha256": file_sha256(path), "expected_split_counts": {"train": 2, "val": 2},
            "native_shape": size, "patch_size": size}


def test_rgb_only_reads_and_six_class_mapping(tmp_path, monkeypatch):
    kwargs = _fixture(tmp_path)
    original = h5py.File
    opened = []
    def recording(path, *args, **kw):
        opened.append(str(path))
        assert "AGL" not in str(path) and "REL" not in str(path)
        return original(path, *args, **kw)
    monkeypatch.setattr(h5py, "File", recording)
    sample = GamusRgbSegmentationDataset(split="val", **kwargs)[0]
    assert sample["image"].shape == (3, 4, 4)
    assert torch.allclose(sample["image"], torch.full((3, 4, 4), .2))
    assert sample["labels"].flatten()[:7].tolist() == [255, 0, 4, 1, 2, 3, 5]
    assert len(opened) == 2
    assert not any("height" in key or "prior" in key for key in sample)


def test_image_and_class_validity_independent_and_no_black_mask(tmp_path):
    kwargs = _fixture(tmp_path)
    with h5py.File(tmp_path / "images/val/DC_val_RGB.h5", "r+") as handle:
        handle["image"][0, 1] = np.nan
        handle["image"][0, 2] = 0  # black RGB is valid unless declared nodata
    sample = GamusRgbSegmentationDataset(split="val", **kwargs)[0]
    assert sample["classification_valid_mask"][0, 1]
    assert not sample["image_valid_mask"][0, 1]
    assert sample["image_valid_mask"][0, 0] and not sample["classification_valid_mask"][0, 0]
    assert sample["image_valid_mask"][0, 2] and sample["dark_pixel_proxy_mask"][0, 2]
    assert torch.isfinite(sample["image"]).all()


@pytest.mark.parametrize("value", [7, -1, 1.5, 255])
def test_unexpected_class_ids_fail_closed(tmp_path, value):
    kwargs = _fixture(tmp_path)
    with h5py.File(tmp_path / "classes/val/DC_val_CLS.h5", "r+") as handle:
        handle["image"][0, 0] = value
    with pytest.raises(ValueError, match="Unexpected source class"):
        GamusRgbSegmentationDataset(split="val", **kwargs)[0]


def test_native_validation_and_forbidden_splits(tmp_path):
    kwargs = _fixture(tmp_path)
    with pytest.raises(ValueError, match="Only approved"):
        GamusRgbSegmentationDataset(split="test", **kwargs)
    with pytest.raises(ValueError, match="Validation must"):
        GamusRgbSegmentationDataset(split="val", **{**kwargs, "patch_size": 2})
    with pytest.raises(ValueError, match="Validation must"):
        GamusRgbSegmentationDataset(split="val", augment=True, **kwargs)
    with h5py.File(tmp_path / "classes/val/DC_val_CLS.h5", "r+") as handle:
        del handle["image"]
        handle.create_dataset("image", data=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="Unaligned"):
        GamusRgbSegmentationDataset(split="val", **kwargs)[0]


@pytest.mark.parametrize("sample_id", ["NYC_01", "DC_../../test/IMG", "DC_01.h5"])
def test_unapproved_geography_or_traversal_never_discovered(tmp_path, sample_id):
    kwargs = _fixture(tmp_path)
    path = kwargs["approved_index_path"]
    index = json.loads(path.read_text())
    index["splits"]["train"]["approved_sample_ids"][0] = sample_id
    path.write_text(json.dumps(index))
    kwargs["approved_index_sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="safe DC/PHL"):
        GamusRgbSegmentationDataset(split="val", **kwargs)


def test_augmentation_keeps_rgb_label_and_masks_aligned(tmp_path):
    kwargs = _fixture(tmp_path)
    with h5py.File(tmp_path / "images/train/DC_train_RGB.h5", "r+") as rgb, h5py.File(tmp_path / "classes/train/DC_train_CLS.h5", "r+") as cls:
        raw = cls["image"][:]
        rgb["image"][:] = np.repeat(raw[..., None], 3, axis=2)
    torch.manual_seed(9)
    sample = GamusRgbSegmentationDataset(split="train", augment=True, **kwargs)[0]
    raw = np.rint(sample["image"][0].numpy() * 255).astype(int)
    mapping = np.array([255, 0, 4, 1, 2, 3, 5])
    np.testing.assert_array_equal(mapping[raw], sample["labels"].numpy())


def test_water_sampler_preserves_each_city_mass(tmp_path):
    kwargs = _fixture(tmp_path, size=384)
    dataset = GamusRgbSegmentationDataset(split="train", **kwargs)
    path = tmp_path / "sampling.jsonl"
    rows = [{"schema": "msr.gamus.train_six_class_tile_index.v1", "sample_id": r.sample_id,
             "city": r.city, "uniform_random_crop_384": {"water_hit_probability": float(i)}}
            for i, r in enumerate(dataset.records)]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    config = {"data": {"fine_class_sampling_index_path": str(path), "fine_class_sampling_index_sha256": file_sha256(path),
                       "water_sampling_boost": 2, "preserve_city_sampling_mass": True}}
    sampler = build_train_sampler(dataset, config, torch.Generator().manual_seed(1))
    assert sampler.weights.tolist() == [1., 1.]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="SHA-256"):
        build_train_sampler(dataset, config, torch.Generator())


def test_validation_content_binding_and_cache_detect_source_drift(tmp_path):
    kwargs = _fixture(tmp_path)
    dataset = GamusRgbSegmentationDataset(split="val", **kwargs)
    roles = {}
    for role in ("class", "image"):
        rows = []
        for record in dataset.records:
            path = record.image_path if role == "image" else record.class_path
            rows.append({"role": role, "sample_id": record.sample_id, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)})
        roles[role] = {"file_count": len(rows), "total_bytes": sum(r["size_bytes"] for r in rows), "content_manifest_sha256": canonical_sha256(rows)}
    report = {"schema": "msr.gamus_dcphl_paired_independent_replay.v1", "passes": True,
              "approved_index": {"sha256": kwargs["approved_index_sha256"]},
              "validation_contract": {"validation_ids_sha256": canonical_sha256(list(dataset.sample_ids))},
              "source_content_binding": {"per_role": roles}}
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(report))
    cache = tmp_path / "hash_cache.json"
    first = authenticate_validation_content(dataset, path, file_sha256(path), cache)
    second = authenticate_validation_content(dataset, path, file_sha256(path), cache)
    assert first["newly_hashed_file_count"] == 4 and second["newly_hashed_file_count"] == 0
    assert not second["height_and_prior_files_read"]
    with h5py.File(dataset.records[0].image_path, "r+") as handle:
        handle["image"][0, 0, 0] = 52
    with pytest.raises(ValueError, match="content differs"):
        authenticate_validation_content(dataset, path, file_sha256(path), cache)
