import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from msr.data.gamus_dataset import GAMUS_SIX_CLASS_NAMES
from msr.data.gamus_rgb_segmentation import (
    GamusRgbSegmentationDataset, build_train_sampler, canonical_sha256,
    file_sha256, source_metadata_snapshot,
)
from msr.data.gamus_rgb_targeted_v2 import (
    TARGET_INDEX_SCHEMA, TARGET_NAMES, GamusRgbTargetedCropDatasetV2,
    candidate_origins, read_rgb_class_window, target_candidate_eligibility,
    training_tile_statistics, window_counts,
)


def fixture(tmp_path, size=8, patch=4, minimum=2):
    index = {"schema": "msr.gamus.approved_samples.v1", "dataset": "GAMUS", "splits": {}}
    for split in ("train", "val"):
        ids = [f"DC_{split}", f"PHL_{split}"]
        index["splits"][split] = {"approved_count": 2, "approved_sample_ids": ids, "semantic_eligible_sample_ids": ids}
    # Deliberately create NO validation files: this path must be train-only.
    for sample_id in index["splits"]["train"]["approved_sample_ids"]:
        classes = np.full((size, size), 6, dtype=np.float32)
        classes[:size // 2, :size // 2] = 1  # ground
        classes[size // 2:, size // 2:] = 2  # low vegetation
        image = np.repeat(classes[..., None], 3, axis=2)
        image[0, 0] = np.nan
        image[0, 1] = 0  # undeclared black is still valid
        classes[0, 2] = 0
        for role, suffix, data in (("images", "RGB", image), ("classes", "CLS", classes)):
            path = tmp_path / role / "train" / f"{sample_id}_{suffix}.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("image", data=data)
    approved = tmp_path / "approved.json"
    approved.write_text(json.dumps(index))
    kwargs = {"root": tmp_path, "split": "train", "approved_index_path": approved,
        "approved_index_sha256": file_sha256(approved), "expected_split_counts": {"train": 2, "val": 2},
        "native_shape": size, "patch_size": patch, "random_crop": True, "augment": True}
    dataset = GamusRgbSegmentationDataset(**kwargs)
    origins = candidate_origins(size, patch, max(1, patch // 2))
    rows = []
    for record in dataset.records:
        arrays = read_rgb_class_window(record, native_shape=size, patch_size=size, row=0, col=0)
        rows.append({"sample_id": record.sample_id, "city": record.city,
            **training_tile_statistics(arrays, patch, origins, minimum)})
    payload = {"schema": TARGET_INDEX_SCHEMA, "split": "train", "approved_index_sha256": kwargs["approved_index_sha256"],
        "train_ids_sha256": canonical_sha256(list(dataset.sample_ids)), "native_shape": size, "patch_size": patch,
        "rgb_scale": 255., "class_names": list(GAMUS_SIX_CLASS_NAMES), "target_names": list(TARGET_NAMES),
        "source_metadata": source_metadata_snapshot(dataset), "min_target_pixels": minimum,
        "target_quantile": .75, "candidate_origins": origins, "tiles": rows}
    targeted_path = tmp_path / "targeted.json"
    targeted_path.write_text(json.dumps(payload))
    return kwargs, {"targeted_index_path": targeted_path, "targeted_index_sha256": file_sha256(targeted_path)}, payload


def rewrite(target_kwargs, payload):
    target_kwargs["targeted_index_path"].write_text(json.dumps(payload))
    target_kwargs["targeted_index_sha256"] = file_sha256(target_kwargs["targeted_index_path"])


def test_control_byte_matches_sealed_v1_and_never_reads_other_splits(tmp_path, monkeypatch):
    kwargs, target_kwargs, _ = fixture(tmp_path)
    base = GamusRgbSegmentationDataset(**kwargs)
    control = GamusRgbTargetedCropDatasetV2(**kwargs, **target_kwargs, target_probability=0)
    original, opened = h5py.File, []
    def train_only(path, *args, **kwargs):
        assert Path(path).parent.name == "train"
        assert Path(path).name.endswith(("_RGB.h5", "_CLS.h5"))
        opened.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(h5py, "File", train_only)
    torch.manual_seed(123)
    expected = [base[i % 2] for i in range(12)]
    expected_state = torch.get_rng_state().clone()
    torch.manual_seed(123)
    actual = [control[i % 2] for i in range(12)]
    assert torch.equal(torch.get_rng_state(), expected_state)
    for a, b in zip(actual, expected):
        assert a.keys() == b.keys()
        for key in a:
            assert torch.equal(a[key], b[key]) if isinstance(a[key], torch.Tensor) else a[key] == b[key]
    assert len(opened) == 48


def test_targeting_preserves_common_global_rng_and_geometry(tmp_path):
    kwargs, target_kwargs, _ = fixture(tmp_path)
    torch.manual_seed(891)
    control = GamusRgbTargetedCropDatasetV2(**kwargs, **target_kwargs, target_probability=0)
    for i in range(20):
        control[i % 2]
    state = torch.get_rng_state().clone()
    torch.manual_seed(891)
    targeted = GamusRgbTargetedCropDatasetV2(**kwargs, **target_kwargs, target_probability=.5)
    for i in range(20):
        targeted[i % 2]
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(control._target_generator.get_state(), targeted._target_generator.get_state())


def test_target_selection_uniform_and_support_is_joint_valid(tmp_path):
    kwargs, target_kwargs, payload = fixture(tmp_path)
    targeted = GamusRgbTargetedCropDatasetV2(**kwargs, **target_kwargs, target_probability=1)
    torch.manual_seed(234)
    target_counts = {name: 0 for name in TARGET_NAMES}
    for _ in range(4000):
        row, col, name = targeted.choose_crop(0)
        target_counts[name] += 1
        j = targeted.candidate_origins.index([row, col])
        assert j in payload["tiles"][0]["eligible_candidate_indices"][name]
        cid = GAMUS_SIX_CLASS_NAMES.index(name)
        assert payload["tiles"][0]["candidate_valid_class_counts"][j][cid] >= payload["tiles"][0]["target_minimum_valid_pixels"][name]
    assert 1800 < target_counts["ground"] < 2200
    arrays = read_rgb_class_window(targeted.records[0], native_shape=8, patch_size=8, row=0, col=0)
    assert arrays["classification_valid_mask"][0, 0] and not arrays["image_valid_mask"][0, 0]
    assert arrays["image_valid_mask"][0, 1] and arrays["dark_pixel_proxy_mask"][0, 1]
    assert arrays["image_valid_mask"][0, 2] and not arrays["classification_valid_mask"][0, 2]
    assert payload["tiles"][0]["joint_valid_pixels"] == 62


def test_empty_target_fallback_and_one_target_only(tmp_path):
    kwargs, target_kwargs, payload = fixture(tmp_path)
    for row in payload["tiles"]:
        counts = np.array(row["candidate_valid_class_counts"])
        counts[:, [0, 4]] = 0
        row["candidate_valid_class_counts"] = counts.tolist()
        row["target_minimum_valid_pixels"], row["eligible_candidate_indices"] = target_candidate_eligibility(counts, 2, .75)
    rewrite(target_kwargs, payload)
    targeted = GamusRgbTargetedCropDatasetV2(**kwargs, **target_kwargs, target_probability=1)
    assert all(targeted.choose_crop(0)[2] == "ordinary" for _ in range(20))
    for row in payload["tiles"]:
        counts = np.array(row["candidate_valid_class_counts"])
        counts[:] = 0
        counts[0, 4] = 8
        row["candidate_valid_class_counts"] = counts.tolist()
        row["target_minimum_valid_pixels"], row["eligible_candidate_indices"] = target_candidate_eligibility(counts, 2, .75)
    rewrite(target_kwargs, payload)
    targeted = GamusRgbTargetedCropDatasetV2(**kwargs, **target_kwargs, target_probability=1)
    assert all(targeted.choose_crop(0) == (0, 0, "low_vegetation") for _ in range(20))


@pytest.mark.parametrize("split", ["val", "test", "NYC", "Christchurch"])
def test_target_dataset_rejects_nontraining_before_open(tmp_path, split):
    with pytest.raises(ValueError, match="training-only"):
        GamusRgbTargetedCropDatasetV2(tmp_path, split, targeted_index_path="missing", targeted_index_sha256="bad")


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p.update(split="val"), "training-only"),
    (lambda p: p["tiles"][0].update(sample_id="NYC_01"), "training ID"),
    (lambda p: p["tiles"][0].update(city="PHL"), "city identity"),
    (lambda p: p["candidate_origins"].__setitem__(0, [-1, 0]), "crop origins"),
    (lambda p: p["tiles"][0]["candidate_valid_class_counts"][0].__setitem__(0, 17), "pixel counts"),
    (lambda p: p["tiles"][0]["eligible_candidate_indices"].update(ground=[]), "eligibility"),
    (lambda p: p["tiles"][0]["target_minimum_valid_pixels"].update(ground=1), "thresholds"),
    (lambda p: p.update(target_quantile=1.1), "quantile"),
])
def test_index_contract_fails_closed(tmp_path, mutation, match):
    kwargs, target_kwargs, payload = fixture(tmp_path)
    mutation(payload)
    rewrite(target_kwargs, payload)
    with pytest.raises(ValueError, match=match):
        GamusRgbTargetedCropDatasetV2(**kwargs, **target_kwargs)


def test_source_metadata_and_index_authentication_drift(tmp_path):
    kwargs, target_kwargs, _ = fixture(tmp_path)
    target_kwargs["targeted_index_path"].write_text(target_kwargs["targeted_index_path"].read_text() + " ")
    with pytest.raises(ValueError, match="SHA-256"):
        GamusRgbTargetedCropDatasetV2(**kwargs, **target_kwargs)
    target_kwargs["targeted_index_sha256"] = file_sha256(target_kwargs["targeted_index_path"])
    with h5py.File(tmp_path / "images/train/DC_train_RGB.h5", "r+") as handle:
        handle["image"][1, 1, 0] = 55
    with pytest.raises(ValueError, match="source metadata"):
        GamusRgbTargetedCropDatasetV2(**kwargs, **target_kwargs)


def test_exact_window_counts_and_exposure_match_brute_force(tmp_path):
    kwargs, _, payload = fixture(tmp_path)
    dataset = GamusRgbSegmentationDataset(**kwargs)
    arrays = read_rgb_class_window(dataset.records[0], native_shape=8, patch_size=8, row=0, col=0)
    mask = (arrays["labels"] == 0) & arrays["image_valid_mask"] & arrays["classification_valid_mask"]
    brute = np.array([[mask[r:r + 4, c:c + 4].sum() for c in range(5)] for r in range(5)])
    np.testing.assert_array_equal(window_counts(mask, 4), brute)
    ordinary = payload["tiles"][0]["ordinary_uniform_crop"]
    assert ordinary["expected_valid_class_pixels"]["ground"] == brute.mean()
    assert ordinary["meaningful_support_probability"]["ground"] == (brute >= 2).mean()
    path = Path(__file__).parents[1] / "scripts/audit_rgb_training_crops_v2.py"
    spec = importlib.util.spec_from_file_location("audit_rgb_training_crops_v2_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.expected_crop_exposure(payload["tiles"], np.ones(2), 4, 2)
    assert report["DC"]["arms"]["control"]["expected_valid_class_pixels"]["ground"] == brute.mean()
    assert report["DC"]["city_probability"] == .5
    assert report["DC"]["effective_targeted_crop_probability"] == .5


def test_image_sampler_weights_order_and_rng_identical_between_arms(tmp_path):
    kwargs, target_kwargs, _ = fixture(tmp_path, size=384, patch=384)
    control = GamusRgbTargetedCropDatasetV2(**kwargs, **target_kwargs, target_probability=0)
    targeted = GamusRgbTargetedCropDatasetV2(**kwargs, **target_kwargs, target_probability=.5)
    fine = tmp_path / "fine.jsonl"
    fine.write_text("\n".join(json.dumps({"schema": "msr.gamus.train_six_class_tile_index.v1",
        "sample_id": record.sample_id, "city": record.city,
        "uniform_random_crop_384": {"water_hit_probability": float(i)}}) for i, record in enumerate(control.records)))
    config = {"data": {"fine_class_sampling_index_path": fine, "fine_class_sampling_index_sha256": file_sha256(fine)},
        "training": {"samples_per_epoch": 100}}
    a = build_train_sampler(control, config, torch.Generator().manual_seed(7))
    b = build_train_sampler(targeted, config, torch.Generator().manual_seed(7))
    assert a.weights.tolist() == b.weights.tolist() == [1., 1.]
    assert list(a) == list(b)
