from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import msr.evaluation.external_height_holdout as holdout


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY_ROOT / "configs" / "external_metric_height_nz_bop_v1.json"


def test_repository_external_height_contract_is_valid_and_sealed() -> None:
    contract = holdout.load_contract(CONTRACT_PATH)
    digest = holdout.verify_contract_seal(CONTRACT_PATH)

    assert contract["benchmark_id"] == "nz_bop_2018_2019_bc36_2314_v1"
    assert digest == "ddf68cdf382366450d29ed433ddb07a737c2fe07581e03b7c32e290a68eb456c"
    assert contract["policy"]["training_permitted"] is False
    assert contract["policy"]["model_selection_permitted"] is False
    assert contract["height_reference"]["scoring_supports"]["object_only"]["threshold_m"] == 2.0
    assert contract["height_reference"]["scoring_supports"]["tall_object"]["threshold_m"] == 5.0


@pytest.mark.parametrize(
    ("path", "unsafe_value"),
    [
        (("policy", "training_permitted"), True),
        (("policy", "model_selection_permitted"), True),
        (("selection_rule", "visual_selection_permitted"), True),
        (("height_reference", "upper_height_cutoff_m"), 200.0),
        (("height_reference", "negative_policy"), "clip"),
    ],
)
def test_contract_rejects_leakage_or_posthoc_filtering(
    path: tuple[str, str], unsafe_value: object
) -> None:
    contract = holdout.load_contract(CONTRACT_PATH)
    mutated = deepcopy(contract)
    mutated[path[0]][path[1]] = unsafe_value

    with pytest.raises(holdout.ExternalHoldoutError):
        holdout.validate_contract(mutated)


def test_contract_rejects_nonofficial_raster_host() -> None:
    contract = holdout.load_contract(CONTRACT_PATH)
    mutated = deepcopy(contract)
    next(source for source in mutated["sources"] if source["role"] == "rgb")[
        "url"
    ] = "https://example.com/untrusted.tif"

    with pytest.raises(holdout.ExternalHoldoutError, match="official LINZ bucket"):
        holdout.validate_contract(mutated)


def test_data_root_must_be_below_explicit_allowed_root(tmp_path: Path) -> None:
    contract = holdout.load_contract(CONTRACT_PATH)
    contract["data_root"] = str(tmp_path.parent / "outside")

    with pytest.raises(holdout.ExternalHoldoutError, match="must be a child"):
        holdout.resolve_data_root(contract, allowed_root=tmp_path)


def _write_raster(
    path: Path,
    values: np.ndarray,
    *,
    transform: rasterio.Affine,
    nodata: float | None = None,
) -> None:
    bands = values if values.ndim == 3 else values[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=bands.shape[-1],
        height=bands.shape[-2],
        count=bands.shape[0],
        dtype=str(bands.dtype),
        crs="EPSG:2193",
        transform=transform,
        nodata=nodata,
    ) as target:
        target.write(bands)


def _synthetic_contract(tmp_path: Path) -> dict[str, object]:
    contract = holdout.load_contract(CONTRACT_PATH)
    contract["benchmark_id"] = "nz_synthetic_holdout_v1"
    contract["data_root"] = str(tmp_path / "holdout")
    contract["grid"].update(
        {
            "prepared_width": 4,
            "prepared_height": 3,
            "prepared_transform_gdal": [100.0, 1.0, 0.0, 203.0, 0.0, -1.0],
            "prepared_bounds": [100.0, 200.0, 104.0, 203.0],
        }
    )
    holdout.validate_contract(contract)
    return contract


def test_prepare_package_seals_aligned_masks_without_model_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _synthetic_contract(tmp_path)
    root = Path(contract["data_root"])
    raw = root / "raw"
    raw.mkdir(parents=True)

    rgb = np.full((3, 30, 40), 120, dtype=np.uint8)
    rgb_path = raw / "rgb.tif"
    _write_raster(rgb_path, rgb, transform=from_origin(100.0, 203.0, 0.1, 0.1))

    dem_values = np.full((7, 8), 50.0, dtype=np.float32)
    dsm_values = dem_values.copy()
    # The fixed 4x3 crop begins at row 2, column 2.
    crop = np.asarray(
        [
            [50.0, 52.1, 56.0, 49.5],
            [50.0, 53.0, 55.0, 57.0],
            [50.0, 50.0, 50.0, 50.0],
        ],
        dtype=np.float32,
    )
    dsm_values[2:5, 2:6] = crop
    dem_path = raw / "dem.tif"
    dsm_path = raw / "dsm.tif"
    _write_raster(dem_path, dem_values, transform=from_origin(98.0, 205.0, 1.0, 1.0), nodata=-9999.0)
    _write_raster(dsm_path, dsm_values, transform=from_origin(98.0, 205.0, 1.0, 1.0), nodata=-9999.0)

    fake_paths = {
        "rgb": rgb_path,
        "dsm": dsm_path,
        "dem": dem_path,
    }
    for role in holdout.REQUIRED_SOURCE_ROLES - set(fake_paths):
        metadata = raw / f"{role}.json"
        metadata.write_text("{}", encoding="utf-8")
        fake_paths[role] = metadata
    monkeypatch.setattr(holdout, "verify_source_files", lambda *_args, **_kwargs: fake_paths)

    contract_digest = sha256(json.dumps(contract, sort_keys=True).encode("utf-8")).hexdigest()
    manifest_path = holdout.prepare_package(contract, contract_digest, root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["inference_run"] is False
    assert manifest["training_or_tuning_permitted"] is False
    assert manifest["support_counts"] == {
        "all_valid": 11,
        "object_gt2m": 5,
        "tall_object_gt5m": 2,
        "negative_reference_artifacts_excluded": 1,
    }
    with rasterio.open(root / "prepared" / "reference_ndsm_m.tif") as raster:
        assert raster.crs == rasterio.crs.CRS.from_epsg(2193)
        assert raster.shape == (3, 4)
        assert raster.read(1)[0, 3] == raster.nodata
    with rasterio.open(root / "prepared" / "mask_object_gt2m.tif") as raster:
        assert int(raster.read(1).sum()) == 5
    assert (root / "NO_TRAIN_OR_TUNE.txt").is_file()
    assert not (root / "CONSUMED.json").exists()


def test_one_shot_plan_must_be_frozen_before_irreversible_consumption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "package"
    root.mkdir()
    manifest = root / "SEALED_MANIFEST.json"
    manifest.write_text("sealed", encoding="utf-8")
    package = holdout.VerifiedPackage(
        root=root,
        manifest_path=manifest,
        manifest_sha256=holdout.sha256_file(manifest),
        contract_sha256="0" * 64,
        consumed=False,
    )
    contract = holdout.load_contract(CONTRACT_PATH)
    checkpoint = tmp_path / "candidate.pt"
    config = tmp_path / "candidate.yaml"
    code = tmp_path / "code"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("model: fixed\n", encoding="utf-8")
    code.mkdir()
    (code / "predict.py").write_text("# fixed\n", encoding="utf-8")

    with pytest.raises(holdout.ExternalHoldoutError, match="frozen evaluation plan"):
        holdout.mark_consumed(package, reason="first evaluation")

    plan = holdout.seal_evaluation_plan(
        package,
        contract,
        candidate_checkpoint=checkpoint,
        candidate_config=config,
        code_path=code,
    )
    assert plan.is_file()
    marker = holdout.mark_consumed(package, reason="first and only model evaluation")
    assert marker.is_file()

    with pytest.raises(holdout.ExternalHoldoutError, match="already been consumed"):
        holdout.mark_consumed(
            holdout.VerifiedPackage(**{**package.__dict__, "consumed": True}),
            reason="illegal rerun",
        )
