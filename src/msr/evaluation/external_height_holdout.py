"""Fail-closed acquisition and sealing for external metric-height evidence.

The helpers in this module deliberately contain no inference, training, model
selection, or checkpoint-promotion logic.  They acquire a preregistered public
scene, verify publisher hashes and metadata, prepare aligned reference rasters,
and make one-shot evaluation state explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import from_bounds


CONTRACT_SCHEMA = "msr.external_metric_height_holdout_contract.v1"
PACKAGE_SCHEMA = "msr.external_metric_height_holdout_package.v1"
PLAN_SCHEMA = "msr.external_metric_height_evaluation_plan.v1"
CONSUMED_SCHEMA = "msr.external_metric_height_consumption.v1"

OFFICIAL_SOURCE_HOSTS = frozenset(
    {
        "nz-imagery.s3.ap-southeast-2.amazonaws.com",
        "nz-elevation.s3.ap-southeast-2.amazonaws.com",
    }
)
REQUIRED_SOURCE_ROLES = frozenset(
    {
        "imagery_bucket_licence",
        "elevation_bucket_licence",
        "rgb_collection_metadata",
        "rgb_item_metadata",
        "rgb",
        "dsm_collection_metadata",
        "dsm_item_metadata",
        "dsm",
        "dem_collection_metadata",
        "dem_item_metadata",
        "dem",
    }
)
REQUIRED_OUTPUTS = frozenset(
    {
        "rgb_aligned_1m.tif",
        "lidar_dsm_aligned_m.tif",
        "lidar_dem_aligned_m.tif",
        "reference_ndsm_m.tif",
        "mask_all_valid.tif",
        "mask_object_gt2m.tif",
        "mask_tall_object_gt5m.tif",
    }
)
REQUIRED_POLICY = {
    "role": "external_test_only",
    "one_shot_evaluation": True,
    "training_permitted": False,
    "validation_permitted": False,
    "calibration_permitted": False,
    "threshold_selection_permitted": False,
    "model_selection_permitted": False,
    "showcase_selection_permitted": False,
    "official_test_reuse": False,
    "model_inference_during_preparation": False,
    "candidate_must_be_frozen_before_evaluation": True,
    "consumption_marker": "CONSUMED.json",
}
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ExternalHoldoutError(RuntimeError):
    """Raised when an external holdout invariant is not satisfied."""


@dataclass(frozen=True)
class VerifiedPackage:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    contract_sha256: str
    consumed: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).resolve().open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    source = Path(path).resolve()
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise ExternalHoldoutError(f"Cannot hash missing path: {source}")
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise ExternalHoldoutError(f"Cannot hash empty directory: {source}")
    digest = sha256()
    for item in files:
        relative = item.relative_to(source).as_posix()
        digest.update(f"{relative}\0{sha256_file(item)}\n".encode("utf-8"))
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalHoldoutError(message)


def _require_sha256(value: Any, label: str) -> str:
    _require(isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None,
             f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _safe_filename(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be non-empty")
    path = Path(value)
    _require(
        not path.is_absolute() and path.name == value and value not in {".", ".."},
        f"{label} must be a plain filename without traversal",
    )
    return value


def load_contract(path: str | Path) -> dict[str, Any]:
    contract_path = Path(path).resolve()
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalHoldoutError(f"Cannot read holdout contract {contract_path}: {error}") from error
    validate_contract(payload)
    return payload


def validate_contract(contract: Mapping[str, Any]) -> None:
    """Validate the safety-critical fields of a holdout contract.

    Unknown descriptive fields are allowed, but no safety field is optional.
    """

    _require(contract.get("schema") == CONTRACT_SCHEMA, "Unexpected contract schema")
    _require(contract.get("status") == "preregistered_unconsumed",
             "Contract must remain preregistered_unconsumed")
    benchmark_id = contract.get("benchmark_id")
    _require(
        isinstance(benchmark_id, str)
        and re.fullmatch(r"[a-z0-9][a-z0-9_]{7,95}", benchmark_id) is not None,
        "benchmark_id must be a stable lowercase slug",
    )

    geography = contract.get("geography")
    _require(isinstance(geography, Mapping), "geography is required")
    _require(geography.get("country") == "New Zealand", "Country must be New Zealand")
    _require(geography.get("known_msr_training_overlap") is False,
             "Known Monocular Surface Reconstruction training overlap must be false")

    selection = contract.get("selection_rule")
    _require(isinstance(selection, Mapping), "selection_rule is required")
    _require(selection.get("method") == "first_lexicographic_complete_overlap",
             "Selection must be deterministic and non-visual")
    _require(selection.get("visual_selection_permitted") is False,
             "Visual selection must be prohibited")

    licence = contract.get("licence")
    _require(isinstance(licence, Mapping), "licence is required")
    _require(licence.get("spdx") == "CC-BY-4.0", "Only the sealed CC-BY-4.0 source is accepted")
    _require(licence.get("licensor") == "BOPLASS", "Expected BOPLASS attribution")
    _require(
        urlparse(str(licence.get("attribution_url", ""))).hostname == "www.linz.govt.nz",
        "Attribution URL must be the official LINZ page",
    )

    policy = contract.get("policy")
    _require(isinstance(policy, Mapping), "policy is required")
    for field, expected in REQUIRED_POLICY.items():
        _require(policy.get(field) == expected, f"Unsafe policy value for {field}")
    _safe_filename(policy.get("consumption_marker"), "policy.consumption_marker")
    forbidden = policy.get("forbidden_sources")
    _require(
        isinstance(forbidden, list)
        and "GAMUS official test" in forbidden
        and "DFC19 showcase" in forbidden,
        "Forbidden official-test/showcase sources must remain explicit",
    )

    grid = contract.get("grid")
    _require(isinstance(grid, Mapping), "grid is required")
    _require(grid.get("horizontal_crs") == "EPSG:2193", "Expected official NZTM2000 grid")
    _require(grid.get("horizontal_units") == "metre", "Horizontal units must be metres")
    _require(float(grid.get("rgb_native_gsd_m", -1)) == 0.1, "RGB GSD must be 0.1 m")
    _require(float(grid.get("reference_gsd_m", -1)) == 1.0, "Reference GSD must be 1 m")
    _require(int(grid.get("prepared_width", 0)) > 0 and int(grid.get("prepared_height", 0)) > 0,
             "Prepared dimensions must be positive")
    transform = grid.get("prepared_transform_gdal")
    bounds = grid.get("prepared_bounds")
    _require(isinstance(transform, list) and len(transform) == 6,
             "prepared_transform_gdal must contain six values")
    _require(isinstance(bounds, list) and len(bounds) == 4,
             "prepared_bounds must contain four values")

    height = contract.get("height_reference")
    _require(isinstance(height, Mapping), "height_reference is required")
    _require(height.get("kind") == "ndsm", "Reference must be an nDSM")
    _require(height.get("formula") == "lidar_dsm_m - lidar_dem_m",
             "Reference must be paired DSM-minus-DEM")
    _require(height.get("value_units") == "metre", "Height values must be metres")
    _require(height.get("negative_policy") == "exclude", "Negative artifacts must be excluded")
    _require(height.get("upper_height_cutoff_m") is None,
             "Post-hoc upper height filtering is prohibited")
    supports = height.get("scoring_supports")
    _require(isinstance(supports, Mapping), "scoring_supports are required")
    _require(float(supports.get("object_only", {}).get("threshold_m", -1)) == 2.0,
             "Object-only threshold must remain preregistered at >2 m")
    _require(float(supports.get("tall_object", {}).get("threshold_m", -1)) == 5.0,
             "Tall-object threshold must remain preregistered at >5 m")

    sources = contract.get("sources")
    _require(isinstance(sources, list), "sources must be a list")
    roles: set[str] = set()
    filenames: set[str] = set()
    for index, source in enumerate(sources):
        _require(isinstance(source, Mapping), f"sources[{index}] must be an object")
        role = str(source.get("role", ""))
        _require(role in REQUIRED_SOURCE_ROLES, f"Unknown source role: {role}")
        _require(role not in roles, f"Duplicate source role: {role}")
        roles.add(role)
        filename = _safe_filename(source.get("filename"), f"sources[{index}].filename")
        _require(filename not in filenames, f"Duplicate source filename: {filename}")
        filenames.add(filename)
        parsed = urlparse(str(source.get("url", "")))
        _require(parsed.scheme == "https" and parsed.hostname in OFFICIAL_SOURCE_HOSTS,
                 f"Source {role} is not on an approved official LINZ bucket")
        _require_sha256(source.get("sha256"), f"sources[{index}].sha256")
        _require(isinstance(source.get("size_bytes"), int) and source["size_bytes"] > 0,
                 f"sources[{index}].size_bytes must be positive")
        if role.endswith("_metadata"):
            _require(isinstance(source.get("stac_expectation"), Mapping),
                     f"{role} requires pinned STAC expectations")
    _require(roles == REQUIRED_SOURCE_ROLES,
             f"Source roles differ from the sealed set: {sorted(roles ^ REQUIRED_SOURCE_ROLES)}")

    outputs = contract.get("prepared_outputs")
    _require(isinstance(outputs, list), "prepared_outputs must be a list")
    normalized_outputs = {_safe_filename(value, "prepared output") for value in outputs}
    _require(normalized_outputs == REQUIRED_OUTPUTS,
             "Prepared outputs differ from the sealed benchmark interface")


def verify_contract_seal(contract_path: str | Path, seal_path: str | Path | None = None) -> str:
    path = Path(contract_path).resolve()
    seal = Path(seal_path).resolve() if seal_path is not None else path.with_suffix(".sha256")
    try:
        parts = seal.read_text(encoding="utf-8").strip().split()
    except OSError as error:
        raise ExternalHoldoutError(f"Cannot read contract seal {seal}: {error}") from error
    _require(len(parts) == 2, "Contract seal must contain '<sha256> <filename>'")
    expected = _require_sha256(parts[0], "contract seal")
    _require(parts[1] == path.name, "Contract seal filename mismatch")
    actual = sha256_file(path)
    _require(actual == expected, f"Contract hash mismatch: expected {expected}, got {actual}")
    load_contract(path)
    return actual


def resolve_data_root(
    contract: Mapping[str, Any],
    *,
    allowed_root: str | Path = r"D:\MSRData",
) -> Path:
    root = Path(str(contract.get("data_root", ""))).resolve()
    allowed = Path(allowed_root).resolve()
    try:
        common = Path(os.path.commonpath([str(root), str(allowed)]))
    except ValueError as error:
        raise ExternalHoldoutError(f"Holdout root is outside {allowed}: {root}") from error
    _require(common == allowed and root != allowed,
             f"Holdout root must be a child of {allowed}, got {root}")
    return root


def _source_map(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(source["role"]): source for source in contract["sources"]}


def _verify_one(path: Path, source: Mapping[str, Any]) -> None:
    _require(path.is_file(), f"Missing source file: {path}")
    actual_size = path.stat().st_size
    _require(actual_size == int(source["size_bytes"]),
             f"Size mismatch for {path.name}: expected {source['size_bytes']}, got {actual_size}")
    actual_hash = sha256_file(path)
    _require(actual_hash == source["sha256"],
             f"SHA-256 mismatch for {path.name}: expected {source['sha256']}, got {actual_hash}")


def verify_source_files(contract: Mapping[str, Any], root: str | Path) -> dict[str, Path]:
    raw = Path(root).resolve() / "raw"
    paths: dict[str, Path] = {}
    for role, source in _source_map(contract).items():
        path = raw / str(source["filename"])
        _verify_one(path, source)
        paths[role] = path
    validate_stac_metadata(contract, paths)
    return paths


def _compare_json_value(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, list) and all(isinstance(value, (int, float)) for value in expected):
        _require(isinstance(actual, list) and len(actual) == len(expected), f"{label} length mismatch")
        _require(np.allclose(actual, expected, rtol=0.0, atol=1e-7), f"{label} mismatch")
    else:
        _require(actual == expected, f"{label} mismatch: expected {expected!r}, got {actual!r}")


def validate_stac_metadata(
    contract: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    """Check downloaded STAC semantics in addition to their byte hashes."""

    sources = _source_map(contract)
    documents: dict[str, dict[str, Any]] = {}
    for role, source in sources.items():
        expectation = source.get("stac_expectation")
        if expectation is None:
            continue
        try:
            document = json.loads(paths[role].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ExternalHoldoutError(f"Invalid STAC metadata for {role}: {error}") from error
        documents[role] = document
        for key, expected in expectation.items():
            if key == "asset_sha256":
                checksum = document.get("assets", {}).get("visual", {}).get("file:checksum")
                _compare_json_value(checksum, f"1220{expected}", f"{role}.assets.visual.file:checksum")
            else:
                _compare_json_value(document.get(key), expected, f"{role}.{key}")

    _require(
        sources["rgb_item_metadata"]["stac_expectation"]["asset_sha256"]
        == sources["rgb"]["sha256"],
        "RGB STAC asset checksum differs from the pinned raster checksum",
    )
    _require(
        sources["dsm_item_metadata"]["stac_expectation"]["asset_sha256"]
        == sources["dsm"]["sha256"],
        "DSM STAC asset checksum differs from the pinned raster checksum",
    )
    _require(
        sources["dem_item_metadata"]["stac_expectation"]["asset_sha256"]
        == sources["dem"]["sha256"],
        "DEM STAC asset checksum differs from the pinned raster checksum",
    )

    rgb_collection = documents["rgb_collection_metadata"]
    item_ids = sorted(
        Path(str(link.get("href", ""))).stem
        for link in rgb_collection.get("links", [])
        if link.get("rel") == "item"
    )
    _require(bool(item_ids), "RGB collection contains no STAC item links")
    _require(
        item_ids[0] == contract["selection_rule"]["rgb_item_id"],
        "Selected RGB item is not the first lexicographic collection item",
    )

    rgb_item = documents["rgb_item_metadata"]
    dsm_item = documents["dsm_item_metadata"]
    dem_item = documents["dem_item_metadata"]
    _require(dsm_item["bbox"] == dem_item["bbox"], "DSM and DEM item bounds differ")
    rgb_bbox = [float(value) for value in rgb_item["bbox"]]
    elevation_bbox = [float(value) for value in dsm_item["bbox"]]
    _require(
        elevation_bbox[0] <= rgb_bbox[0]
        and elevation_bbox[1] <= rgb_bbox[1]
        and elevation_bbox[2] >= rgb_bbox[2]
        and elevation_bbox[3] >= rgb_bbox[3],
        "Selected RGB item is not completely contained by DSM/DEM coverage",
    )

    capture = contract["capture"]
    for role, start_field, end_field in (
        ("rgb_item_metadata", "rgb_start_utc", "rgb_end_utc"),
        ("dsm_item_metadata", "lidar_start_utc", "lidar_end_utc"),
        ("dem_item_metadata", "lidar_start_utc", "lidar_end_utc"),
    ):
        properties = documents[role].get("properties", {})
        _require(properties.get("start_datetime") == capture[start_field],
                 f"{role} capture start differs from contract")
        _require(properties.get("end_datetime") == capture[end_field],
                 f"{role} capture end differs from contract")

    for role in ("rgb_collection_metadata", "dsm_collection_metadata", "dem_collection_metadata"):
        providers = documents[role].get("providers", [])
        _require(
            any(
                provider.get("name") == contract["licence"]["licensor"]
                and "licensor" in provider.get("roles", [])
                for provider in providers
            ),
            f"{role} does not authenticate the pinned licensor",
        )


def _download_one(
    source: Mapping[str, Any],
    destination: Path,
    *,
    progress: Callable[[str, int, int], None] | None,
    timeout_s: float,
) -> None:
    if destination.exists():
        _verify_one(destination, source)
        if progress:
            progress(destination.name, destination.stat().st_size, int(source["size_bytes"]))
        return
    partial = destination.with_name(f"{destination.name}.partial")
    offset = partial.stat().st_size if partial.exists() else 0
    expected_size = int(source["size_bytes"])
    _require(offset <= expected_size, f"Partial file is larger than expected: {partial}")
    headers = {"User-Agent": "Monocular Surface Reconstruction-external-holdout/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(str(source["url"]), headers=headers)
    with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - pinned HTTPS hosts only
        status = int(getattr(response, "status", 200))
        if offset and status != 206:
            offset = 0
            mode = "wb"
        else:
            mode = "ab" if offset else "wb"
        with partial.open(mode) as target:
            downloaded = offset
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(destination.name, downloaded, expected_size)
    _verify_one(partial, source)
    os.replace(partial, destination)


def acquire_sources(
    contract: Mapping[str, Any],
    root: str | Path,
    *,
    progress: Callable[[str, int, int], None] | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Path]:
    """Download only the preregistered files and verify publisher hashes."""

    validate_contract(contract)
    destination_root = Path(root).resolve()
    marker = destination_root / str(contract["policy"]["consumption_marker"])
    _require(not marker.exists(), "Consumed holdout cannot be reacquired or changed")
    raw = destination_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for source in contract["sources"]:
        _download_one(
            source,
            raw / str(source["filename"]),
            progress=progress,
            timeout_s=timeout_s,
        )
    return verify_source_files(contract, destination_root)


def _expected_transform(contract: Mapping[str, Any]) -> Affine:
    # Contract stores GDAL order: c, a, b, f, d, e.
    return Affine.from_gdal(*[float(value) for value in contract["grid"]["prepared_transform_gdal"]])


def _same_transform(left: Affine, right: Affine) -> bool:
    return bool(np.allclose(tuple(left), tuple(right), rtol=0.0, atol=1e-9))


def _write_raster_atomic(path: Path, values: np.ndarray, profile: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"Refusing to overwrite prepared output: {path}")
    temporary = path.with_name(f"{path.name}.partial.tif")
    if temporary.exists():
        temporary.unlink()
    with rasterio.open(temporary, "w", **profile) as target:
        target.write(values)
    os.replace(temporary, path)


def _read_aligned_elevation(
    source: rasterio.DatasetReader,
    *,
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    transform: Affine,
) -> tuple[np.ndarray, np.ndarray]:
    window = from_bounds(*bounds, transform=source.transform)
    rounded = window.round_offsets().round_lengths()
    _require(
        np.allclose(
            (window.col_off, window.row_off, window.width, window.height),
            (rounded.col_off, rounded.row_off, rounded.width, rounded.height),
            rtol=0.0,
            atol=1e-7,
        ),
        "Reference crop is not aligned to source pixels",
    )
    _require(int(rounded.width) == width and int(rounded.height) == height,
             "Reference crop dimensions differ from the sealed output grid")
    _require(_same_transform(source.window_transform(rounded), transform),
             "Reference crop transform differs from the sealed output grid")
    values = source.read(1, window=rounded).astype(np.float32, copy=False)
    valid = source.read_masks(1, window=rounded) > 0
    if source.nodata is not None:
        valid &= ~np.isclose(values, source.nodata, equal_nan=True)
    valid &= np.isfinite(values)
    return values, valid


def _raster_manifest_entry(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as source:
        return {
            "filename": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "width": source.width,
            "height": source.height,
            "count": source.count,
            "dtype": list(source.dtypes),
            "crs": source.crs.to_string() if source.crs else None,
            "transform_gdal": list(source.transform.to_gdal()),
            "nodata": source.nodata,
        }


def prepare_package(
    contract: Mapping[str, Any],
    contract_sha256: str,
    root: str | Path,
) -> Path:
    """Prepare the fixed RGB/DSM/DEM/nDSM grids without running a model."""

    validate_contract(contract)
    _require_sha256(contract_sha256, "contract_sha256")
    destination_root = Path(root).resolve()
    marker = destination_root / str(contract["policy"]["consumption_marker"])
    _require(not marker.exists(), "Consumed holdout cannot be prepared again")
    manifest_path = destination_root / "SEALED_MANIFEST.json"
    if manifest_path.exists():
        return verify_prepared_package(contract, contract_sha256, destination_root).manifest_path

    paths = verify_source_files(contract, destination_root)
    output = destination_root / "prepared"
    output.mkdir(parents=True, exist_ok=True)
    existing = [output / name for name in contract["prepared_outputs"] if (output / name).exists()]
    _require(not existing, f"Prepared files exist without a sealed manifest: {existing}")

    grid = contract["grid"]
    width = int(grid["prepared_width"])
    height = int(grid["prepared_height"])
    bounds = tuple(float(value) for value in grid["prepared_bounds"])
    transform = _expected_transform(contract)
    expected_crs = rasterio.crs.CRS.from_string(str(grid["horizontal_crs"]))

    with rasterio.open(paths["rgb"]) as rgb_source:
        _require(rgb_source.crs == expected_crs, "RGB CRS mismatch")
        _require(rgb_source.count >= 3, "RGB source has fewer than three bands")
        _require(np.isclose(abs(rgb_source.transform.a), float(grid["rgb_native_gsd_m"])),
                 "RGB source GSD mismatch")
        _require(np.allclose(tuple(rgb_source.bounds), bounds, rtol=0.0, atol=1e-6),
                 "RGB bounds differ from preregistered bounds")
        rgb = rgb_source.read(
            (1, 2, 3),
            out_shape=(3, height, width),
            resampling=Resampling.average,
        ).astype(np.uint8, copy=False)
        rgb_valid = rgb_source.dataset_mask(
            out_shape=(height, width), resampling=Resampling.nearest
        ) > 0

    reference_values: dict[str, np.ndarray] = {}
    reference_valid: dict[str, np.ndarray] = {}
    for role in ("dsm", "dem"):
        with rasterio.open(paths[role]) as source:
            _require(source.crs == expected_crs, f"{role.upper()} CRS mismatch")
            _require(source.count == 1, f"{role.upper()} must contain one band")
            _require(np.isclose(abs(source.transform.a), float(grid["reference_gsd_m"])),
                     f"{role.upper()} GSD mismatch")
            source_bounds = source.bounds
            _require(
                source_bounds.left <= bounds[0]
                and source_bounds.bottom <= bounds[1]
                and source_bounds.right >= bounds[2]
                and source_bounds.top >= bounds[3],
                f"{role.upper()} does not completely cover the RGB tile",
            )
            values, valid = _read_aligned_elevation(
                source,
                bounds=bounds,
                width=width,
                height=height,
                transform=transform,
            )
            reference_values[role] = values
            reference_valid[role] = valid

    signed_ndsm = reference_values["dsm"] - reference_values["dem"]
    all_valid = (
        rgb_valid
        & reference_valid["dsm"]
        & reference_valid["dem"]
        & np.isfinite(signed_ndsm)
        & (signed_ndsm >= 0.0)
    )
    _require(bool(all_valid.any()), "No valid aligned reference pixels")
    object_mask = all_valid & (signed_ndsm > 2.0)
    tall_mask = all_valid & (signed_ndsm > 5.0)
    _require(bool(object_mask.any()), "Preregistered tile contains no >2 m object support")
    _require(bool(tall_mask.any()), "Preregistered tile contains no >5 m tall-object support")

    nodata = -9999.0
    dsm = np.where(all_valid, reference_values["dsm"], nodata).astype(np.float32)
    dem = np.where(all_valid, reference_values["dem"], nodata).astype(np.float32)
    ndsm = np.where(all_valid, signed_ndsm, nodata).astype(np.float32)
    rgb[:, ~all_valid] = 0

    common = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "crs": expected_crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "DEFLATE",
    }
    _write_raster_atomic(
        output / "rgb_aligned_1m.tif",
        rgb,
        {**common, "count": 3, "dtype": "uint8", "nodata": None},
    )
    for filename, values in (
        ("lidar_dsm_aligned_m.tif", dsm),
        ("lidar_dem_aligned_m.tif", dem),
        ("reference_ndsm_m.tif", ndsm),
    ):
        _write_raster_atomic(
            output / filename,
            values[None],
            {**common, "count": 1, "dtype": "float32", "nodata": nodata},
        )
    for filename, values in (
        ("mask_all_valid.tif", all_valid),
        ("mask_object_gt2m.tif", object_mask),
        ("mask_tall_object_gt5m.tif", tall_mask),
    ):
        _write_raster_atomic(
            output / filename,
            values.astype(np.uint8, copy=False)[None],
            {**common, "count": 1, "dtype": "uint8", "nodata": 0},
        )

    source_entries = {
        role: {
            "filename": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "url": _source_map(contract)[role]["url"],
        }
        for role, path in sorted(paths.items())
    }
    output_entries = {
        name: _raster_manifest_entry(output / name)
        for name in sorted(contract["prepared_outputs"])
    }
    manifest = {
        "schema": PACKAGE_SCHEMA,
        "benchmark_id": contract["benchmark_id"],
        "created_utc": utc_now(),
        "contract_sha256": contract_sha256,
        "status": "prepared_unconsumed",
        "purpose": contract["purpose"],
        "licence": contract["licence"],
        "capture": contract["capture"],
        "grid": contract["grid"],
        "height_reference": contract["height_reference"],
        "policy": contract["policy"],
        "sources": source_entries,
        "outputs": output_entries,
        "support_counts": {
            "all_valid": int(all_valid.sum()),
            "object_gt2m": int(object_mask.sum()),
            "tall_object_gt5m": int(tall_mask.sum()),
            "negative_reference_artifacts_excluded": int(
                (rgb_valid & reference_valid["dsm"] & reference_valid["dem"] & (signed_ndsm < 0)).sum()
            ),
        },
        "inference_run": False,
        "training_or_tuning_permitted": False,
    }
    temporary_manifest = manifest_path.with_suffix(".json.partial")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    no_train_marker = destination_root / "NO_TRAIN_OR_TUNE.txt"
    no_train_marker.write_text(
        "EXTERNAL TEST ONLY. Never train, validate, calibrate, select, or tune on this package.\n",
        encoding="utf-8",
    )
    return verify_prepared_package(contract, contract_sha256, destination_root).manifest_path


def verify_prepared_package(
    contract: Mapping[str, Any],
    contract_sha256: str,
    root: str | Path,
) -> VerifiedPackage:
    validate_contract(contract)
    destination_root = Path(root).resolve()
    verify_source_files(contract, destination_root)
    manifest_path = destination_root / "SEALED_MANIFEST.json"
    _require(manifest_path.is_file(), f"Missing sealed package manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ExternalHoldoutError(f"Invalid package manifest: {error}") from error
    _require(manifest.get("schema") == PACKAGE_SCHEMA, "Unexpected package manifest schema")
    _require(manifest.get("benchmark_id") == contract["benchmark_id"], "Benchmark ID mismatch")
    _require(manifest.get("contract_sha256") == contract_sha256, "Contract digest mismatch")
    _require(manifest.get("status") == "prepared_unconsumed", "Package status is not sealed")
    _require(manifest.get("inference_run") is False, "Preparation manifest claims inference")
    _require(manifest.get("training_or_tuning_permitted") is False, "Unsafe package policy")
    outputs = manifest.get("outputs")
    _require(isinstance(outputs, Mapping) and set(outputs) == REQUIRED_OUTPUTS,
             "Package output set mismatch")
    for name, entry in outputs.items():
        path = destination_root / "prepared" / name
        _require(path.is_file(), f"Missing prepared output: {path}")
        _require(path.stat().st_size == int(entry["size_bytes"]), f"Size mismatch: {name}")
        _require(sha256_file(path) == entry["sha256"], f"Hash mismatch: {name}")
        with rasterio.open(path) as source:
            _require(source.crs is not None and source.crs.to_string() == contract["grid"]["horizontal_crs"],
                     f"CRS mismatch: {name}")
            _require(source.width == int(contract["grid"]["prepared_width"]), f"Width mismatch: {name}")
            _require(source.height == int(contract["grid"]["prepared_height"]), f"Height mismatch: {name}")
            _require(_same_transform(source.transform, _expected_transform(contract)),
                     f"Transform mismatch: {name}")
    marker = destination_root / str(contract["policy"]["consumption_marker"])
    return VerifiedPackage(
        root=destination_root,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        contract_sha256=contract_sha256,
        consumed=marker.exists(),
    )


def seal_evaluation_plan(
    package: VerifiedPackage,
    contract: Mapping[str, Any],
    *,
    candidate_checkpoint: str | Path,
    candidate_config: str | Path,
    code_path: str | Path,
    app_pointer: str | Path | None = None,
) -> Path:
    """Freeze candidate state before the one permitted model evaluation."""

    _require(not package.consumed, "Cannot seal a plan after holdout consumption")
    plan_path = package.root / "EVALUATION_PLAN.json"
    _require(not plan_path.exists(), f"Evaluation plan already exists: {plan_path}")
    checkpoint = Path(candidate_checkpoint).resolve()
    config = Path(candidate_config).resolve()
    code = Path(code_path).resolve()
    _require(checkpoint.is_file(), f"Missing candidate checkpoint: {checkpoint}")
    _require(config.is_file(), f"Missing candidate config: {config}")
    _require(code.exists(), f"Missing code path: {code}")
    frozen = {
        "candidate_checkpoint_sha256": sha256_file(checkpoint),
        "candidate_config_sha256": sha256_file(config),
        "code_sha256": sha256_tree(code),
    }
    if app_pointer is not None:
        pointer = Path(app_pointer).resolve()
        _require(pointer.is_file(), f"Missing app pointer: {pointer}")
        frozen["app_pointer_sha256"] = sha256_file(pointer)
    plan = {
        "schema": PLAN_SCHEMA,
        "benchmark_id": contract["benchmark_id"],
        "created_utc": utc_now(),
        "status": "frozen_unconsumed",
        "contract_sha256": package.contract_sha256,
        "package_manifest_sha256": package.manifest_sha256,
        "candidate": frozen,
        "supports": ["all_valid", "object_gt2m", "tall_object_gt5m"],
        "metrics": ["rmse_m", "mae_m", "bias_m", "correlation", "r2"],
        "no_tuning_or_reselection": True,
        "publish_complete_result_regardless_of_outcome": True,
    }
    with plan_path.open("x", encoding="utf-8") as target:
        json.dump(plan, target, indent=2, sort_keys=True)
        target.write("\n")
    return plan_path


def mark_consumed(package: VerifiedPackage, *, reason: str) -> Path:
    """Create the irreversible marker before the first model reads the scene."""

    _require(not package.consumed, "Holdout has already been consumed")
    _require(isinstance(reason, str) and bool(reason.strip()), "Consumption reason is required")
    plan_path = package.root / "EVALUATION_PLAN.json"
    _require(plan_path.is_file(), "A frozen evaluation plan is required before consumption")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _require(plan.get("schema") == PLAN_SCHEMA and plan.get("status") == "frozen_unconsumed",
             "Evaluation plan is not a valid frozen plan")
    _require(plan.get("package_manifest_sha256") == package.manifest_sha256,
             "Evaluation plan references a different package")
    marker = package.root / "CONSUMED.json"
    payload = {
        "schema": CONSUMED_SCHEMA,
        "benchmark_id": plan["benchmark_id"],
        "consumed_utc": utc_now(),
        "reason": reason.strip(),
        "evaluation_plan_sha256": sha256_file(plan_path),
        "irreversible": True,
    }
    with marker.open("x", encoding="utf-8") as target:
        json.dump(payload, target, indent=2, sort_keys=True)
        target.write("\n")
    return marker
