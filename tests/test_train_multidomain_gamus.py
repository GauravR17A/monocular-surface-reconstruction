import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from msr.models.height_net import HeightNet

h5py = pytest.importorskip("h5py")


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_multidomain.py"
PILOT_CONFIG = (
    Path(__file__).parents[1]
    / "configs"
    / "multidomain_surface_gamus_direct_height_pilot.yaml"
)
SPEC = importlib.util.spec_from_file_location("train_multidomain_gamus_test", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_direct_height_pilot_config_is_full_train_and_test_free() -> None:
    config = yaml.safe_load(PILOT_CONFIG.read_text(encoding="utf-8"))

    assert config["data"]["dataset"] == "gamus"
    assert "train_samples_per_epoch" not in config["data"]
    assert config["data"]["validation_patch_size"] == 1024
    assert config["data"]["require_approved_index"] is True
    assert config["data"]["approved_index_file_sha256"] == (
        "5320d2e97be357b1e1725d7f2d9640493522ae3d13f1a051b63de55b530c05aa"
    )
    assert "test_manifest" not in config["data"]
    assert config["model"]["fusion_mode"] == "legacy"
    assert config["training"]["height_weight"] == pytest.approx(1.0)
    assert config["training"]["batch_size"] == 2
    assert config["training"]["gradient_accumulation"] == 4
    assert config["training"]["drop_last"] is False
    assert config["evaluation"]["promotion_eligible"] is False
    guards = config["evaluation"]["validation_guards"]["gamus"]
    assert guards["object_domain_macro_rmse_m"]["min_improvement"] == pytest.approx(
        0.20
    )
    assert guards["correlation"]["max_drop"] == pytest.approx(0.01)
    assert guards["r2"]["max_drop"] == pytest.approx(0.02)


def _write_h5(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("image", data=values)


def _triplet(root: Path, split: str, sample_id: str) -> None:
    height = np.arange(20, dtype=np.float32).reshape(4, 5)
    rgb = np.stack((height, height + 20, height + 40), axis=-1).astype(np.uint8)
    classes = np.resize(
        np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32), (4, 5)
    )
    _write_h5(root / "images" / split / f"{sample_id}_RGB.h5", rgb)
    _write_h5(root / "heights" / split / f"{sample_id}_AGL.h5", height)
    _write_h5(root / "classes" / split / f"{sample_id}_CLS.h5", classes)


def _datasets(root: Path, splits: tuple[str, ...] = ("train", "val", "test")):
    data_config = {
        "patch_size": 3,
        "validation_patch_size": 4,
        "train_samples_per_epoch": 2,
        "num_workers": 0,
        "train_radiometric_policy": "raw",
        "validation_radiometric_policy": "raw",
    }
    datasets = {
        split: MODULE.make_gamus_dataset(
            root, split, data_config, training=split == "train"
        )
        for split in splits
    }
    return data_config, datasets


def test_native_gamus_training_loader_matches_batch_contract(tmp_path: Path) -> None:
    for split, tile in (
        ("train", "DC_01_01"),
        ("val", "DC_02_01"),
    ):
        _triplet(tmp_path, split, tile)
    data_config, datasets = _datasets(tmp_path, ("train", "val"))

    # Reusing a city is valid in the official split; exact tile IDs may not leak.
    MODULE.validate_gamus_splits(
        datasets, require_complete_official_splits=False
    )
    loader = MODULE.make_gamus_loader(
        datasets["train"], data_config, {"batch_size": 1}, training=True
    )
    batch = next(iter(loader))

    assert set(batch) == {
        "image",
        "height",
        "image_valid_mask",
        "classification_valid_mask",
        "height_valid_mask",
        "fine_class_target",
        "dark_pixel_proxy_mask",
        "valid_mask",
        "regression_mask",
        "building_mask",
        "vegetation_mask",
        "domain_target",
        "domain_valid_mask",
        "relative_prior",
        "sample_id",
        "region",
        "landscape",
    }
    assert batch["image"].shape == (1, 3, 3, 3)
    assert batch["height"].shape == (1, 1, 3, 3)
    assert batch["fine_class_target"].shape == (1, 3, 3)
    assert batch["sample_id"] == ["DC_01_01"]


def test_gamus_development_factory_does_not_resolve_test_split(tmp_path: Path) -> None:
    _triplet(tmp_path, "train", "DC_01_01")
    _triplet(tmp_path, "val", "PHL_01_01")
    data_config, _ = _datasets(tmp_path, ("train", "val"))

    datasets = MODULE.make_gamus_development_datasets(tmp_path, data_config)

    assert tuple(datasets) == ("train", "val")
    assert not (tmp_path / "images" / "test").exists()
    MODULE.validate_gamus_splits(
        datasets, require_complete_official_splits=False
    )


def test_gamus_approved_index_can_be_required_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="approved_index_path is missing"):
        MODULE.make_gamus_dataset(
            tmp_path,
            "train",
            {
                "patch_size": 4,
                "require_approved_index": True,
            },
            training=True,
        )

    approved_index = tmp_path / "approved.json"
    approved_index.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        MODULE.make_gamus_dataset(
            tmp_path,
            "train",
            {
                "patch_size": 4,
                "approved_index_path": str(approved_index),
                "approved_index_file_sha256": "0" * 64,
                "require_approved_index": True,
            },
            training=True,
        )


def test_gamus_completeness_uses_declared_approved_counts(tmp_path: Path) -> None:
    _triplet(tmp_path, "train", "NYC_01_01")
    _triplet(tmp_path, "val", "DC_01_01")
    approved_index = tmp_path / "approved.json"
    approved_index.write_text(
        json.dumps(
            {
                "schema": "msr.gamus.approved_samples.v1",
                "splits": {
                    "train": {
                        "source_count": 5004,
                        "expected_official_count": 5004,
                        "approved_count": 1,
                        "approved_sample_ids": ["NYC_01_01"],
                    },
                    "val": {
                        "source_count": 859,
                        "expected_official_count": 859,
                        "approved_count": 1,
                        "approved_sample_ids": ["DC_01_01"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    data_config, _ = _datasets(tmp_path, ("train", "val"))
    data_config.update(
        {
            "approved_index_path": str(approved_index),
            "approved_index_file_sha256": hashlib.sha256(
                approved_index.read_bytes()
            ).hexdigest(),
            "require_approved_index": True,
        }
    )

    datasets = MODULE.make_gamus_development_datasets(tmp_path, data_config)

    MODULE.validate_gamus_splits(
        datasets, require_complete_official_splits=True
    )
    assert {split: len(dataset.records) for split, dataset in datasets.items()} == {
        "train": 1,
        "val": 1,
    }


class _ToySurfaceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.encoder = nn.Module()
        self.base_model.encoder.stages_3 = nn.Linear(2, 2)
        self.base_model.deepest = nn.Linear(2, 2)
        self.base_model.decoder_stages = nn.ModuleList((nn.Linear(2, 2),))
        self.base_model.height_head = nn.Linear(2, 1)
        self.base_model.building_head = nn.Linear(2, 1)
        self.adapter = nn.Linear(2, 2)
        self.domain_head = nn.Linear(2, 3)
        self.fine_semantic_head = nn.Linear(2, 6)
        self.canopy_height_head = nn.Linear(2, 1)
        self.log_variance_head = nn.Linear(2, 1)
        self.base_trainable = False


def test_explicit_parameter_groups_freeze_everything_else() -> None:
    model = _ToySurfaceModel()
    groups = MODULE.configure_explicit_parameter_groups(
        model,
        {
            "freeze_base_epochs": 0,
            "parameter_groups": [
                {
                    "name": "deep_encoder",
                    "prefixes": ["base_model.encoder.stages_3."],
                    "learning_rate": 5.0e-6,
                },
                {
                    "name": "decoder",
                    "prefixes": [
                        "base_model.deepest.",
                        "base_model.decoder_stages.",
                        "base_model.height_head.",
                    ],
                    "learning_rate": 3.0e-5,
                },
                {
                    "name": "adapter",
                    "prefixes": ["adapter.", "domain_head.", "canopy_height_head."],
                    "learning_rate": 1.0e-4,
                },
            ],
        },
    )

    assert groups is not None
    assert [group["group_name"] for group in groups] == [
        "deep_encoder",
        "decoder",
        "adapter",
    ]
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(
        name.startswith(
            (
                "base_model.encoder.stages_3.",
                "base_model.deepest.",
                "base_model.decoder_stages.",
                "base_model.height_head.",
                "adapter.",
                "domain_head.",
                "canopy_height_head.",
            )
        )
        for name in trainable
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith(("base_model.building_head.", "log_variance_head."))
    )
    assert model.base_trainable is True


def test_head_only_parameter_group_freezes_every_height_tensor() -> None:
    model = _ToySurfaceModel()
    groups = MODULE.configure_explicit_parameter_groups(
        model,
        {
            "freeze_base_epochs": 0,
            "parameter_groups": [
                {
                    "name": "fine_semantic_head_only",
                    "prefixes": ["fine_semantic_head."],
                    "learning_rate": 1.0e-3,
                }
            ],
        },
    )

    assert groups is not None
    assert [group["group_name"] for group in groups] == [
        "fine_semantic_head_only"
    ]
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == {"fine_semantic_head.weight", "fine_semantic_head.bias"}
    assert model.base_trainable is False


def test_refined_head_only_group_keeps_every_height_tensor_frozen() -> None:
    model = MODULE.DomainGatedSurfaceNet(
        HeightNet(
            backbone="resnet18",
            pretrained=False,
            decoder_channels=(64, 32, 16, 8, 4),
            auxiliary_building_head=True,
        ),
        hidden_channels=8,
        fine_semantic_classes=6,
        fine_semantic_head_type="spatial_refined",
        freeze_base=True,
    )
    groups = MODULE.configure_explicit_parameter_groups(
        model,
        {
            "freeze_base_epochs": 0,
            "parameter_groups": [
                {
                    "name": "fine_semantic_head_only",
                    "prefixes": ["fine_semantic_head."],
                    "learning_rate": 1.0e-3,
                }
            ],
        },
    )

    assert groups is not None
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert len(trainable) > 2
    assert trainable == {
        name
        for name, _ in model.named_parameters()
        if name.startswith("fine_semantic_head.")
    }
    assert all(name.startswith("fine_semantic_head.") for name in trainable)
    assert model.base_trainable is False


def test_fine_class_sampling_is_sealed_and_preserves_city_mass(tmp_path: Path) -> None:
    rows = [
        ("DC_a", "DC", 0.0),
        ("DC_b", "DC", 1.0),
        ("PHL_a", "PHL", 0.25),
        ("PHL_b", "PHL", 0.25),
    ]
    path = tmp_path / "index.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "schema": "msr.gamus.train_six_class_tile_index.v1",
                    "sample_id": sample_id,
                    "city": city,
                    "uniform_random_crop_384": {
                        "water_hit_probability": probability
                    },
                }
            )
            + "\n"
            for sample_id, city, probability in rows
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset = SimpleNamespace(
        records=[SimpleNamespace(sample_id=sample_id) for sample_id, _, _ in rows]
    )

    weights, diagnostics = MODULE.gamus_fine_class_sampling_weights(
        dataset,
        {
            "fine_class_sampling_index_path": path,
            "fine_class_sampling_index_sha256": digest,
            "water_sampling_boost": 2.0,
            "preserve_city_sampling_mass": True,
        },
    )

    assert weights == pytest.approx([0.5, 1.5, 1.0, 1.0])
    assert sum(weights[:2]) == pytest.approx(2.0)
    assert sum(weights[2:]) == pytest.approx(2.0)
    assert diagnostics["uniform_expected_water_crop_hit_rate"] == pytest.approx(
        0.375
    )
    assert diagnostics["weighted_expected_water_crop_hit_rate"] == pytest.approx(
        0.5
    )


def test_fine_class_sampling_rejects_incomplete_index(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema": "msr.gamus.train_six_class_tile_index.v1",
                "sample_id": "DC_a",
                "city": "DC",
                "uniform_random_crop_384": {"water_hit_probability": 0.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset = SimpleNamespace(
        records=[SimpleNamespace(sample_id="DC_a"), SimpleNamespace(sample_id="DC_b")]
    )

    with pytest.raises(ValueError, match="does not exactly match"):
        MODULE.gamus_fine_class_sampling_weights(
            dataset,
            {
                "fine_class_sampling_index_path": path,
                "fine_class_sampling_index_sha256": digest,
            },
        )


def test_direct_height_loss_reaches_every_intended_trainable_component() -> None:
    base = HeightNet(
        backbone="resnet18",
        pretrained=False,
        decoder_channels=(64, 32, 16, 8, 4),
        auxiliary_building_head=True,
    )
    model = MODULE.DomainGatedSurfaceNet(
        base,
        hidden_channels=8,
        fusion_mode="legacy",
        freeze_base=True,
    )
    MODULE.configure_explicit_parameter_groups(
        model,
        {
            "freeze_base_epochs": 0,
            "parameter_groups": [
                {
                    "name": "final_encoder_stage",
                    "prefixes": ["base_model.encoder.layer4."],
                    "learning_rate": 5.0e-6,
                },
                {
                    "name": "metric_decoder",
                    "prefixes": [
                        "base_model.deepest.",
                        "base_model.decoder_stages.",
                        "base_model.height_head.",
                    ],
                    "learning_rate": 3.0e-5,
                },
                {
                    "name": "surface_adapter",
                    "prefixes": [
                        "adapter.",
                        "domain_head.",
                        "building_residual_head.",
                        "canopy_height_head.",
                        "refinement_strength_head.",
                    ],
                    "learning_rate": 1.0e-4,
                },
            ],
        },
    )
    # The real warm start has learned, nonzero heads. Mirror that condition so
    # this test exercises the adapter path rather than the fresh model's
    # deliberately zero-initialized head weights.
    with torch.no_grad():
        for head in (
            model.domain_head,
            model.building_residual_head,
            model.canopy_height_head,
            model.refinement_strength_head,
        ):
            head.weight.fill_(0.01)
    model.eval()
    image = torch.randn(2, 3, 32, 32)
    prior = torch.rand(2, 1, 32, 32)
    output = model(image, prior)
    target = torch.full_like(output["height"], 12.0)
    valid = torch.ones_like(target, dtype=torch.bool)
    domain = torch.zeros(2, 32, 32, dtype=torch.long)
    criterion = MODULE.MultiDomainSurfaceLoss(
        height_weight=1.0,
        semantic_weight=0.0,
        building_weight=0.0,
        canopy_weight=0.0,
        ground_suppression_weight=0.0,
    )

    loss, _ = criterion(output, target, valid, domain, valid[:, 0])
    loss.backward()

    parameters = dict(model.named_parameters())
    gradient_prefixes = (
        "base_model.encoder.layer4.",
        "base_model.deepest.",
        "base_model.decoder_stages.",
        "base_model.height_head.",
        "adapter.",
        "domain_head.",
        "building_residual_head.",
        "canopy_height_head.",
        "refinement_strength_head.",
    )
    for prefix in gradient_prefixes:
        gradients = [
            parameter.grad
            for name, parameter in parameters.items()
            if name.startswith(prefix)
        ]
        assert gradients
        assert any(
            gradient is not None and torch.count_nonzero(gradient).item() > 0
            for gradient in gradients
        ), prefix
    assert parameters["base_model.encoder.conv1.weight"].grad is None
    assert parameters["base_model.building_head.weight"].grad is None
    assert parameters["log_variance_head.weight"].grad is None


def test_warmup_cosine_scheduler_uses_optimizer_steps() -> None:
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW((parameter,), lr=1.0)
    scheduler, interval = MODULE.make_learning_rate_scheduler(
        optimizer,
        {
            "epochs": 2,
            "gradient_accumulation": 2,
            "learning_rate_schedule": "warmup_cosine_steps",
            "warmup_optimizer_steps": 2,
            "minimum_learning_rate_factor": 0.1,
        },
        batches_per_epoch=4,
    )

    assert interval == "optimizer_step"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    observed = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])
    assert observed[0] == pytest.approx(1.0)
    assert observed[-1] == pytest.approx(0.1)


def test_partial_gradient_accumulation_window_uses_its_actual_size() -> None:
    assert [
        MODULE.gradient_accumulation_window_size(step, 6, 4)
        for step in range(1, 7)
    ] == [4, 4, 4, 4, 2, 2]
    assert MODULE.gradient_accumulation_window_size(2501, 2501, 4) == 1


def test_protected_baseline_fusion_override_is_temporary() -> None:
    model = _ToySurfaceModel()
    model.fusion_mode = "legacy"
    model.vegetation_expert_fusion_strength = 0.0

    with MODULE.temporary_fusion_evaluation_overrides(
        model,
        {
            "fusion_mode": "protected_vegetation",
            "vegetation_expert_fusion_strength": 0.1,
        },
    ):
        assert model.fusion_mode == "protected_vegetation"
        assert model.vegetation_expert_fusion_strength == pytest.approx(0.1)

    assert model.fusion_mode == "legacy"
    assert model.vegetation_expert_fusion_strength == pytest.approx(0.0)


def test_validation_guards_support_predeclared_minimum_improvement() -> None:
    baseline = {"gamus": {"rmse_m": 5.0, "macro_f1": 0.70}}
    current = {"gamus": {"rmse_m": 4.75, "macro_f1": 0.72}}
    config = {
        "validation_guards": {
            "gamus": {
                "rmse_m": {
                    "min_improvement": 0.20,
                    "baseline": "initial",
                },
                "macro_f1": {"min_gain": 0.01, "baseline": "initial"},
            }
        }
    }

    details, eligible = MODULE.validation_guard_eligibility(
        current, baseline, config
    )

    assert eligible is True
    assert details["suites"]["gamus"]["guards"]["rmse_m"]["delta"] == pytest.approx(
        -0.25
    )


def test_gamus_validation_rejects_tile_leakage_and_incomplete_release(
    tmp_path: Path,
) -> None:
    for split in ("train", "val", "test"):
        _triplet(tmp_path, split, "DC_01_01" if split != "test" else "DC_02_01")
    _, datasets = _datasets(tmp_path)

    with pytest.raises(ValueError, match="tile leakage between train and val"):
        MODULE.validate_gamus_splits(
            datasets, require_complete_official_splits=False
        )

    # Once the duplicate is replaced, a deliberate subset passes split leakage
    # checks but cannot accidentally masquerade as the complete official release.
    for suffix in ("RGB", "AGL", "CLS"):
        category = {"RGB": "images", "AGL": "heights", "CLS": "classes"}[suffix]
        old = tmp_path / category / "val" / f"DC_01_01_{suffix}.h5"
        new = tmp_path / category / "val" / f"DC_03_01_{suffix}.h5"
        old.rename(new)
    _, datasets = _datasets(tmp_path)
    MODULE.validate_gamus_splits(
        datasets, require_complete_official_splits=False
    )
    with pytest.raises(ValueError, match="Incomplete GAMUS official split counts"):
        MODULE.validate_gamus_splits(
            datasets, require_complete_official_splits=True
        )


def _validation_metrics(
    *,
    overall: float,
    landscapes: dict[str, float],
    domains: dict[str, float] | None = None,
    building_expert: float = 4.0,
    canopy_expert: float = 5.0,
    building_router_error: float = 0.25,
) -> dict[str, object]:
    domain_values = domains or {
        "ground": 1.0,
        "building": 2.0,
        "vegetation": 3.0,
    }
    return {
        "rmse_m": overall,
        "landscapes": {
            name: {"rmse_m": value} for name, value in landscapes.items()
        },
        "domains": {
            name: {"rmse_m": value} for name, value in domain_values.items()
        },
        "experts": {
            "building": {"rmse_m": building_expert},
            "vegetation": {"rmse_m": canopy_expert},
        },
        "building_expert_rmse_m": building_expert,
        "canopy_expert_rmse_m": canopy_expert,
        "building_router_error": building_router_error,
    }


def test_initial_checkpoint_metrics_are_recomputed_only_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _validation_metrics(overall=5.0, landscapes={"urban": 5.0})
    recomputed = _validation_metrics(overall=7.0, landscapes={"mixed": 7.0})
    calls: list[str] = []

    def fake_validate(*args, **kwargs):
        calls.append(kwargs["precision"])
        return {}, recomputed

    monkeypatch.setattr(MODULE, "validate", fake_validate)
    metrics, source = MODULE.resolve_initial_checkpoint_metrics(
        object(),
        object(),
        object(),
        object(),
        precision="bf16",
        stored_metrics=stored,
        recompute_on_current_validation=False,
    )
    assert metrics is stored
    assert source == "stored_checkpoint"
    assert calls == []

    metrics, source = MODULE.resolve_initial_checkpoint_metrics(
        object(),
        object(),
        object(),
        object(),
        precision="bf16",
        stored_metrics=stored,
        recompute_on_current_validation=True,
    )
    assert metrics is recomputed
    assert source == "current_validation"
    assert calls == ["bf16"]


def test_gamus_warm_start_requires_current_validation_baseline() -> None:
    with pytest.raises(ValueError, match="GAMUS warm start"):
        MODULE.validate_initial_checkpoint_metric_strategy(
            dataset_kind="gamus",
            initial_checkpoint="checkpoint.pt",
            recompute_on_current_validation=False,
        )

    MODULE.validate_initial_checkpoint_metric_strategy(
        dataset_kind="gamus",
        initial_checkpoint="checkpoint.pt",
        recompute_on_current_validation=True,
    )


def test_six_class_training_configuration_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="fine_semantic_classes"):
        MODULE.validate_six_class_training_config(
            dataset_kind="gamus",
            model_config={},
            training_config={"fine_semantic_weight": 0.25},
        )
    with pytest.raises(ValueError, match="data.dataset: gamus"):
        MODULE.validate_six_class_training_config(
            dataset_kind="manifest",
            model_config={"fine_semantic_classes": 6},
            training_config={"fine_semantic_weight": 0.25},
        )
    with pytest.raises(ValueError, match="fine_semantic_head"):
        MODULE.validate_six_class_training_config(
            dataset_kind="gamus",
            model_config={"fine_semantic_classes": 6},
            training_config={
                "fine_semantic_weight": 0.25,
                "parameter_groups": [
                    {"prefixes": ["adapter."]},
                ],
            },
        )

    MODULE.validate_six_class_training_config(
        dataset_kind="gamus",
        model_config={"fine_semantic_classes": 6},
        training_config={
            "fine_semantic_weight": 0.25,
            "fine_semantic_class_weights": [1.0] * 6,
            "parameter_groups": [
                {"prefixes": ["adapter.", "fine_semantic_head."]},
            ],
        },
    )


def test_initial_checkpoint_guard_compares_native_mixed_metrics() -> None:
    baseline = _validation_metrics(
        overall=5.0,
        landscapes={"mixed": 6.0},
        domains={"ground": 1.0, "building": 2.0, "vegetation": 3.0},
    )
    current = _validation_metrics(
        overall=5.02,
        landscapes={"mixed": 6.03},
        domains={"ground": 1.01, "building": 2.02, "vegetation": 2.98},
    )

    regressions, passes = MODULE.initial_checkpoint_regressions(
        current,
        baseline,
        {
            "max_initial_overall_rmse_regression_m": 0.05,
            "max_initial_mixed_rmse_regression_m": 0.05,
            "max_initial_ground_rmse_regression_m": 0.05,
            "max_initial_building_rmse_regression_m": 0.01,
            "max_initial_vegetation_rmse_regression_m": 0.05,
        },
    )

    assert set(regressions) == {
        "overall",
        "mixed",
        "ground",
        "building",
        "vegetation",
    }
    assert regressions["mixed"] == pytest.approx(0.03)
    assert regressions["building"] == pytest.approx(0.02)
    assert passes is False


def test_initial_checkpoint_guard_rejects_cross_dataset_landscape_metrics() -> None:
    current = _validation_metrics(overall=5.0, landscapes={"mixed": 5.0})
    stored = _validation_metrics(
        overall=5.0,
        landscapes={"urban": 5.0, "forest": 5.0},
    )

    with pytest.raises(
        ValueError,
        match="recompute_initial_metrics_on_current_validation",
    ):
        MODULE.initial_checkpoint_regressions(current, stored, {})


def test_recomputed_baseline_initializes_exact_configured_selection_score() -> None:
    baseline = _validation_metrics(
        overall=5.0,
        landscapes={"mixed": 6.0},
        domains={"ground": 1.0, "building": 2.0, "vegetation": 3.0},
        building_expert=4.0,
    )
    baseline["landscape_macro_rmse_m"] = 6.0
    config = {
        "primary_metric": "landscape_macro_rmse_m",
        "building_selection_weight": 0.5,
        "vegetation_selection_weight": 0.25,
        "building_expert_selection_weight": 0.1,
    }

    expected = 6.0 + 0.5 * 2.0 + 0.25 * 3.0 + 0.1 * 4.0
    assert MODULE.selection_score(baseline, config) == pytest.approx(expected)
    assert MODULE.resolve_initial_best_selection(
        baseline, "current_validation", config
    ) == pytest.approx(expected)

    # Historical stored-metric warm starts retain their old infinity default,
    # and an explicit legacy baseline continues to take precedence.
    assert np.isinf(
        MODULE.resolve_initial_best_selection(baseline, "stored_checkpoint", config)
    )
    assert MODULE.resolve_initial_best_selection(
        baseline,
        "current_validation",
        {**config, "initial_selection_value_m": 123.4},
    ) == pytest.approx(123.4)


def test_same_baseline_specialist_guards_are_opt_in_and_zero_regression_safe() -> None:
    baseline = _validation_metrics(
        overall=5.0,
        landscapes={"mixed": 6.0},
        building_expert=4.0,
        canopy_expert=5.0,
        building_router_error=0.25,
    )
    current = _validation_metrics(
        overall=5.0,
        landscapes={"mixed": 6.0},
        building_expert=4.02,
        canopy_expert=4.98,
        building_router_error=0.255,
    )

    product_only, product_passes = MODULE.initial_checkpoint_regressions(
        current, baseline, {}
    )
    assert set(product_only) == {
        "overall",
        "mixed",
        "ground",
        "building",
        "vegetation",
    }
    assert product_passes is True

    regressions, passes = MODULE.initial_checkpoint_regressions(
        current,
        baseline,
        {
            "max_initial_building_expert_rmse_regression_m": 0.0,
            "max_initial_canopy_expert_rmse_regression_m": 0.0,
            "max_initial_building_router_error_regression": 0.0,
        },
    )
    assert regressions["building_expert"] == pytest.approx(0.02)
    assert regressions["canopy_expert"] == pytest.approx(-0.02)
    assert regressions["building_router_error"] == pytest.approx(0.005)
    assert passes is False


def test_legacy_explicit_building_expert_reference_still_takes_precedence() -> None:
    baseline = _validation_metrics(
        overall=5.0,
        landscapes={"mixed": 6.0},
        building_expert=7.245523,
    )
    current = _validation_metrics(
        overall=5.0,
        landscapes={"mixed": 6.0},
        building_expert=7.245523,
    )

    dynamic_regressions, dynamic_passes = MODULE.initial_checkpoint_regressions(
        current,
        baseline,
        {"max_initial_building_expert_rmse_regression_m": 0.001},
    )
    assert dynamic_regressions["building_expert"] == pytest.approx(0.0)
    assert dynamic_passes is True

    legacy_regressions, legacy_passes = MODULE.initial_checkpoint_regressions(
        current,
        baseline,
        {
            "initial_building_expert_rmse_m": 7.202152,
            "max_initial_building_expert_rmse_regression_m": 0.001,
        },
    )
    assert legacy_regressions["building_expert"] == pytest.approx(0.043371)
    assert legacy_passes is False
