import importlib.util
from pathlib import Path

import pytest
import torch

from msr.models.domain_surface_net import DomainGatedSurfaceNet
from msr.models.height_net import HeightNet
from msr.training.losses import MultiDomainSurfaceLoss


ROOT = Path(__file__).parents[1]
TRAINER_PATH = ROOT / "scripts" / "train_multidomain.py"
TRAINER_SPEC = importlib.util.spec_from_file_location(
    "hierarchical_v4_trainer_test", TRAINER_PATH
)
assert TRAINER_SPEC and TRAINER_SPEC.loader
TRAINER = importlib.util.module_from_spec(TRAINER_SPEC)
TRAINER_SPEC.loader.exec_module(TRAINER)


def _surface_model() -> DomainGatedSurfaceNet:
    base = HeightNet(
        backbone="resnet18",
        pretrained=False,
        decoder_channels=(64, 32, 16, 8, 4),
        auxiliary_building_head=True,
    )
    return DomainGatedSurfaceNet(
        base,
        hidden_channels=8,
        fine_semantic_classes=6,
        fine_semantic_head_type="hierarchical_vegetation",
        freeze_base=True,
    )


def _classification_only_criterion() -> MultiDomainSurfaceLoss:
    return MultiDomainSurfaceLoss(
        height_weight=0.0,
        semantic_weight=0.0,
        fine_semantic_weight=1.0,
        fine_semantic_vegetation_split_weight=0.5,
        building_weight=0.0,
        canopy_weight=0.0,
        ground_suppression_weight=0.0,
    )


def test_vegetation_split_loss_uses_only_valid_low_vegetation_and_tree_pixels() -> None:
    model = _surface_model().eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32), torch.rand(1, 1, 32, 32))
    split_logits = torch.full((1, 1, 32, 32), 20.0, requires_grad=True)
    split_logits.data[:, :, :2, :2] = 0.0
    output = {**output, "fine_semantic_vegetation_split_logits": split_logits}

    fine_target = torch.full((1, 32, 32), 255, dtype=torch.long)
    fine_target[:, 0, :2] = 4
    fine_target[:, 1, :2] = 5
    classification_valid = fine_target != 255
    image_valid = torch.ones(1, 1, 32, 32, dtype=torch.bool)
    image_valid[:, :, 1, 1] = False

    total, components = _classification_only_criterion()(
        output,
        torch.zeros(1, 1, 32, 32),
        torch.zeros(1, 1, 32, 32, dtype=torch.bool),
        torch.zeros(1, 32, 32, dtype=torch.long),
        torch.ones(1, 32, 32, dtype=torch.bool),
        fine_class_target=fine_target,
        classification_valid_mask=classification_valid,
        image_valid_mask=image_valid,
    )
    total.backward()

    assert components["fine_semantic_vegetation_split"].item() == pytest.approx(
        torch.log(torch.tensor(2.0)).item()
    )
    assert split_logits.grad is not None
    assert torch.count_nonzero(split_logits.grad[:, :, :2, :2]) == 3
    assert torch.count_nonzero(split_logits.grad[:, :, 2:, :]) == 0
    assert split_logits.grad[0, 0, 1, 1] == 0


def test_vegetation_split_loss_never_requires_height_validity() -> None:
    model = _surface_model().eval()
    output = model(torch.randn(1, 3, 32, 32), torch.rand(1, 1, 32, 32))
    fine_target = torch.full((1, 32, 32), 4, dtype=torch.long)

    total, components = _classification_only_criterion()(
        output,
        torch.zeros(1, 1, 32, 32),
        torch.zeros(1, 1, 32, 32, dtype=torch.bool),
        torch.zeros(1, 32, 32, dtype=torch.long),
        torch.ones(1, 32, 32, dtype=torch.bool),
        fine_class_target=fine_target,
        classification_valid_mask=torch.ones_like(fine_target, dtype=torch.bool),
        image_valid_mask=torch.ones(1, 1, 32, 32, dtype=torch.bool),
    )

    assert torch.isfinite(total)
    assert torch.isfinite(components["fine_semantic_vegetation_split"])


def test_hierarchical_training_configuration_is_fail_closed() -> None:
    common = {
        "fine_semantic_weight": 1.0,
        "fine_semantic_class_weights": [1.0] * 6,
        "parameter_groups": [{"prefixes": ["fine_semantic_head."]}],
    }
    with pytest.raises(ValueError, match="requires a positive"):
        TRAINER.validate_six_class_training_config(
            dataset_kind="gamus",
            model_config={
                "fine_semantic_classes": 6,
                "fine_semantic_head_type": "hierarchical_vegetation",
            },
            training_config=common,
        )
    with pytest.raises(ValueError, match="only valid"):
        TRAINER.validate_six_class_training_config(
            dataset_kind="gamus",
            model_config={
                "fine_semantic_classes": 6,
                "fine_semantic_head_type": "spatial_refined",
            },
            training_config={
                **common,
                "fine_semantic_vegetation_split_weight": 0.5,
            },
        )

    TRAINER.validate_six_class_training_config(
        dataset_kind="gamus",
        model_config={
            "fine_semantic_classes": 6,
            "fine_semantic_head_type": "hierarchical_vegetation",
        },
        training_config={
            **common,
            "fine_semantic_vegetation_split_weight": 0.5,
        },
    )


def test_split_loss_weight_rejects_partial_configuration() -> None:
    with pytest.raises(ValueError, match="requires a positive fine_semantic_weight"):
        MultiDomainSurfaceLoss(fine_semantic_vegetation_split_weight=0.5)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        MultiDomainSurfaceLoss(fine_semantic_vegetation_split_weight=-0.1)
