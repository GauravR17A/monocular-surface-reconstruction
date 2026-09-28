import pytest
import torch

from msr.training.losses import (
    CompositeHeightLoss,
    FrozenDomainHeadTeacher,
    Stage2SemanticRepairLoss,
    masked_gradient_l1,
    masked_huber,
    masked_mse,
)


def test_masked_huber_ignores_invalid_pixel():
    prediction = torch.tensor([[[[0.0, 100.0]]]])
    target = torch.tensor([[[[0.0, 0.0]]]])
    valid = torch.tensor([[[[True, False]]]])

    assert masked_huber(prediction, target, valid).item() == 0.0


def test_gradient_loss_rewards_matching_edges():
    target = torch.tensor([[[[0.0, 0.0], [10.0, 10.0]]]])
    valid = torch.ones_like(target, dtype=torch.bool)

    assert masked_gradient_l1(target.clone(), target, valid).item() == 0.0


def test_masked_mse_penalizes_large_error_quadratically():
    prediction = torch.tensor([[[[0.0, 10.0]]]])
    target = torch.zeros_like(prediction)
    valid = torch.ones_like(prediction, dtype=torch.bool)

    assert masked_mse(prediction, target, valid).item() == pytest.approx(50.0)


def test_composite_loss_backpropagates():
    height = torch.full((1, 1, 4, 4), 1.0, requires_grad=True)
    logits = torch.zeros_like(height, requires_grad=True)
    output = {"height": height, "building_logits": logits}
    target = torch.full_like(height, 2.0)
    valid = torch.ones_like(height, dtype=torch.bool)
    building = torch.ones_like(height)

    total, components = CompositeHeightLoss()(output, target, valid, building)
    total.backward()

    assert total.item() == pytest.approx(components["total"].item())
    assert height.grad is not None
    assert logits.grad is not None


def test_composite_loss_reports_tail_and_foreground_mse():
    height = torch.zeros((1, 1, 2, 2), requires_grad=True)
    output = {"height": height}
    target = torch.tensor([[[[0.0, 5.0], [25.0, 40.0]]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    building = target >= 2.0

    _, components = CompositeHeightLoss(
        foreground_mse_weight=0.03,
        tall_regression_weight=0.2,
        tall_threshold_m=20.0,
    )(output, target, valid, building)

    assert components["foreground_mse"].item() > 0.0
    assert components["tall_height"].item() > 0.0


def test_stage2_semantic_repair_applies_source_and_urban_pixel_weights():
    logits = torch.zeros((3, 3, 1, 1), requires_grad=True)
    output = {"domain_logits": logits}
    domain_target = torch.tensor([[[1]], [[0]], [[0]]])
    domain_valid = torch.ones_like(domain_target, dtype=torch.bool)
    criterion = Stage2SemanticRepairLoss(
        gamus_distillation_weight=0.0,
        legacy_fused_height_weight=0.0,
    )

    total, components = criterion(
        output,
        domain_target,
        domain_valid,
        ["gamus", "legacy", "legacy"],
        batch_landscapes=["mixed", "urban", "forest"],
    )

    log_three = torch.log(torch.tensor(3.0)).item()
    assert components["gamus_semantic"].item() == pytest.approx(1.5 * log_three)
    assert components["legacy_semantic"].item() == pytest.approx(
        0.5 * (0.35 + 1.0) * log_three
    )
    assert total.item() == pytest.approx(
        (0.75 * 1.5 + 0.25 * 0.5 * (0.35 + 1.0)) * log_three
    )


def test_stage2_distillation_uses_temperature_and_only_updates_student():
    student = torch.tensor(
        [[[[0.5]], [[-0.25]], [[1.0]]]], requires_grad=True
    )
    teacher = torch.tensor(
        [[[[2.0]], [[-1.0]], [[0.25]]]], requires_grad=True
    )
    criterion = Stage2SemanticRepairLoss(
        semantic_weight=0.0,
        distillation_temperature=2.0,
        gamus_distillation_weight=0.20,
        legacy_fused_height_weight=0.0,
    )

    total, components = criterion(
        {"domain_logits": student},
        torch.zeros((1, 1, 1), dtype=torch.long),
        torch.ones((1, 1, 1), dtype=torch.bool),
        ["gamus"],
        batch_landscapes=["mixed"],
        teacher_domain_logits=teacher,
    )
    total.backward()

    expected = 4.0 * torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student.detach() / 2.0, dim=1),
        torch.nn.functional.softmax(teacher.detach() / 2.0, dim=1),
        reduction="batchmean",
    )
    assert components["gamus_distillation"].item() == pytest.approx(expected.item())
    assert total.item() == pytest.approx(0.20 * expected.item())
    assert student.grad is not None
    assert teacher.grad is None


def test_stage2_legacy_height_guard_ignores_gamus_height_error():
    logits = torch.zeros((2, 3, 1, 1), requires_grad=True)
    height = torch.tensor([[[[100.0]]], [[[2.0]]]], requires_grad=True)
    target = torch.zeros_like(height)
    criterion = Stage2SemanticRepairLoss(
        semantic_weight=0.0,
        gamus_distillation_weight=0.0,
        legacy_fused_height_weight=0.025,
        huber_delta_m=2.0,
    )

    total, components = criterion(
        {"domain_logits": logits, "height": height},
        torch.zeros((2, 1, 1), dtype=torch.long),
        torch.zeros((2, 1, 1), dtype=torch.bool),
        ["gamus", "legacy"],
        batch_landscapes=["mixed", "urban"],
        target=target,
        regression_mask=torch.ones_like(target, dtype=torch.bool),
    )
    total.backward()

    assert components["legacy_fused_height"].item() == pytest.approx(2.0)
    assert total.item() == pytest.approx(0.05)
    assert height.grad is not None
    assert height.grad[0].item() == 0.0
    assert height.grad[1].item() > 0.0


def test_stage2_requires_landscape_tags_for_legacy_replay():
    criterion = Stage2SemanticRepairLoss(
        gamus_distillation_weight=0.0,
        legacy_fused_height_weight=0.0,
    )
    with pytest.raises(ValueError, match="batch_landscapes"):
        criterion(
            {"domain_logits": torch.zeros((1, 3, 1, 1))},
            torch.zeros((1, 1, 1), dtype=torch.long),
            torch.ones((1, 1, 1), dtype=torch.bool),
            ["legacy"],
        )


def test_frozen_domain_head_teacher_is_an_immutable_snapshot():
    source_head = torch.nn.Conv2d(2, 3, kernel_size=1)
    teacher = FrozenDomainHeadTeacher(source_head)
    features = torch.randn((1, 2, 2, 2), requires_grad=True)
    prior = torch.randn((1, 3, 2, 2), requires_grad=True)
    expected = teacher(features, prior)

    with torch.no_grad():
        source_head.weight.add_(100.0)
        source_head.bias.add_(100.0)
    actual = teacher(features, prior)

    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
