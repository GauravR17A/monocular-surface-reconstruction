import pytest
import torch

from msr.training.balanced_height_loss import SourceBalancedHeightLoss


def _batch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    domain: torch.Tensor,
    *,
    source=("gamus",),
    landscape=("mixed",),
):
    return {"height": prediction}, {
        "image": torch.zeros(prediction.shape[0], 3, *prediction.shape[-2:]),
        "height": target,
        "regression_mask": mask,
        "domain_target": domain,
        "source": source,
        "landscape": landscape,
    }


def test_missing_labels_are_ignored_and_receive_no_gradient() -> None:
    prediction = torch.tensor([[[[1.0, 90.0]]]], requires_grad=True)
    target = torch.tensor([[[[3.0, float("nan")]]]])
    mask = torch.tensor([[[[True, False]]]])
    domain = torch.tensor([[[0, 99]]])
    output, batch = _batch(prediction, target, mask, domain)

    loss, _ = SourceBalancedHeightLoss(mse_weight=0.0)(output, batch)
    loss.backward()

    assert loss.item() == pytest.approx(2.0)  # Huber(error=2, delta=2)
    assert prediction.grad is not None
    assert prediction.grad[0, 0, 0, 0].item() != 0.0
    assert prediction.grad[0, 0, 0, 1].item() == 0.0


def test_duplicate_pixels_do_not_change_a_sample_or_source_mean() -> None:
    criterion = SourceBalancedHeightLoss(huber_delta_m=2.0, mse_weight=0.02)
    prediction = torch.tensor([[[[1.0, 1.0]]]])
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    domain = torch.zeros((1, 1, 2), dtype=torch.long)
    first, _ = criterion(*_batch(prediction, target, mask, domain))

    duplicated_prediction = prediction.repeat(1, 1, 1, 4)
    duplicated_target = target.repeat(1, 1, 1, 4)
    duplicated_mask = mask.repeat(1, 1, 1, 4)
    duplicated_domain = domain.repeat(1, 1, 4)
    second, _ = criterion(
        *_batch(
            duplicated_prediction,
            duplicated_target,
            duplicated_mask,
            duplicated_domain,
        )
    )

    torch.testing.assert_close(first, second)


def test_sources_domains_and_samples_are_reduced_with_equal_weight() -> None:
    # GAMUS has two ground samples with losses 0 and 2; its sample mean is 1.
    # With quadratic Huber here, HighBuild's error 4 gives loss 8.
    # OpenCanopy's errors 6 and 10 give domain losses 18 and 50, so mean 34.
    # Overall source-balanced mean is therefore (1 + 8 + 34) / 3.
    prediction = torch.tensor(
        [
            [[[0.0, 0.0]]],
            [[[2.0, 2.0]]],
            [[[4.0, 4.0]]],
            [[[6.0, 10.0]]],
        ],
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    domain = torch.tensor([[[0, 0]], [[0, 0]], [[1, 1]], [[0, 2]]])
    output, batch = _batch(
        prediction,
        target,
        mask,
        domain,
        source=("gamus", "gamus", "legacy", "legacy"),
        landscape=("mixed", "mixed", "urban", "forest"),
    )

    loss, components = SourceBalancedHeightLoss(
        huber_delta_m=100.0, mse_weight=0.0
    )(output, batch)

    # With a large Huber delta the per-pixel loss is 0.5 * error^2.
    assert components["gamus"].item() == pytest.approx(1.0)
    assert components["highbuild"].item() == pytest.approx(8.0)
    assert components["open_canopy"].item() == pytest.approx(34.0)
    assert loss.item() == pytest.approx((1.0 + 8.0 + 34.0) / 3.0)


def test_all_invalid_is_zero_and_differentiable() -> None:
    prediction = torch.tensor([[[[float("nan"), 5.0]]]], requires_grad=True)
    target = torch.tensor([[[[float("nan"), float("inf")]]]])
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    domain = torch.full((1, 1, 2), 99, dtype=torch.long)
    output, batch = _batch(prediction, target, mask, domain)

    loss, components = SourceBalancedHeightLoss()(output, batch)
    loss.backward()

    assert loss.item() == 0.0
    assert components["total"].item() == 0.0
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad).item() == 0


def test_nonfinite_target_is_rejected_only_when_valid() -> None:
    prediction = torch.zeros((1, 1, 1, 2), requires_grad=True)
    target = torch.tensor([[[[0.0, float("nan")]]]])
    domain = torch.zeros((1, 1, 2), dtype=torch.long)

    valid_nan = torch.tensor([[[[False, True]]]])
    with pytest.raises(ValueError, match="Non-finite target"):
        SourceBalancedHeightLoss()(*_batch(prediction, target, valid_nan, domain))

    invalid_nan = torch.tensor([[[[True, False]]]])
    loss, _ = SourceBalancedHeightLoss()(
        *_batch(prediction, target, invalid_nan, domain)
    )
    assert torch.isfinite(loss)


def test_unknown_legacy_landscape_fails_closed() -> None:
    prediction = torch.zeros((1, 1, 1, 1))
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    domain = torch.zeros((1, 1, 1), dtype=torch.long)

    with pytest.raises(ValueError, match="urban.*forest"):
        SourceBalancedHeightLoss()(
            *_batch(
                prediction,
                target,
                mask,
                domain,
                source=("legacy",),
                landscape=("hilly",),
            )
        )
