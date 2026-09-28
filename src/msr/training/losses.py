"""Masked robust regression and structure-preserving losses."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    loss = F.huber_loss(prediction, target, reduction="none", delta=delta)
    return _masked_mean(loss, valid_mask)


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    return _masked_mean(torch.square(prediction - target), valid_mask)


def masked_gradient_l1(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    mask_dx = valid_mask[..., :, 1:] & valid_mask[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    mask_dy = valid_mask[..., 1:, :] & valid_mask[..., :-1, :]
    return 0.5 * (
        _masked_mean(torch.abs(pred_dx - target_dx), mask_dx)
        + _masked_mean(torch.abs(pred_dy - target_dy), mask_dy)
    )


def _batch_label_mask(
    labels: Sequence[str],
    expected: str,
    *,
    batch_size: int,
    device: torch.device,
    argument_name: str,
) -> torch.Tensor:
    if isinstance(labels, str) or len(labels) != batch_size:
        raise ValueError(
            f"{argument_name} must contain one string per batch item "
            f"(expected {batch_size})"
        )
    if any(not isinstance(label, str) for label in labels):
        raise TypeError(f"{argument_name} must contain only strings")
    return torch.tensor(
        [label.strip().lower() == expected for label in labels],
        dtype=torch.bool,
        device=device,
    )


class FrozenDomainHeadTeacher(nn.Module):
    """Immutable Stage-1 domain head used for low-cost logit distillation.

    The adapter and protected-base features are frozen during semantic repair,
    so copying only the small domain head avoids a second full-model forward.
    ``prior_domain_logits`` must be the fixed prior emitted by the student
    model, making the returned logits directly comparable to ``domain_logits``.
    """

    def __init__(self, domain_head: nn.Module) -> None:
        super().__init__()
        self.domain_head = deepcopy(domain_head)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # A teacher snapshot must remain deterministic even when a containing
        # module is switched into training mode.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(
        self,
        adapter_features: torch.Tensor,
        prior_domain_logits: torch.Tensor,
    ) -> torch.Tensor:
        residual_logits = self.domain_head(adapter_features.detach())
        if residual_logits.shape != prior_domain_logits.shape:
            raise ValueError(
                "prior_domain_logits must match the frozen domain-head output shape"
            )
        return (residual_logits + prior_domain_logits.detach()).detach()


class Stage2SemanticRepairLoss(nn.Module):
    """Source-aware semantic-only repair objective for mixed replay.

    GAMUS contributes class-balanced semantic cross entropy and Stage-1 logit
    distillation. Legacy samples contribute unweighted cross entropy plus a
    small fused-height guard. Only non-building pixels from legacy *urban*
    samples are downweighted; forest replay retains full ground supervision.

    This loss controls the objective, not parameter selection. The caller must
    freeze every model parameter except ``domain_head`` for Stage 2.
    """

    def __init__(
        self,
        *,
        semantic_weight: float = 1.0,
        gamus_semantic_weight: float = 0.75,
        legacy_semantic_weight: float = 0.25,
        gamus_class_weights: tuple[float, float, float] | list[float] = (
            1.0,
            1.5,
            0.75,
        ),
        legacy_class_weights: tuple[float, float, float] | list[float] = (
            1.0,
            1.0,
            1.0,
        ),
        legacy_urban_ground_pixel_weight: float = 0.35,
        distillation_temperature: float = 2.0,
        gamus_distillation_weight: float = 0.20,
        legacy_fused_height_weight: float = 0.025,
        huber_delta_m: float = 2.0,
    ) -> None:
        super().__init__()
        if (
            semantic_weight < 0
            or gamus_semantic_weight < 0
            or legacy_semantic_weight < 0
        ):
            raise ValueError("semantic loss weights must be nonnegative")
        if legacy_urban_ground_pixel_weight <= 0:
            raise ValueError("legacy_urban_ground_pixel_weight must be positive")
        if distillation_temperature <= 0:
            raise ValueError("distillation_temperature must be positive")
        if gamus_distillation_weight < 0 or legacy_fused_height_weight < 0:
            raise ValueError("Stage-2 auxiliary loss weights must be nonnegative")
        if huber_delta_m <= 0:
            raise ValueError("huber_delta_m must be positive")

        gamus_weights = torch.tensor(gamus_class_weights, dtype=torch.float32)
        legacy_weights = torch.tensor(legacy_class_weights, dtype=torch.float32)
        for name, weights in (
            ("gamus_class_weights", gamus_weights),
            ("legacy_class_weights", legacy_weights),
        ):
            if weights.shape != (3,) or torch.any(weights <= 0):
                raise ValueError(f"{name} must contain three positive values")

        self.semantic_weight = float(semantic_weight)
        self.gamus_semantic_weight = float(gamus_semantic_weight)
        self.legacy_semantic_weight = float(legacy_semantic_weight)
        self.legacy_urban_ground_pixel_weight = float(
            legacy_urban_ground_pixel_weight
        )
        self.distillation_temperature = float(distillation_temperature)
        self.gamus_distillation_weight = float(gamus_distillation_weight)
        self.legacy_fused_height_weight = float(legacy_fused_height_weight)
        self.huber_delta_m = float(huber_delta_m)
        self.register_buffer("gamus_class_weights", gamus_weights)
        self.register_buffer("legacy_class_weights", legacy_weights)

    def forward(
        self,
        output: dict[str, torch.Tensor],
        domain_target: torch.Tensor,
        domain_valid_mask: torch.Tensor,
        batch_sources: Sequence[str],
        *,
        batch_landscapes: Sequence[str] | None = None,
        teacher_domain_logits: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        regression_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        student_logits = output["domain_logits"]
        if student_logits.ndim != 4 or student_logits.shape[1] != 3:
            raise ValueError("output['domain_logits'] must have shape (batch, 3, H, W)")
        batch_size, _, height, width = student_logits.shape
        if domain_target.shape != (batch_size, height, width):
            raise ValueError("domain_target must match the domain-logit spatial shape")
        if domain_valid_mask.shape != domain_target.shape:
            raise ValueError("domain_valid_mask must match domain_target")

        gamus_samples = _batch_label_mask(
            batch_sources,
            "gamus",
            batch_size=batch_size,
            device=student_logits.device,
            argument_name="batch_sources",
        )
        legacy_samples = _batch_label_mask(
            batch_sources,
            "legacy",
            batch_size=batch_size,
            device=student_logits.device,
            argument_name="batch_sources",
        )
        if not torch.all(gamus_samples | legacy_samples):
            unknown = sorted(
                {
                    str(source)
                    for source in batch_sources
                    if str(source).strip().lower() not in {"gamus", "legacy"}
                }
            )
            raise ValueError(f"Unsupported Stage-2 batch source(s): {unknown}")
        if batch_landscapes is None and bool(torch.any(legacy_samples)):
            raise ValueError(
                "batch_landscapes is required to downweight only legacy urban ground pixels"
            )
        urban_samples = torch.zeros_like(legacy_samples)
        if batch_landscapes is not None:
            urban_samples = _batch_label_mask(
                batch_landscapes,
                "urban",
                batch_size=batch_size,
                device=student_logits.device,
                argument_name="batch_landscapes",
            )

        domain_valid = domain_valid_mask.bool()
        gamus_valid = domain_valid & gamus_samples[:, None, None]
        legacy_valid = domain_valid & legacy_samples[:, None, None]
        gamus_ce = F.cross_entropy(
            student_logits,
            domain_target.long(),
            weight=self.gamus_class_weights,
            reduction="none",
        )
        legacy_ce = F.cross_entropy(
            student_logits,
            domain_target.long(),
            weight=self.legacy_class_weights,
            reduction="none",
        )
        semantic_values = torch.where(
            gamus_samples[:, None, None], gamus_ce, legacy_ce
        )
        legacy_urban_ground = (
            legacy_samples[:, None, None]
            & urban_samples[:, None, None]
            & (domain_target == 0)
        )
        semantic_pixel_weights = torch.ones_like(semantic_values)
        semantic_pixel_weights = torch.where(
            legacy_urban_ground,
            semantic_pixel_weights.new_tensor(
                self.legacy_urban_ground_pixel_weight
            ),
            semantic_pixel_weights,
        )
        weighted_semantic_values = semantic_values * semantic_pixel_weights
        gamus_semantic = _masked_mean(weighted_semantic_values, gamus_valid)
        legacy_semantic = _masked_mean(weighted_semantic_values, legacy_valid)
        # Normalize within each source before applying the deliberate replay
        # mixture, so nodata prevalence cannot silently alter the 3:1 objective.
        semantic = (
            self.gamus_semantic_weight * gamus_semantic
            + self.legacy_semantic_weight * legacy_semantic
        )

        zero = student_logits.sum() * 0.0
        gamus_distillation = zero
        if self.gamus_distillation_weight > 0 and bool(torch.any(gamus_valid)):
            if teacher_domain_logits is None:
                raise ValueError(
                    "teacher_domain_logits is required for GAMUS Stage-2 distillation"
                )
            if teacher_domain_logits.shape != student_logits.shape:
                raise ValueError(
                    "teacher_domain_logits must match output['domain_logits']"
                )
            temperature = self.distillation_temperature
            distillation_values = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_domain_logits.detach() / temperature, dim=1),
                reduction="none",
            ).sum(dim=1)
            gamus_distillation = temperature**2 * _masked_mean(
                distillation_values, gamus_valid
            )

        legacy_fused_height = zero
        if self.legacy_fused_height_weight > 0 and bool(torch.any(legacy_samples)):
            if target is None or regression_mask is None:
                raise ValueError(
                    "target and regression_mask are required for the legacy fused-height guard"
                )
            fused_height = output.get("height")
            if fused_height is None:
                raise KeyError("output['height'] is required for the legacy height guard")
            if target.shape != fused_height.shape:
                raise ValueError("target must match output['height']")
            if regression_mask.shape != target.shape:
                raise ValueError("regression_mask must match target")
            legacy_regression_valid = (
                regression_mask.bool() & legacy_samples[:, None, None, None]
            )
            legacy_fused_height = masked_huber(
                fused_height,
                target,
                legacy_regression_valid,
                delta=self.huber_delta_m,
            )

        total = (
            self.semantic_weight * semantic
            + self.gamus_distillation_weight * gamus_distillation
            + self.legacy_fused_height_weight * legacy_fused_height
        )
        components = {
            "semantic": semantic,
            "gamus_semantic": gamus_semantic,
            "legacy_semantic": legacy_semantic,
            "gamus_distillation": gamus_distillation,
            "legacy_fused_height": legacy_fused_height,
            "total": total,
        }
        return total, components


class CompositeHeightLoss(nn.Module):
    def __init__(
        self,
        *,
        huber_delta_m: float = 1.0,
        gradient_weight: float = 0.1,
        foreground_regression_weight: float = 1.0,
        foreground_mse_weight: float = 0.0,
        tall_regression_weight: float = 0.0,
        tall_threshold_m: float = 20.0,
        building_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.huber_delta_m = huber_delta_m
        self.gradient_weight = gradient_weight
        self.foreground_regression_weight = foreground_regression_weight
        self.foreground_mse_weight = foreground_mse_weight
        self.tall_regression_weight = tall_regression_weight
        self.tall_threshold_m = tall_threshold_m
        self.building_weight = building_weight

    def forward(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        building_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        height = output["height"]
        regression = masked_huber(
            height, target, valid_mask, delta=self.huber_delta_m
        )
        foreground = masked_huber(
            height,
            target,
            valid_mask & building_mask.bool() if building_mask is not None else valid_mask,
            delta=self.huber_delta_m,
        )
        gradient = masked_gradient_l1(height, target, valid_mask)
        total = (
            regression
            + self.foreground_regression_weight * foreground
            + self.gradient_weight * gradient
        )
        components = {
            "height": regression,
            "foreground_height": foreground,
            "gradient": gradient,
        }

        if building_mask is not None and self.foreground_mse_weight > 0.0:
            foreground_mse = masked_mse(
                height, target, valid_mask & building_mask.bool()
            )
            total = total + self.foreground_mse_weight * foreground_mse
            components["foreground_mse"] = foreground_mse
        if self.tall_regression_weight > 0.0:
            tall = masked_huber(
                height,
                target,
                valid_mask & (target >= self.tall_threshold_m),
                delta=self.huber_delta_m,
            )
            total = total + self.tall_regression_weight * tall
            components["tall_height"] = tall

        if building_mask is not None and "building_logits" in output:
            binary = F.binary_cross_entropy_with_logits(
                output["building_logits"], building_mask, reduction="none"
            )
            building = _masked_mean(binary, valid_mask)
            total = total + self.building_weight * building
            components["building"] = building
        components["total"] = total
        return total, components


class MultiDomainSurfaceLoss(nn.Module):
    """Balance full-surface regression with explicit semantic expert targets."""

    def __init__(
        self,
        *,
        huber_delta_m: float = 2.0,
        height_weight: float = 1.0,
        semantic_weight: float = 0.3,
        fine_semantic_weight: float = 0.0,
        fine_semantic_vegetation_split_weight: float = 0.0,
        building_weight: float = 1.0,
        building_mse_weight: float = 0.0,
        tall_building_weight: float = 0.0,
        tall_building_threshold_m: float = 20.0,
        canopy_weight: float = 1.0,
        canopy_mse_weight: float = 0.0,
        tall_canopy_weight: float = 0.0,
        tall_canopy_threshold_m: float = 15.0,
        fused_building_weight: float = 0.0,
        fused_building_mse_weight: float = 0.0,
        fused_vegetation_weight: float = 0.0,
        fused_tall_building_weight: float = 0.0,
        fused_tall_vegetation_weight: float = 0.0,
        building_distillation_weight: float = 0.0,
        ground_suppression_weight: float = 0.25,
        refinement_weight: float = 0.0,
        semantic_class_weights: tuple[float, float, float] | list[float] | None = None,
        fine_semantic_class_weights: Sequence[float] | None = None,
        fine_semantic_focal_gamma: float = 0.0,
        uncertainty_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.huber_delta_m = huber_delta_m
        self.height_weight = height_weight
        self.semantic_weight = semantic_weight
        if fine_semantic_weight < 0:
            raise ValueError("fine_semantic_weight must be nonnegative")
        self.fine_semantic_weight = float(fine_semantic_weight)
        if (
            not math.isfinite(fine_semantic_vegetation_split_weight)
            or fine_semantic_vegetation_split_weight < 0
        ):
            raise ValueError(
                "fine_semantic_vegetation_split_weight must be finite and nonnegative"
            )
        if fine_semantic_vegetation_split_weight > 0 and fine_semantic_weight <= 0:
            raise ValueError(
                "fine_semantic_vegetation_split_weight requires a positive "
                "fine_semantic_weight"
            )
        self.fine_semantic_vegetation_split_weight = float(
            fine_semantic_vegetation_split_weight
        )
        if not math.isfinite(fine_semantic_focal_gamma) or fine_semantic_focal_gamma < 0:
            raise ValueError("fine_semantic_focal_gamma must be finite and nonnegative")
        self.fine_semantic_focal_gamma = float(fine_semantic_focal_gamma)
        self.building_weight = building_weight
        self.building_mse_weight = building_mse_weight
        self.tall_building_weight = tall_building_weight
        self.tall_building_threshold_m = tall_building_threshold_m
        self.canopy_weight = canopy_weight
        self.canopy_mse_weight = canopy_mse_weight
        self.tall_canopy_weight = tall_canopy_weight
        self.tall_canopy_threshold_m = tall_canopy_threshold_m
        self.fused_building_weight = fused_building_weight
        self.fused_building_mse_weight = fused_building_mse_weight
        self.fused_vegetation_weight = fused_vegetation_weight
        self.fused_tall_building_weight = fused_tall_building_weight
        self.fused_tall_vegetation_weight = fused_tall_vegetation_weight
        self.building_distillation_weight = building_distillation_weight
        self.ground_suppression_weight = ground_suppression_weight
        self.refinement_weight = refinement_weight
        self.uncertainty_weight = uncertainty_weight
        weights = (
            torch.tensor(semantic_class_weights, dtype=torch.float32)
            if semantic_class_weights is not None
            else None
        )
        if weights is not None and (weights.shape != (3,) or torch.any(weights <= 0)):
            raise ValueError("semantic_class_weights must contain three positive values")
        self.register_buffer("semantic_class_weights", weights)
        fine_weights = (
            torch.tensor(fine_semantic_class_weights, dtype=torch.float32)
            if fine_semantic_class_weights is not None
            else None
        )
        if fine_weights is not None and (
            fine_weights.shape != (6,) or torch.any(fine_weights <= 0)
        ):
            raise ValueError(
                "fine_semantic_class_weights must contain six positive values"
            )
        self.register_buffer("fine_semantic_class_weights", fine_weights)

    def forward(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
        regression_mask: torch.Tensor,
        domain_target: torch.Tensor,
        domain_valid_mask: torch.Tensor,
        *,
        fine_class_target: torch.Tensor | None = None,
        classification_valid_mask: torch.Tensor | None = None,
        image_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if domain_target.ndim != 3:
            raise ValueError("domain_target must have shape (batch, height, width)")
        domain_valid = domain_valid_mask.bool()
        regression_valid = regression_mask.bool()
        final_height = masked_huber(
            output["height"], target, regression_valid, delta=self.huber_delta_m
        )
        semantic_values = F.cross_entropy(
            output["domain_logits"],
            domain_target.long(),
            weight=self.semantic_class_weights,
            reduction="none",
        )
        semantic = _masked_mean(semantic_values, domain_valid)

        urban = regression_valid & (domain_target[:, None] == 1)
        forest = regression_valid & (domain_target[:, None] == 2)
        ground = regression_valid & (domain_target[:, None] == 0)
        building = masked_huber(
            output["building_height"], target, urban, delta=self.huber_delta_m
        )
        canopy = masked_huber(
            output["canopy_height"], target, forest, delta=self.huber_delta_m
        )
        ground_suppression = _masked_mean(output["height"], ground)
        # The protected urban expert should remain dominant except where the
        # LiDAR labels explicitly identify vegetation. This directly teaches
        # the otherwise-saturated safety gate when it is allowed to open.
        vegetation_target = (domain_target[:, None] == 2).to(
            output["refinement_strength"].dtype
        )
        refinement_values = F.binary_cross_entropy_with_logits(
            output["refinement_logits"], vegetation_target, reduction="none"
        )
        refinement = _masked_mean(refinement_values, domain_valid[:, None])

        total = (
            self.height_weight * final_height
            + self.semantic_weight * semantic
            + self.building_weight * building
            + self.canopy_weight * canopy
            + self.ground_suppression_weight * ground_suppression
            + self.refinement_weight * refinement
        )
        components = {
            "height": final_height,
            "semantic": semantic,
            "building_height": building,
            "canopy_height": canopy,
            "ground_suppression": ground_suppression,
            "refinement": refinement,
        }
        if self.fine_semantic_weight > 0.0:
            fine_logits = output.get("fine_semantic_logits")
            if fine_logits is None:
                raise KeyError(
                    "fine_semantic_weight requires output['fine_semantic_logits']"
                )
            if fine_logits.ndim != 4 or fine_logits.shape[1] != 6:
                raise ValueError(
                    "output['fine_semantic_logits'] must have shape (batch, 6, H, W)"
                )
            expected_shape = (
                fine_logits.shape[0],
                fine_logits.shape[2],
                fine_logits.shape[3],
            )
            if (
                fine_class_target is None
                or classification_valid_mask is None
                or image_valid_mask is None
            ):
                raise ValueError(
                    "fine_class_target, classification_valid_mask, and "
                    "image_valid_mask are required when fine_semantic_weight is positive"
                )
            if fine_class_target.shape != expected_shape:
                raise ValueError(
                    "fine_class_target must match the six-class logit spatial shape"
                )
            if classification_valid_mask.shape != expected_shape:
                raise ValueError(
                    "classification_valid_mask must match fine_class_target"
                )
            expected_image_shape = (
                expected_shape[0],
                1,
                expected_shape[1],
                expected_shape[2],
            )
            if image_valid_mask.shape == expected_image_shape:
                image_valid = image_valid_mask[:, 0].bool()
            elif image_valid_mask.shape == expected_shape:
                image_valid = image_valid_mask.bool()
            else:
                raise ValueError(
                    "image_valid_mask must match fine_class_target, with an optional "
                    "singleton channel"
                )
            fine_valid = classification_valid_mask.bool() & image_valid
            if bool(
                torch.any(
                    fine_valid
                    & ((fine_class_target < 0) | (fine_class_target >= 6))
                )
            ):
                raise ValueError(
                    "valid fine_class_target values must be within [0, 5]"
                )
            safe_fine_target = fine_class_target.long().masked_fill(~fine_valid, 0)
            fine_values = F.cross_entropy(
                fine_logits,
                safe_fine_target,
                weight=self.fine_semantic_class_weights,
                reduction="none",
            )
            if self.fine_semantic_focal_gamma > 0.0:
                # Down-weight already-easy pixels while retaining the configured
                # class weights.  This is especially useful for rare GAMUS
                # categories such as water, which an ordinary mean CE can
                # nearly ignore behind abundant vegetation and ground pixels.
                target_probability = fine_logits.softmax(dim=1).gather(
                    1, safe_fine_target[:, None]
                )[:, 0]
                fine_values = fine_values * (
                    1.0 - target_probability
                ).pow(self.fine_semantic_focal_gamma)
            fine_semantic = _masked_mean(fine_values, fine_valid)
            total = total + self.fine_semantic_weight * fine_semantic
            components["fine_semantic"] = fine_semantic
            if self.fine_semantic_vegetation_split_weight > 0.0:
                split_logits = output.get(
                    "fine_semantic_vegetation_split_logits"
                )
                if split_logits is None:
                    raise KeyError(
                        "fine_semantic_vegetation_split_weight requires "
                        "output['fine_semantic_vegetation_split_logits']"
                    )
                if split_logits.shape != (
                    expected_shape[0],
                    1,
                    expected_shape[1],
                    expected_shape[2],
                ):
                    raise ValueError(
                        "output['fine_semantic_vegetation_split_logits'] must "
                        "have shape (batch, 1, H, W)"
                    )
                vegetation_valid = fine_valid & (
                    (fine_class_target == 4) | (fine_class_target == 5)
                )
                tree_target = (fine_class_target == 5).to(split_logits.dtype)
                split_values = F.binary_cross_entropy_with_logits(
                    split_logits[:, 0], tree_target, reduction="none"
                )
                vegetation_split = _masked_mean(
                    split_values, vegetation_valid
                )
                total = total + (
                    self.fine_semantic_vegetation_split_weight
                    * vegetation_split
                )
                components["fine_semantic_vegetation_split"] = vegetation_split
        if self.canopy_mse_weight > 0.0:
            canopy_mse = masked_mse(output["canopy_height"], target, forest)
            total = total + self.canopy_mse_weight * canopy_mse
            components["canopy_mse"] = canopy_mse
        if self.tall_canopy_weight > 0.0:
            tall_canopy = masked_huber(
                output["canopy_height"],
                target,
                forest & (target >= self.tall_canopy_threshold_m),
                delta=self.huber_delta_m,
            )
            total = total + self.tall_canopy_weight * tall_canopy
            components["tall_canopy_height"] = tall_canopy
        if self.building_mse_weight > 0.0:
            building_mse = masked_mse(output["building_height"], target, urban)
            total = total + self.building_mse_weight * building_mse
            components["building_mse"] = building_mse
        if self.tall_building_weight > 0.0:
            tall_building = masked_huber(
                output["building_height"],
                target,
                urban & (target >= self.tall_building_threshold_m),
                delta=self.huber_delta_m,
            )
            total = total + self.tall_building_weight * tall_building
            components["tall_building_height"] = tall_building
        if self.fused_building_weight > 0.0:
            fused_building = masked_huber(
                output["height"], target, urban, delta=self.huber_delta_m
            )
            total = total + self.fused_building_weight * fused_building
            components["fused_building_height"] = fused_building
        if self.fused_building_mse_weight > 0.0:
            fused_building_mse = masked_mse(output["height"], target, urban)
            total = total + self.fused_building_mse_weight * fused_building_mse
            components["fused_building_mse"] = fused_building_mse
        if self.fused_vegetation_weight > 0.0:
            fused_vegetation = masked_huber(
                output["height"], target, forest, delta=self.huber_delta_m
            )
            total = total + self.fused_vegetation_weight * fused_vegetation
            components["fused_vegetation_height"] = fused_vegetation
        if self.fused_tall_building_weight > 0.0:
            fused_tall_building = masked_huber(
                output["height"],
                target,
                urban & (target >= self.tall_building_threshold_m),
                delta=self.huber_delta_m,
            )
            total = total + (
                self.fused_tall_building_weight * fused_tall_building
            )
            components["fused_tall_building_height"] = fused_tall_building
        if self.fused_tall_vegetation_weight > 0.0:
            fused_tall_vegetation = masked_huber(
                output["height"],
                target,
                forest & (target >= self.tall_canopy_threshold_m),
                delta=self.huber_delta_m,
            )
            total = total + (
                self.fused_tall_vegetation_weight * fused_tall_vegetation
            )
            components["fused_tall_vegetation_height"] = fused_tall_vegetation
        if self.building_distillation_weight > 0.0:
            building_distillation = masked_huber(
                output["height"],
                output["base_height"].detach(),
                urban,
                delta=self.huber_delta_m,
            )
            total = total + (
                self.building_distillation_weight * building_distillation
            )
            components["building_distillation"] = building_distillation
        if self.uncertainty_weight > 0.0:
            squared_error = torch.square(output["height"] - target)
            heteroscedastic = 0.5 * (
                torch.exp(-output["log_variance"]) * squared_error
                + output["log_variance"]
            )
            uncertainty = _masked_mean(heteroscedastic, regression_valid)
            total = total + self.uncertainty_weight * uncertainty
            components["uncertainty"] = uncertainty
        components["total"] = total
        return total, components
