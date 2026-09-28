import numpy as np
import torch

from msr.inference.predict import load_predictor, predict_height
from msr.models.domain_surface_net import DomainGatedSurfaceNet
from msr.models.height_net import HeightNet


class ConstantHeightModel(torch.nn.Module):
    def forward(self, image):
        batch, _, height, width = image.shape
        return {
            "height": torch.full(
                (batch, 1, height, width), 7.5, device=image.device
            )
        }


def test_tiled_prediction_blends_to_constant_without_seams():
    image = np.full((3, 83, 91), 128, dtype=np.uint8)

    result = predict_height(
        image,
        model=ConstantHeightModel(),
        device="cpu",
        tile_size=32,
        overlap=8,
        amp=False,
    )

    assert result.height_map.shape == (83, 91)
    assert np.allclose(result.height_map, 7.5)
    assert result.building_probability_map is None
    assert result.vegetation_probability_map is None
    assert result.fine_semantic_probability_maps is None
    assert result.valid_mask.all()


class ConstantBuildingModel(torch.nn.Module):
    def forward(self, image):
        batch, _, height, width = image.shape
        return {
            "height": torch.ones((batch, 1, height, width), device=image.device),
            "building_logits": torch.zeros((batch, 1, height, width), device=image.device),
        }


def test_tiled_prediction_returns_building_probability() -> None:
    result = predict_height(
        np.full((3, 37, 41), 128, dtype=np.uint8),
        model=ConstantBuildingModel(),
        device="cpu",
        tile_size=24,
        overlap=8,
        amp=False,
    )
    assert result.building_probability_map is not None
    assert np.allclose(result.building_probability_map, 0.5)


class ProtectedBuildingModel(torch.nn.Module):
    def forward(self, image):
        batch, _, height, width = image.shape
        shape = (batch, 1, height, width)
        return {
            "height": torch.ones(shape, device=image.device),
            "building_logits": torch.full(shape, -8.0, device=image.device),
            "protected_building_logits": torch.full(shape, 2.0, device=image.device),
        }


def test_tiled_prediction_prefers_protected_building_expert() -> None:
    result = predict_height(
        np.full((3, 37, 41), 128, dtype=np.uint8),
        model=ProtectedBuildingModel(),
        device="cpu",
        tile_size=24,
        overlap=8,
        amp=False,
    )
    assert result.building_probability_map is not None
    assert np.allclose(result.building_probability_map, torch.sigmoid(torch.tensor(2.0)))


class ConstantSurfaceModel(torch.nn.Module):
    def forward(self, image):
        batch, _, height, width = image.shape
        return {
            "height": torch.full((batch, 1, height, width), 3.0, device=image.device),
            "vegetation_logits": torch.zeros(
                (batch, 1, height, width), device=image.device
            ),
            "log_variance": torch.zeros(
                (batch, 1, height, width), device=image.device
            ),
        }


def test_tiled_prediction_returns_vegetation_and_confidence() -> None:
    result = predict_height(
        np.full((3, 31, 35), 128, dtype=np.uint8),
        model=ConstantSurfaceModel(),
        device="cpu",
        tile_size=24,
        overlap=8,
        amp=False,
    )
    assert result.vegetation_probability_map is not None
    assert result.confidence_map is not None
    assert np.allclose(result.vegetation_probability_map, 0.5)
    assert np.allclose(result.confidence_map, 0.5)


class ConstantFineSemanticModel(torch.nn.Module):
    def forward(self, image):
        batch, _, height, width = image.shape
        logits = torch.zeros((batch, 6, height, width), device=image.device)
        logits[:, 2] = 2.0
        return {
            "height": torch.ones((batch, 1, height, width), device=image.device),
            "fine_semantic_probabilities": logits.softmax(dim=1),
        }


def test_tiled_prediction_returns_six_class_probability_maps() -> None:
    result = predict_height(
        np.full((3, 31, 35), 128, dtype=np.uint8),
        model=ConstantFineSemanticModel(),
        device="cpu",
        tile_size=24,
        overlap=8,
        amp=False,
    )

    assert result.fine_semantic_probability_maps is not None
    assert result.fine_semantic_probability_maps.shape == (6, 31, 35)
    np.testing.assert_allclose(
        result.fine_semantic_probability_maps.sum(axis=0), 1.0, atol=1e-6
    )
    assert np.all(
        result.fine_semantic_probability_maps[2]
        > result.fine_semantic_probability_maps[0]
    )


def test_load_predictor_reconstructs_refined_six_class_head(tmp_path) -> None:
    base = HeightNet(
        backbone="resnet18",
        pretrained=False,
        decoder_channels=(64, 32, 16, 8, 4),
        auxiliary_building_head=True,
    )
    base_path = tmp_path / "base.pt"
    torch.save(
        {
            "model_type": "height_net_v1",
            "epoch": 2,
            "config": {
                "model": {
                    "backbone": "resnet18",
                    "decoder_channels": [64, 32, 16, 8, 4],
                    "auxiliary_building_head": True,
                }
            },
            "model": base.state_dict(),
        },
        base_path,
    )
    refined = DomainGatedSurfaceNet(
        base,
        hidden_channels=8,
        fine_semantic_classes=6,
        fine_semantic_head_type="spatial_refined",
        freeze_base=True,
    )
    refined_path = tmp_path / "refined.pt"
    torch.save(
        {
            "model_type": "domain_gated_surface_v4_six_class",
            "epoch": 3,
            "config": {
                "model": {
                    "base_checkpoint": str(base_path),
                    "hidden_channels": 8,
                    "fine_semantic_classes": 6,
                    "fine_semantic_head_type": "spatial_refined",
                }
            },
            "model": refined.state_dict(),
        },
        refined_path,
    )

    loaded, metadata = load_predictor(refined_path, device="cpu")

    assert isinstance(loaded, DomainGatedSurfaceNet)
    assert loaded.fine_semantic_head_type == "spatial_refined"
    assert metadata["fine_semantic_head_type"] == "spatial_refined"
    assert metadata["fine_semantic_classes"] == list(
        DomainGatedSurfaceNet.fine_semantic_names
    )
    assert loaded.state_dict().keys() == refined.state_dict().keys()
    for name, expected in refined.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], expected)


def test_relative_prior_must_align_with_rgb() -> None:
    try:
        predict_height(
            np.zeros((3, 20, 20), dtype=np.uint8),
            model=ConstantHeightModel(),
            relative_prior=np.zeros((19, 20), dtype=np.float32),
            device="cpu",
            amp=False,
        )
    except ValueError as error:
        assert "relative_prior" in str(error)
    else:
        raise AssertionError("Expected mismatched relative prior to be rejected")


class PriorEchoSurfaceModel(DomainGatedSurfaceNet):
    """Minimal domain-model stub that exposes prior contamination in tiling."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, image, relative_prior):
        assert relative_prior is not None
        return {"height": relative_prior}


def test_relative_prior_nodata_does_not_poison_overlapping_tiles() -> None:
    image = np.full((3, 48, 80), 128, dtype=np.uint8)
    prior = np.full((48, 80), 2.0, dtype=np.float32)
    prior[:, -4:] = np.nan
    source_valid = np.ones((48, 80), dtype=bool)
    source_valid[:, -4:] = False

    result = predict_height(
        image,
        model=PriorEchoSurfaceModel(),
        relative_prior=prior,
        valid_mask=source_valid,
        device="cpu",
        tile_size=32,
        overlap=8,
        amp=False,
    )

    assert np.isfinite(result.height_map[:, :-4]).all()
    assert np.allclose(result.height_map[:, :-4], 2.0)
    assert np.isnan(result.height_map[:, -4:]).all()
    np.testing.assert_array_equal(result.valid_mask, source_valid)


def test_valid_mask_must_align_with_rgb() -> None:
    try:
        predict_height(
            np.zeros((3, 20, 20), dtype=np.uint8),
            model=ConstantHeightModel(),
            valid_mask=np.ones((20, 19), dtype=bool),
            device="cpu",
            amp=False,
        )
    except ValueError as error:
        assert "valid_mask" in str(error)
    else:
        raise AssertionError("Expected mismatched valid mask to be rejected")
