import hashlib

import numpy as np
import pytest
import torch

from msr.inference.predict import load_predictor, predict_height
from msr.models.domain_surface_net import DomainGatedSurfaceNet
from msr.models.height_net import HeightNet
from msr.models.residual_height import ResidualHeightSurfaceNet


def _checkpoint(tmp_path):
    base_config = dict(backbone="resnet18", pretrained=False,
                       decoder_channels=(32, 16, 8, 4, 4), auxiliary_building_head=True)
    base = HeightNet(**base_config).eval()
    base_path = tmp_path / "base.pt"
    torch.save(dict(model=base.state_dict(), config={"model": base_config}), base_path)
    protected_config = dict(base_checkpoint=str(base_path), hidden_channels=8, fine_semantic_classes=6)
    protected = DomainGatedSurfaceNet(base, hidden_channels=8, fine_semantic_classes=6).eval()
    protected_path = tmp_path / "protected.pt"
    torch.save(dict(model_type="domain_gated_surface_v4_six_class", model=protected.state_dict(),
                    config={"model": protected_config}), protected_path)
    candidate = ResidualHeightSurfaceNet(protected, hidden_channels=8).eval()
    config = dict(protected_checkpoint=str(protected_path),
                  protected_checkpoint_sha256=hashlib.sha256(protected_path.read_bytes()).hexdigest(),
                  hidden_channels=8, maximum_correction_m=80.0)
    path = tmp_path / "candidate.pt"
    payload = dict(model_type=candidate.model_type, model=candidate.state_dict(),
                   config={"model": config}, validation={"test_fixture": True})
    torch.save(payload, path)
    return path, candidate, payload


def test_residual_checkpoint_loads_and_tiled_prior_is_forwarded(tmp_path):
    path, candidate, _ = _checkpoint(tmp_path)
    loaded, metadata = load_predictor(path, device="cpu")
    assert isinstance(loaded, ResidualHeightSurfaceNet)
    assert metadata["fine_semantic_classes"] is not None
    assert metadata["validation_metrics"] == {"test_fixture": True}
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key])
    result = predict_height(np.full((3, 48, 48), 128, dtype=np.uint8),
                            model=loaded, device="cpu", amp=False, tile_size=64, overlap=16,
                            relative_prior=np.full((48, 48), 0.5, dtype=np.float32))
    assert result.height_map.shape == (48, 48)
    assert np.isfinite(result.height_map).all()
    assert result.fine_semantic_probability_maps.shape == (6, 48, 48)


def test_residual_loader_rejects_changed_inherited_tensor(tmp_path):
    path, _, payload = _checkpoint(tmp_path)
    key = next(key for key, value in payload["model"].items()
               if key.startswith("protected.") and value.is_floating_point())
    payload["model"][key] = payload["model"][key] + 0.01
    torch.save(payload, path)
    with pytest.raises(ValueError, match="altered protected tensor"):
        load_predictor(path, device="cpu")
