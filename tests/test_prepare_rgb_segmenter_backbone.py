import importlib.util
import hashlib
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_rgb_segmenter_backbone.py"
    spec = importlib.util.spec_from_file_location("prepare_rgb_segmenter_backbone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_streaming_sha256_supports_project_python(tmp_path):
    module = load_module()
    path = tmp_path / "weights.bin"
    payload = b"backbone-fixture" * 150000
    path.write_bytes(payload)
    assert module.sha256(path) == hashlib.sha256(payload).hexdigest()


def test_backbone_download_is_pinned_and_on_data_volume():
    module = load_module()
    assert len(module.REVISION) == 40
    assert len(module.WEIGHT_SHA256) == 64
    assert str(module.DESTINATION).replace("\\", "/").startswith("D:/MSRData/models/")
    assert set(module.FILES) == {"config.json", "preprocessor_config.json", "pytorch_model.bin", "README.md"}
