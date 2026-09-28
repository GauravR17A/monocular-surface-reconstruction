import importlib.util
from pathlib import Path
import sys
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("oem_stage", SCRIPTS / "stage_christchurch_rgb_holdout.py")
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


@pytest.mark.parametrize("name", [
    "../images/christchurch_1.tif", "OpenEarthMap_wo_xBD/christchurch/images/../bad.tif",
    "OpenEarthMap_wo_xBD/other/images/christchurch_1.tif",
    "OpenEarthMap_wo_xBD/christchurch/images/christchurch_1.png",
    "OpenEarthMap_wo_xBD/christchurch/images/christchurch_1\\evil.tif",
])
def test_member_rejects_unexpected_or_escaping_paths(name):
    with pytest.raises(ValueError):
        stage.relative_member(name)


def test_valid_member_is_local_relative_path():
    assert stage.relative_member("OpenEarthMap_wo_xBD/christchurch/labels/christchurch_1.tif") == Path("labels/christchurch_1.tif")


def test_cached_read_keeps_file_position_and_reuses_block(monkeypatch):
    calls = []
    def underlying(self, size):
        calls.append((self.position, size))
        payload = bytes((self.position+i) % 256 for i in range(size))
        self.position += size
        return payload
    monkeypatch.setattr(stage.RemoteArchive, "read", underlying)
    remote = stage.CachedArchive()
    assert remote.read(3) == bytes([0, 1, 2])
    assert remote.tell() == 3
    remote.seek(2)
    assert remote.read(2) == bytes([2, 3])
    assert len(calls) == 1
    remote.seek(stage.SIZE)
    assert remote.read(3) == b""
