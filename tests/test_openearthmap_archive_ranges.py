import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("oem_ranges", Path(__file__).resolve().parents[1] / "scripts/inspect_openearthmap_archive.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_seek_and_empty_read():
    archive = module.RemoteArchive()
    assert archive.seek(-5, 2) == module.SIZE-5
    assert archive.seek(5, 1) == module.SIZE
    assert archive.read(10) == b""
    with pytest.raises(ValueError):
        archive.seek(-1)
    archive.seek(0)
    with pytest.raises(ValueError):
        archive.read()


@pytest.mark.parametrize("status,header", [(200, ""), (206, "bytes 1-3/4")])
def test_reject_full_download_or_wrong_range_without_reading(status, header):
    class Response:
        headers = {"Content-Range": header}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, *args): raise AssertionError("Must not read body")
    response = Response()
    response.status = status
    archive = module.RemoteArchive(lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match="exact range"):
        archive.read(3)


def test_exact_range_read():
    class Response:
        status = 206
        headers = {"Content-Range": f"bytes 0-2/{module.SIZE}"}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size): return b"abc"
    archive = module.RemoteArchive(lambda *args, **kwargs: Response())
    assert archive.read(3) == b"abc"
    assert archive.tell() == archive.transferred == 3
