import ctypes
import importlib.util
import os
from pathlib import Path
import threading

import pytest


SPEC = importlib.util.spec_from_file_location(
    "windows_atomic_replacement", Path(__file__).parents[1] / "scripts/train_residual_height.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.skipif(os.name != "nt", reason="Real Windows file-sharing semantics")
def test_status_commit_survives_real_windows_reader_lock(tmp_path, monkeypatch):
    source = tmp_path / "status.json.tmp"
    destination = tmp_path / "status.json"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                       ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    create.restype = ctypes.c_void_p
    close = kernel.CloseHandle
    close.argtypes = [ctypes.c_void_p]
    close.restype = ctypes.c_int
    # Share READ but not DELETE, matching a status reader that blocks os.replace.
    handle = create(str(destination), 0x80000000, 0x00000001, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value
    real_replace = os.replace
    attempts = []

    def replace(first, second):
        attempts.append(1)
        return real_replace(first, second)

    monkeypatch.setattr(MODULE.os, "replace", replace)
    release = threading.Timer(0.15, close, args=[handle])
    release.start()
    try:
        MODULE.atomic_replace(source, destination)
    finally:
        release.join(timeout=2)
    assert len(attempts) >= 2
    assert destination.read_text(encoding="utf-8") == "new"
    assert not source.exists()


@pytest.mark.parametrize("winerror,expected_attempts", [(5, 20), (32, 20), (33, 20), (None, 1)])
def test_permanent_or_unrelated_permission_error_is_not_swallowed(tmp_path, monkeypatch,
                                                               winerror, expected_attempts):
    source = tmp_path / "status.json.tmp"
    destination = tmp_path / "status.json"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")
    attempts = []

    def blocked(*_):
        attempts.append(1)
        error = PermissionError(13, "test lock or permission error")
        if winerror is not None:
            error.winerror = winerror
        raise error

    monkeypatch.setattr(MODULE.os, "replace", blocked)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        MODULE.atomic_replace(source, destination)
    assert len(attempts) == expected_attempts
    assert destination.read_text(encoding="utf-8") == "old"
    assert source.read_text(encoding="utf-8") == "new"
