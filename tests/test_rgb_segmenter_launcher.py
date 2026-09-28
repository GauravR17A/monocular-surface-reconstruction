from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_launcher_protects_production_and_requires_scoped_latest_resume():
    text = (ROOT / "scripts/run_rgb_segmenter_v1.ps1").read_text(encoding="utf-8")
    assert "Assert-ProductionProtected" in text
    assert "StartsWith($experimentRoot" in text
    assert '"checkpoint_latest.pt"' in text
    assert '"*_gamus_rgb_segmenter_v1"' in text
    assert "[System.IO.FileShare]::None" in text
    assert "Another trainer is active" in text
    assert "--preflight-only" in text
    assert "no app promotion" in text


def test_viewer_uses_explicit_active_pointer_and_nonblocking_file_sharing():
    text = (ROOT / "scripts/watch_rgb_segmenter_v1.ps1").read_text(encoding="utf-8")
    assert "rgb_segmenter_v1_active.json" in text
    assert "[System.IO.FileShare]::Delete" in text
    assert "Get-ChildItem" not in text
    assert "Stop-Process" not in text
