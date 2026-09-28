"""Public-package scope for checks bound to unbundled original experiment seals."""
import json
from pathlib import Path
import pytest

ARCHIVE_CHECKS = {(item["file"], item["test"]) for item in json.loads(
    (Path(__file__).parent / "historical_archive_checks.json").read_text())}

def pytest_addoption(parser):
    parser.addoption("--run-historical-seals", action="store_true", default=False,
                     help="Attempt exact historical source/experiment checks (requires original bound archive)")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-historical-seals"):
        return
    for item in items:
        path = item.path.relative_to(config.rootpath).as_posix()
        name = item.originalname or item.name.split("[")[0]
        if (path, name) in ARCHIVE_CHECKS:
            item.add_marker(pytest.mark.skip(reason=
                "Historical exact-source/experiment seal requires the original unbundled archive; "
                "renamed public source is not that sealed artifact. See docs/RELEASE.md."))
