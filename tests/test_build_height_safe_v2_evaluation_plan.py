from __future__ import annotations

import ast
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_height_safe_v2_evaluation_plan.py"
)


def test_plan_builder_is_validation_only_and_fail_closed() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert tree is not None
    assert 'contracts.get("official_test_used") is not False' in source
    assert 'evaluation.get("promotion_eligible") is not False' in source
    assert '"official_gamus_test_constructed": False' in source
    assert '"production_promotion_permitted": False' in source
    assert "Refusing to overwrite evaluation evidence" in source
