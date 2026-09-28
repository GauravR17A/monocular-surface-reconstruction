import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('low_surface_report',ROOT/'scripts/report_class_assisted_low_surface.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_serialized_copy_may_have_different_whitespace(tmp_path):
    source=tmp_path/'config.json'; source.write_text('{"epochs":2, "loss": 0.5}\n')
    saved=json.loads(json.dumps({'epochs':2,'loss':.5},indent=2))
    module.verify_config_copy(saved,source,module.sha(source))


def test_changed_source_or_semantic_value_is_rejected(tmp_path):
    source=tmp_path/'config.json'; source.write_text('{"epochs":2}\n')
    digest=module.sha(source)
    with pytest.raises(ValueError,match='configuration changed'):
        module.verify_config_copy({'epochs':3},source,digest)
    source.write_text('{"epochs":3}\n')
    with pytest.raises(ValueError,match='configuration changed'):
        module.verify_config_copy({'epochs':3},source,digest)
