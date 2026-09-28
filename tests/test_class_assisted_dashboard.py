import hashlib
import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / 'scripts/class_assisted_dashboard.py'
spec = importlib.util.spec_from_file_location('height_dashboard', MODULE)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def validation(building=10.0, canopy=8.0):
    return {'groups': {dashboard.GROUPS['Buildings']: {'rmse_m': building},
                       dashboard.GROUPS['Canopy']: {'rmse_m': canopy}}}


def fixture(tmp_path):
    run = tmp_path / 'run'
    registration = tmp_path / 'active.json'
    put(registration, {'run_dir': str(run), 'pid': 123})
    put(run / 'status.json', {'stage': 'training', 'arm': 'uniform', 'epoch': 1, 'completed': 250, 'total': 600})
    put(run / 'config.json', {'guards': {'improvement_m': .5, 'improvement_fraction': .05}})
    put(run / 'baseline.json', validation())
    put(run / 'baseline_binding.json', {'sha256': hashlib.sha256((run / 'baseline.json').read_bytes()).hexdigest()})
    return run, registration


def test_uncommitted_evaluation_is_not_displayed(tmp_path):
    run, registration = fixture(tmp_path)
    put(run / 'uniform/epoch_01_evaluation.json', {'not': 'committed'})
    put(run / 'uniform/commit.json', {'epoch': 1, 'batch_done': 200, 'stage': 'trained'})
    result = dashboard.snapshot(registration)
    assert result['completed_epochs'] == 0
    assert result['progress']['uniform'] == 250
    assert len(result['rows']) == 1


def test_committed_results_separate_safety_from_benefit(tmp_path):
    run, registration = fixture(tmp_path)
    for arm, scores in [('uniform', validation(9.5, 7.8)), ('predicted', validation(9.0, 7.9))]:
        path = run / arm / 'epoch_01_evaluation.json'
        put(path, {'arm': arm, 'epoch': 1, 'validation': scores,
                   'guard': {'legacy_safety_pass': False, 'failed_checks': ['ground/rmse_m']}})
        put(run / arm / 'commit.json', {'epoch': 1, 'batch_done': 600, 'stage': 'evaluated',
                                        'evaluation_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    result = dashboard.snapshot(registration)
    assert result['completed_epochs'] == 2
    assert result['rows'][-1]['benefit'] == 'PASS'
    assert result['rows'][-1]['safety'] == 'FAIL'
    assert 'NOT certified' in dashboard.report_text(result)


def test_benefit_needs_same_epoch_control_and_both_thresholds():
    guards = {'improvement_m': .5, 'improvement_fraction': .05}
    assert dashboard.paired_review(validation(), None, validation(9, 7), guards) == 'PENDING'
    assert dashboard.paired_review(validation(), validation(8, 6), validation(9, 7), guards) == 'FAIL'
    assert dashboard.paired_review(validation(), validation(10, 8), validation(9.6, 8), guards) == 'FAIL'


def test_missing_values_are_not_zero_accuracy():
    assert dashboard.number(None) == 'N/A'
    assert dashboard.metric({}, 'missing') is None


def test_four_epoch_sampling_run_uses_correct_labels_and_budget(tmp_path):
    run,registration=fixture(tmp_path)
    config=dashboard.read_json(run/'config.json')
    config.update({'epochs':4,'pairs_per_epoch':600,'arms':['control','balanced'],
        'arm_labels':{'control':'A: original sampling','balanced':'B: height-balanced'},
        'comparison_summary':'Only sampling differs. Uniform classes in both.', 'conditioning':'uniform_both_arms'})
    put(run/'config.json',config)
    put(run/'status.json',{'stage':'training','arm':'balanced','epoch':3,'completed':200,'total':600})
    result=dashboard.snapshot(registration)
    assert result['progress']['balanced']==1400
    assert 'Only sampling differs' in dashboard.report_text(result)
    assert '0/8' in dashboard.report_text(result)
    assert 'predicted six-class probabilities' not in dashboard.report_text(result)
