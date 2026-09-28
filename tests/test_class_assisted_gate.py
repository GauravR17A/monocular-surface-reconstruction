from copy import deepcopy
from msr.evaluation.class_assisted_gate import safety_checks,paired_benefit

LIMITS={'rmse_m':.15,'mae_m':.10,'absolute_bias_m':.10,'correlation_r2_tolerance':1e-6,'improvement_m':.5,'improvement_fraction':.05}

def report(rmse=10):
    group={'pixel_count':10,'rmse_m':rmse,'mae_m':5,'bias_m':-1,'correlation':.5,'r2':.4}
    return {'support_sha256':'a','semantic_sha256':'b','groups':{k:dict(group) for k in ['highbuild/domain/building','open_canopy/domain/vegetation']}}


def test_benefit_needs_both_absolute_and_relative_gain_and_control_win():
    assert paired_benefit(report(),report(9.8),report(9.5),LIMITS)['benefit_pass']
    assert not paired_benefit(report(),report(9.5),report(9.5),LIMITS)['benefit_pass']
    assert not paired_benefit(report(20),report(20),report(19.4),LIMITS)['benefit_pass']
    assert not paired_benefit(report(4),report(4),report(3.7),LIMITS)['benefit_pass']


def test_guards_fail_on_geometry_identity_bias_or_correlation_regression():
    a=report(); assert safety_checks(a,deepcopy(a),LIMITS)['legacy_safety_pass']
    b=deepcopy(a); b['support_sha256']='changed'
    assert not safety_checks(a,b,LIMITS)['legacy_safety_pass']
    for key,value in [('bias_m',-1.2),('correlation',.49),('r2',None)]:
        b=deepcopy(a); b['groups']['highbuild/domain/building'][key]=value
        assert not safety_checks(a,b,LIMITS)['legacy_safety_pass']
