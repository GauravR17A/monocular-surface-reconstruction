"""Predeclared development-only gates for the two-source matched height pilot."""
import math


def safety_checks(baseline, candidate, limits):
    failures=[]
    if baseline.get('support_sha256')!=candidate.get('support_sha256'):
        failures.append('support_identity')
    if baseline.get('semantic_sha256')!=candidate.get('semantic_sha256'):
        failures.append('protected_semantic_identity')
    old,new=baseline['groups'],candidate['groups']
    if set(old)!=set(new): failures.append('group_identity')
    for key in sorted(set(old)&set(new)):
        a,b=old[key],new[key]
        if a['pixel_count']!=b['pixel_count']: failures.append(key+'/count')
        for metric,limit in [('rmse_m',limits['rmse_m']),('mae_m',limits['mae_m']),('bias_m',limits['absolute_bias_m'])]:
            x,y=a[metric],b[metric]
            delta=abs(y)-abs(x) if metric=='bias_m' else y-x
            if not math.isfinite(x) or not math.isfinite(y) or delta>limit: failures.append(key+'/'+metric)
        for metric in ['correlation','r2']:
            x,y=a.get(metric),b.get(metric)
            if x is not None and math.isfinite(x) and (y is None or not math.isfinite(y) or y-x < -limits['correlation_r2_tolerance']):
                failures.append(key+'/'+metric)
    return {'legacy_safety_pass':not failures,'failed_checks':failures,
            'scope':'HighBuild and OpenCanopy development only; GAMUS not evaluated; no release approval'}


def paired_benefit(baseline, control, candidate, limits):
    priorities={}
    for key in ['highbuild/domain/building','open_canopy/domain/vegetation']:
        old=baseline['groups'][key]['rmse_m']; a=control['groups'][key]['rmse_m']; b=candidate['groups'][key]['rmse_m']
        gain=old-b
        priorities[key]={'baseline_rmse_m':old,'control_rmse_m':a,'candidate_rmse_m':b,'improvement_m':gain,
            'passes':bool(old>0 and gain>=limits['improvement_m'] and gain/old>=limits['improvement_fraction'] and b<a)}
    return {'benefit_pass':any(v['passes'] for v in priorities.values()),'priorities':priorities,'release_eligible':False}
