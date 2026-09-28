"""Verify completed paired evidence and generate a concise non-promotional report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from class_assisted_dashboard import read_json
from msr.evaluation.class_assisted_gate import safety_checks, paired_benefit

ROOT=Path(__file__).resolve().parents[1]
GROUPS={
    'Buildings':'highbuild/domain/building', 'Canopy':'open_canopy/domain/vegetation',
    'Ground':'open_canopy/domain/ground', 'Short vegetation':'open_canopy/short_vegetation',
    'Tall buildings':'highbuild/tall/building','Tall canopy':'open_canopy/tall/vegetation'}


def resolve(value):
    path=Path(value)
    return path if path.is_absolute() else ROOT/path


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):
            digest.update(part)
    return digest.hexdigest()


def verify_config_copy(saved, source, expected_source_sha):
    # The trainer reserializes its config into the run directory. Authenticate
    # original bytes against binding, and the saved copy by JSON semantics.
    if sha(source) != expected_source_sha or read_json(source) != saved:
        raise ValueError('Saved configuration changed')


def verify_run(run):
    outcome=read_json(run/'outcome.json')
    if outcome['stage'] != 'complete' or read_json(run/'status.json')['stage'] != 'complete':
        raise ValueError('Completed run required')
    config,binding=read_json(run/'config.json'),read_json(run/'binding.json')
    verify_config_copy(config, ROOT/'configs/class_assisted_height_low_surface_v3.json', binding['config_sha256'])
    for name,digest in binding['code'].items():
        if sha(ROOT/name) != digest or sha(run/'source_snapshot'/name) != digest:
            raise ValueError(f'Source changed: {name}')
    baseline=read_json(run/'baseline.json')
    if sha(run/'baseline.json') != config['cache_bindings']['baseline.json']:
        raise ValueError('Baseline mismatch')
    if sha(run/'data_binding.json') != config['cache_bindings']['data_binding.json']:
        raise ValueError('Data mismatch')
    proof=read_json(resolve(config['proof_config']))
    protected={}
    for name in ('protected_checkpoint','protected_pointer','classifier_checkpoint'):
        protected[name]=sha(resolve(proof[name])) == proof[name+'_sha256']
    protected['v1_classifier_checkpoint']=sha(resolve(config['v1_classifier_checkpoint'])) == config['v1_classifier_checkpoint_sha256']
    if not all(protected.values()):
        raise ValueError('Protected artifact changed')
    results={}
    for arm in ('uniform','predicted'):
        commit=read_json(run/arm/'commit.json')
        if commit['epoch'] != 2 or commit['stage'] != 'evaluated':
            raise ValueError('Incomplete arm')
        for key in ('checkpoint','evaluation'):
            if sha(run/arm/commit[key]) != commit[key+'_sha256']:
                raise ValueError(f'Commit hash mismatch: {arm}/{key}')
        for epoch in (1,2):
            report=read_json(run/arm/f'epoch_{epoch:02}_evaluation.json')
            if report['arm'] != arm or report['epoch'] != epoch:
                raise ValueError('Report identity mismatch')
            recomputed=safety_checks(baseline,report['validation'],config['guards'])
            if recomputed != report['guard']:
                raise ValueError('Safety results inconsistent')
            results[(arm,epoch)]=report
    if read_json(run/'uniform/initial.json') != read_json(run/'predicted/initial.json'):
        raise ValueError('Initial weights not paired')
    for epoch in (1,2):
        if read_json(run/'uniform'/f'epoch_{epoch:02}_sampling.json') != read_json(run/'predicted'/f'epoch_{epoch:02}_sampling.json'):
            raise ValueError('Sample order not paired')
        benefit=paired_benefit(baseline,results[('uniform',epoch)]['validation'],results[('predicted',epoch)]['validation'],config['guards'])
        saved=next(x for x in outcome['paired_epochs'] if x['epoch']==epoch)
        for key,value in benefit.items():
            if saved[key] != value:
                raise ValueError('Benefit results inconsistent')
    return config,baseline,results,outcome,protected


def create_report(run, output):
    config,baseline,results,outcome,protected=verify_run(run)
    old=resolve(config['previous_outcome']).parent
    old_rows={}
    for arm in ('uniform','predicted'):
        old_rows[arm]=read_json(old/arm/'epoch_02_evaluation.json')['validation']
    lines=['# Low-surface protection experiment: completed development results','',
           f'Run: `{run}`. No automatic app promotion. All metrics below use the same',
           '160 HighBuild and 120 OpenCanopy development images and supported reference pixels.','',
           '| Model | Building RMSE | Canopy RMSE | Ground RMSE | Short-vegetation RMSE | Safety |',
           '|---|---:|---:|---:|---:|---|']
    table=[('Protected app',baseline,'Baseline')]
    table += [(f'Old {arm}, epoch 2',value,'Fail') for arm,value in old_rows.items()]
    table += [(f'New {arm}, epoch {epoch}',r['validation'],'Pass' if r['guard']['legacy_safety_pass'] else 'Fail')
              for (arm,epoch),r in results.items()]
    for label,validation,safety in table:
        values=[validation['groups'][key]['rmse_m'] for key in list(GROUPS.values())[:4]]
        lines.append('| '+label+' | '+' | '.join(f'{value:.3f} m' for value in values)+' | '+safety+' |')
    unrestricted=min(results,key=lambda pair:sum(results[pair]['validation']['groups'][GROUPS[g]]['rmse_m'] for g in ('Buildings','Canopy')))
    eligible=[(a,e) for (a,e),r in results.items() if a=='predicted' and r['guard']['legacy_safety_pass']
              and next(x for x in outcome['paired_epochs'] if x['epoch']==e)['benefit_pass']]
    lines += ['',f'Best unrestricted mean building/canopy: **{unrestricted[0]} epoch {unrestricted[1]}**.',
              f'Best eligible class-assisted epoch: **{outcome["best_eligible_epoch"] if eligible else "none"}**.',
              'Safety and benefit are independent. A better aggregate number is not release approval.','',
              '## Every category and metric','',
              '| Model / category | RMSE m | MAE m | Bias m | Correlation | R2 |','|---|---:|---:|---:|---:|---:|']
    for label,validation,_ in table:
        for name,key in GROUPS.items():
            values=validation['groups'][key]
            lines.append('| '+label+' / '+name+' | '+' | '.join('N/A' if values.get(k) is None else f'{values[k]:.4f}' for k in ('rmse_m','mae_m','bias_m','correlation','r2'))+' |')
    lines += ['','## Safety details','']
    for (arm,epoch),result in results.items():
        failures=result['guard']['failed_checks']
        global_failures=[x for x in failures if '/region/' not in x]
        lines += [f'- {arm} epoch {epoch}: {len(failures)} failed overlapping checks (not images); '+(', '.join(global_failures) or 'no global failures')+'.']
    b=results[('predicted',2)]['validation']['groups']
    a=results[('uniform',2)]['validation']['groups']
    lines += ['','## Interpretation limits','',
        f'At epoch 2, adding class probabilities changes canopy RMSE by {b[GROUPS["Canopy"]]["rmse_m"]-a[GROUPS["Canopy"]]["rmse_m"]:+.4f} m and building RMSE by {b[GROUPS["Buildings"]]["rmse_m"]-a[GROUPS["Buildings"]]["rmse_m"]:+.4f} m versus matched control (negative is better).',
        'One seed and two epochs do not establish a reliable class-conditioning benefit. New-vs-old runs differ in the predeclared loss; do not attribute all changes to classification.',
        'This was not an untouched final test. GAMUS heights, water/road heights, broad hill/plain generalisation and unseen sensors were not validated.',
        'No gates were relaxed. Frozen semantics and fixed support are checked by the original evaluator. Code snapshots, final commits, sampling/initialization and all four protected artifact hashes verified.',
        'The live app still uses the protected checkpoint. All experimental checkpoints remain on D:.']
    output.mkdir(parents=True,exist_ok=True)
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'verification.json').write_text(json.dumps({'run':str(run),'protected_hashes_match':protected,
        'best_unrestricted':list(unrestricted),'eligible_candidates':eligible,'all_checks_passed':True,
        'app_promoted':False},indent=2)+'\n',encoding='utf-8')
    return {'report':str(output/'REPORT.md'),'best_unrestricted':list(unrestricted),'eligible_candidates':eligible}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(create_report(args.run,args.output),indent=2))
