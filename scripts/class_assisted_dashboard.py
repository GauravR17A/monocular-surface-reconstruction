"""Read-only native progress window for the paired height experiment.

Closing this window never stops training. Reports are written outside the run.
No PyTorch import, GPU inference, training configuration edits, or model promotion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / 'outputs/orchestration/class_assisted_height_active.json'
GROUPS = {
    'Buildings': 'highbuild/domain/building',
    'Canopy': 'open_canopy/domain/vegetation',
    'Ground': 'open_canopy/domain/ground',
    'Tall buildings (20m+)': 'highbuild/tall/building',
    'Tall canopy (15m+)': 'open_canopy/tall/vegetation',
    'Short vegetation (0-2m)': 'open_canopy/short_vegetation',
    'Short buildings (<20m)': 'highbuild/height_band/short_building',
    'Medium vegetation (2-15m)': 'open_canopy/height_band/medium_vegetation',
}
PHASES = {
    'checking_reused_inputs': 'Checking saved inputs for the faster run (no repeat download)',
    'preparing_verified_inputs': 'Checking images + preparing depth / classification inputs',
    'baseline_validation': 'Measuring the protected model on the fixed validation set',
    'training': 'Training the experimental height correction',
    'development_validation': 'Checking completed epoch against reference heights',
    'complete': 'Comparison finished - review the evidence below',
    'failed': 'Run needs attention - original app model is still protected',
}


def shared_bytes(path):
    """Allow the Windows writer to replace an active file while it is read."""
    path = Path(path)
    if os.name != 'nt':
        return path.read_bytes()
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(path), 0x80000000, 1 | 2 | 4, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(fd, 'rb') as stream:
        return stream.read()


def read_json(path):
    return json.loads(shared_bytes(path))


def number(value, digits=2):
    return f'{value:.{digits}f}' if isinstance(value, (int, float)) and math.isfinite(value) else 'N/A'


def metric(validation, group, key='rmse_m'):
    return validation.get('groups', {}).get(group, {}).get(key)


def paired_review(baseline, control, candidate, guards):
    """Display the locked benefit rule, separate from the trainer's safety gate."""
    if not baseline or not control:
        return 'PENDING'
    for group in (GROUPS['Buildings'], GROUPS['Canopy']):
        old, a, b = (metric(x, group) for x in (baseline, control, candidate))
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (old, a, b)):
            continue
        gain = old - b
        if old > 0 and gain >= guards['improvement_m'] and gain / old >= guards['improvement_fraction'] and b < a:
            return 'PASS'
    return 'FAIL'


def snapshot(registration=REGISTRATION):
    active = read_json(registration)
    run = Path(active['run_dir'])
    state, config = read_json(run / 'status.json'), read_json(run / 'config.json')
    arms = config.get('arms', ['uniform', 'predicted'])
    labels = config.get('arm_labels', {'uniform': 'A: control', 'predicted': 'B: class-assisted'})
    epochs = config.get('epochs', 2)
    batches_per_epoch = config.get('pairs_per_epoch', 600)
    baseline = None
    if (run / 'baseline_binding.json').exists():
        raw = shared_bytes(run / 'baseline.json')
        if hashlib.sha256(raw).hexdigest() != read_json(run / 'baseline_binding.json')['sha256']:
            raise ValueError('Baseline report hash does not match its saved binding')
        baseline = json.loads(raw)
    rows, commits = [], {}
    if baseline:
        rows.append({'id': 'protected', 'label': 'Protected app model', 'validation': baseline,
                     'safety': 'BASELINE', 'benefit': '-', 'failures': []})
    evaluations = {}
    for arm in arms:
        commit_path = run / arm / 'commit.json'
        if not commit_path.exists():
            continue
        commit = read_json(commit_path)
        commits[arm] = commit
        last_evaluated = commit['epoch'] if commit['stage'] == 'evaluated' else commit['epoch'] - 1
        for epoch in range(1, last_evaluated + 1):
            path = run / arm / f'epoch_{epoch:02}_evaluation.json'
            raw = shared_bytes(path)
            if epoch == commit['epoch'] and hashlib.sha256(raw).hexdigest() != commit['evaluation_sha256']:
                raise ValueError('Latest epoch report does not match its committed hash')
            report = json.loads(raw)
            if report['arm'] != arm or report['epoch'] != epoch:
                raise ValueError('Epoch report identity mismatch')
            evaluations[(arm, epoch)] = report
            benefit = '-' if arm == arms[0] else paired_review(
                baseline, evaluations.get((arms[0], epoch), {}).get('validation'), report['validation'], config['guards'])
            rows.append({'id': f'{arm}_{epoch}', 'label': f'{labels[arm]} / epoch {epoch}',
                         'validation': report['validation'], 'safety': 'PASS' if report['guard']['legacy_safety_pass'] else 'FAIL',
                         'benefit': benefit, 'failures': report['guard']['failed_checks']})
    progress = {}
    for arm in arms:
        commit = commits.get(arm, {})
        batches = max(0, (int(commit.get('epoch', 1)) - 1) * batches_per_epoch + int(commit.get('batch_done', 0)))
        if state.get('arm') == arm:
            if state['stage'] == 'training':
                batches = max(batches, (state['epoch'] - 1) * batches_per_epoch + state['completed'])
            elif state['stage'] == 'development_validation':
                batches = max(batches, state['epoch'] * batches_per_epoch)
        progress[arm] = min(epochs * batches_per_epoch, batches)
    outcome = read_json(run / 'outcome.json') if (run / 'outcome.json').exists() else None
    return {'run_dir': str(run), 'active': active, 'state': state, 'config': config,
            'rows': rows, 'commits': commits, 'progress': progress, 'outcome': outcome,
            'completed_epochs': len(evaluations)}


def report_text(data):
    config = data['config']
    total_epochs = config.get('epochs', 2) * len(config.get('arms', ['uniform', 'predicted']))
    comparison = config.get('comparison_summary', 'A = uniform class inputs. B = predicted six-class probabilities. Same training data, initial weights and sample order.')
    lines = ['MSR | CONTROLLED HEIGHT EXPERIMENT',
             f"Run: {data['run_dir']}", f"Stage: {data['state']['stage']}",
             f"Completed evaluated epochs: {data['completed_epochs']}/{total_epochs}", '', comparison, '',
             'VALIDATION RESULTS (metres; lower RMSE / MAE is better)',
             'Safety applies to all reported supported groups, not only the three displayed below.']
    for row in data['rows']:
        lines += ['', row['label'], f"Legacy safety: {row['safety']} | Matched benefit: {row['benefit']}",
                  'Category                         RMSE      MAE     Bias     Corr       R2']
        for name, group in GROUPS.items():
            values = row['validation'].get('groups', {}).get(group, {})
            lines.append(f"{name:30} " + ' '.join(f'{number(values.get(key), 3 if key in ("correlation", "r2") else 2):>8}'
                         for key in ('rmse_m', 'mae_m', 'bias_m', 'correlation', 'r2')))
        if row['failures']:
            lines += [f"Failed safety checks ({len(row['failures'])}):"] + ['  ' + x for x in row['failures']]
    if not data['rows']:
        lines += ['', 'No completed validation results yet. Values will appear only after evaluation finishes.']
    lines += ['', 'HOW TO READ THIS',
              'RMSE: height-error score that penalizes large mistakes. MAE: average absolute height error.',
              'Bias: predicted minus reference height; negative means underestimated.',
              'Correlation: whether the taller/lower pattern agrees. R2: improvement over a constant-mean predictor.',
              'Benefit: B must beat A at the same epoch and improve building OR canopy RMSE',
              'by both >=0.5 m and >=5% versus the protected baseline.',
              'Safety PASS alone does not prove improvement. A failed safety check blocks candidate eligibility.', '',
              'COVERAGE AND LIMITS',
              'Training: 450 corrected HighBuild urban + 600 OpenCanopy forest images.',
              'Development: 160 HighBuild + 120 OpenCanopy images, evaluated at native resolution.',
              'Building, canopy, forest ground, tall objects and short vegetation: reported where labels exist.',
              'Road/water heights, broad plains/hills, other sensors/resolutions and genuinely new geography:',
              'NOT certified by this experiment. GAMUS height units remain unresolved.',
              'Class-conditioning policy: ' + config.get('conditioning', 'predicted probabilities only in candidate B') + '.',
              'This is development validation, not an untouched final-test claim.',
              'Production model unchanged. No automatic app promotion. No universal accuracy guarantee.']
    if data['outcome']:
        best = data['outcome'].get('best_eligible_epoch')
        lines += ['', f'FINAL: best safety-and-benefit eligible epoch = {best if best is not None else "NONE"}.',
                  'Release remains ineligible until the remaining checks are completed.']
    return '\n'.join(lines) + '\n'


def export_reports(data):
    destination = ROOT / 'outputs/reports/class_assisted_height' / Path(data['run_dir']).name
    destination.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer)
    writer.writerow(['model_epoch', 'group', 'pixel_count', 'rmse_m', 'mae_m', 'bias_m', 'correlation', 'r2', 'legacy_safety', 'matched_benefit'])
    for row in data['rows']:
        for group, values in row['validation'].get('groups', {}).items():
            writer.writerow([row['label'], group] + [values.get(k) for k in ('pixel_count', 'rmse_m', 'mae_m', 'bias_m', 'correlation', 'r2')] + [row['safety'], row['benefit']])
    for name, text in [('report.txt', report_text(data)), ('metrics.csv', buffer.getvalue()),
                       ('dashboard_snapshot.json', json.dumps(data, indent=2))]:
        target = destination / name
        temporary = target.with_suffix(target.suffix + f'.{os.getpid()}.tmp')
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, target)
    return destination


def gpu_status():
    executable = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32/nvidia-smi.exe'
    try:
        result = subprocess.run([str(executable), '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=True)
        load, used, total, temp, power = result.stdout.strip().splitlines()[0].split(', ')
        return f'RTX 4070  |  {load}% load  |  {float(used)/1024:.1f}/{float(total)/1024:.1f} GB  |  {temp} C  |  {power} W'
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 'GPU telemetry temporarily unavailable (training is independent of this window)'


def launch_dashboard(registration):
    import tkinter as tk
    from tkinter import ttk, messagebox
    window = tk.Tk()
    window.title('Monocular Surface Reconstruction | Live training & validation')
    window.configure(bg='#0b1220')
    window.geometry('1120x780')
    window.minsize(940, 650)
    initial = snapshot(registration)
    config = initial['config']
    arms = config.get('arms', ['uniform', 'predicted'])
    labels = config.get('arm_labels', {'uniform': 'A: control', 'predicted': 'B: class-assisted'})
    epochs = config.get('epochs', 2)
    total_epochs = epochs * len(arms)
    max_batches = epochs * config.get('pairs_per_epoch', 600)
    style = ttk.Style(window)
    style.theme_use('clam')
    style.configure('.', font=('Segoe UI', 10), background='#0b1220', foreground='#e5eef9')
    style.configure('Card.TFrame', background='#142035')
    style.configure('Card.TLabel', background='#142035', foreground='#e5eef9')
    style.configure('Accent.TLabel', background='#142035', foreground='#4ee6c0', font=('Segoe UI', 12, 'bold'))
    style.configure('Live.Horizontal.TProgressbar', background='#4ee6c0', troughcolor='#26344a', borderwidth=0, thickness=12)
    style.configure('Control.Horizontal.TProgressbar', background='#67aafa', troughcolor='#26344a', borderwidth=0, thickness=9)
    style.configure('TNotebook', background='#0b1220', borderwidth=0)
    style.configure('TNotebook.Tab', background='#20314c', padding=(20, 10))
    style.map('TNotebook.Tab', background=[('selected', '#315175')])
    style.configure('Treeview', background='#142035', fieldbackground='#142035', foreground='#e5eef9', rowheight=38, borderwidth=0)
    style.configure('Treeview.Heading', background='#21334e', foreground='#b7cce7', relief='flat', font=('Segoe UI', 10, 'bold'))
    style.map('Treeview', background=[('selected', '#2b4668')], foreground=[('selected', '#ffffff')])
    style.configure('TButton', background='#21334e', foreground='#e5eef9', padding=(12, 7))
    outer = ttk.Frame(window, padding=24)
    outer.pack(fill='both', expand=True)
    ttk.Label(outer, text='MSR  /  LIVE EXPERIMENT', foreground='#4ee6c0', font=('Segoe UI', 11, 'bold')).pack(anchor='w')
    ttk.Label(outer, text=config.get('dashboard_title', 'Do better classes mean better heights?'), font=('Segoe UI', 24, 'bold')).pack(anchor='w', pady=(5, 3))
    ttk.Label(outer, text='A fair A/B comparison. Real saved results. Your app model stays protected.', foreground='#9bb2cd').pack(anchor='w', pady=(0, 17))
    card = ttk.Frame(outer, style='Card.TFrame', padding=18)
    card.pack(fill='x')
    phase = ttk.Label(card, text='Connecting to saved progress...', style='Accent.TLabel')
    phase.pack(anchor='w')
    detail = ttk.Label(card, style='Card.TLabel')
    detail.pack(anchor='w', pady=(7, 6))
    main_bar = ttk.Progressbar(card, style='Live.Horizontal.TProgressbar', maximum=100)
    main_bar.pack(fill='x', pady=(0, 12))
    arm_bars, arm_labels = {}, {}
    for arm in arms:
        title = labels[arm]
        line = ttk.Frame(card, style='Card.TFrame')
        line.pack(fill='x', pady=3)
        ttk.Label(line, text=title, width=20, style='Card.TLabel').pack(side='left')
        arm_bars[arm] = ttk.Progressbar(line, maximum=max_batches, style='Control.Horizontal.TProgressbar')
        arm_bars[arm].pack(side='left', fill='x', expand=True, padx=12)
        arm_labels[arm] = ttk.Label(line, text=f'0 / {max_batches:,} batches', width=22, style='Card.TLabel')
        arm_labels[arm].pack(side='right')
    gpu_label = ttk.Label(card, text='Reading GPU...', foreground='#9bb2cd', style='Card.TLabel')
    gpu_label.pack(anchor='w', pady=(10, 0))
    notebook = ttk.Notebook(outer)
    results_page, report_page = ttk.Frame(notebook), ttk.Frame(notebook)
    notebook.add(results_page, text='Epoch results')
    notebook.add(report_page, text='Full report + tested coverage')
    columns = ('model', 'building', 'canopy', 'ground', 'safety', 'benefit')
    table = ttk.Treeview(results_page, columns=columns, show='headings', height=5, selectmode='browse')
    for key, label, width in zip(columns, ['Model / epoch', 'Building RMSE', 'Canopy RMSE', 'Ground RMSE', 'Legacy safety', 'Matched benefit'], [245, 140, 140, 140, 130, 135]):
        table.heading(key, text=label)
        table.column(key, width=width, minwidth=100, stretch=True)
    table.pack(fill='x', pady=(12, 5))
    table.tag_configure('failed', foreground='#ffb99c')
    table.tag_configure('qualified', foreground='#4ee6c0')
    explanation = ttk.Label(results_page, text='Waiting for completed validation. No placeholder scores are used.', foreground='#9bb2cd', wraplength=1030)
    explanation.pack(anchor='w', pady=(5, 8))
    selected = tk.Text(results_page, height=4, bg='#101a2c', fg='#cbdbee', relief='flat', font=('Consolas', 10), padx=12, pady=8, wrap='word')
    selected.pack(fill='both', expand=True)
    report_scroll = ttk.Scrollbar(report_page)
    report_scroll.pack(side='right', fill='y')
    report = tk.Text(report_page, height=8, bg='#101a2c', fg='#d7e4f3', relief='flat', font=('Consolas', 10), padx=16, pady=14, wrap='word', yscrollcommand=report_scroll.set)
    report.pack(fill='both', expand=True, pady=(10, 0))
    report_scroll.configure(command=report.yview)
    footer = ttk.Frame(outer)
    footer.pack(side='bottom', fill='x')
    saved = ttk.Label(footer, text='Refresh: 2 seconds | Closing this window will NOT stop training.', foreground='#9bb2cd')
    saved.pack(side='left', fill='x', expand=True)
    state = {'data': None, 'signature': None, 'report_dir': None, 'gpu_future': None, 'gpu_time': 0, 'ticks': 0, 'export_time': 0}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dashboard-gpu')

    def replace_text(widget, text):
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', text)
        widget.configure(state='disabled')

    def show_selection(_event=None):
        data = state['data']
        if not data:
            return
        chosen = table.selection()
        row = next((r for r in data['rows'] if chosen and r['id'] == chosen[0]), None)
        if row is None:
            replace_text(selected, 'Select a completed row to see MAE, bias, correlation and R2.\nLower RMSE / MAE is better. Development results do not guarantee every uploaded image.')
            return
        lines = [f"{row['label']}  |  {len(row['failures'])} failed safety checks", '                     RMSE     MAE    Bias    Corr      R2']
        for name in ('Buildings', 'Canopy', 'Ground'):
            values = row['validation'].get('groups', {}).get(GROUPS[name], {})
            lines.append(f'{name:18}' + ''.join(f'{number(values.get(key), 3 if key in ("correlation", "r2") else 2):>8}' for key in ('rmse_m', 'mae_m', 'bias_m', 'correlation', 'r2')))
        replace_text(selected, '\n'.join(lines))

    def open_reports():
        if not state['data']:
            return
        try:
            directory = export_reports(state['data'])
            os.startfile(str(directory))
        except OSError as error:
            messagebox.showerror('Report export', str(error), parent=window)

    ttk.Button(footer, text='Open saved reports', command=open_reports).pack(side='right')
    table.bind('<<TreeviewSelect>>', show_selection)
    notebook.pack(fill='both', expand=True, pady=(18, 10))

    def refresh():
        try:
            data = snapshot(registration)
            state['data'] = data
            current = data['state']
            phase.configure(text=PHASES.get(current['stage'], current['stage']))
            done, total = int(current.get('completed', 0)), int(current.get('total', 0))
            percent = 100 * done / total if total else (100 if current['stage'] == 'complete' else 0)
            main_bar['value'] = percent
            if current['stage'] == 'failed':
                detail.configure(text=str(current.get('error', 'See saved failure report'))[:160], foreground='#ffb99c')
            else:
                epoch = f" | {current.get('arm', '')} epoch {current['epoch']}/{epochs}" if current.get('epoch', 0) else ''
                loss = f" | training loss {current['loss']:.3f} (not accuracy)" if 'loss' in current else ''
                stamp = datetime.fromisoformat(current['updated_utc']).astimezone()
                age = max(0, int((datetime.now(timezone.utc) - stamp).total_seconds()))
                detail_text = f'{done:,} / {total:,}  ({percent:.1f}%)' + epoch + loss
                if current['stage'] == 'complete':
                    detail_text = f"All {data['completed_epochs']}/{total_epochs} epochs evaluated | Best eligible epoch: {current.get('best_eligible_epoch') or 'NONE'}"
                detail.configure(text=detail_text + f' | saved {age}s ago', foreground='#e5eef9')
            for arm in arm_bars:
                batches = data['progress'][arm]
                arm_bars[arm]['value'] = batches
                arm_labels[arm].configure(text=f'{batches:,} / {max_batches:,} batches')
            signature = json.dumps({'rows': data['rows'], 'outcome': data['outcome']}, sort_keys=True)
            if signature != state['signature']:
                old_selection = table.selection()
                table.delete(*table.get_children())
                for row in data['rows']:
                    values = [row['label']] + [number(metric(row['validation'], GROUPS[n])) + ' m' for n in ('Buildings', 'Canopy', 'Ground')] + [row['safety'], row['benefit']]
                    tag = 'failed' if row['safety'] == 'FAIL' else ('qualified' if row['safety'] == row['benefit'] == 'PASS' else '')
                    table.insert('', 'end', iid=row['id'], values=values, tags=(tag,))
                if old_selection and table.exists(old_selection[0]):
                    table.selection_set(old_selection[0])
                replace_text(report, report_text(data))
                show_selection()
                state['signature'] = signature
            explanation.configure(text=f"{data['completed_epochs']}/{total_epochs} evaluated epochs. RMSE is in metres; lower is better. Both safety and benefit are required; no automatic release.")
            if time.monotonic() - state['export_time'] >= 30:
                state['report_dir'] = export_reports(data)
                replace_text(report, report_text(data))
                state['export_time'] = time.monotonic()
            future = state['gpu_future']
            if future is not None and future.done():
                gpu_label.configure(text=future.result())
                state['gpu_future'] = None
            if state['gpu_future'] is None and time.monotonic() - state['gpu_time'] >= 10:
                state['gpu_future'] = executor.submit(gpu_status)
                state['gpu_time'] = time.monotonic()
        except Exception as error:
            phase.configure(text='Waiting for a consistent saved update')
            detail.configure(text=str(error)[:160], foreground='#ffb99c')
        window.after(2000, refresh)

    def close():
        executor.shutdown(wait=False, cancel_futures=True)
        window.destroy()

    window.protocol('WM_DELETE_WINDOW', close)
    refresh()
    window.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, default=REGISTRATION)
    parser.add_argument('--once', action='store_true', help='Print and export the report without opening a window')
    args = parser.parse_args()
    if args.once:
        data = snapshot(args.registration)
        print(report_text(data))
        print('Reports:', export_reports(data))
    else:
        launch_dashboard(args.registration)


if __name__ == '__main__':
    main()
