#!/usr/bin/env python3
"""Read-only seven-joint torque recording while an existing controller holds an object.

This program never enables/disables motors, changes control mode, moves joints,
initializes a gripper, or sends a gripper target. Delta torque is a fixed-pose
baseline difference, not a calibrated external torque/force estimate.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import signal
import time

import numpy as np


def read_feedback(robot, wall_time=None):
    joints = robot.get_joint_angles()
    motors = [robot.get_motor_states(i) for i in range(1, 8)]
    if joints is None or any(m is None for m in motors):
        return None
    stamps = np.array([joints.timestamp, *(m.timestamp for m in motors)], dtype=float)
    torque = np.array([m.msg.torque for m in motors], dtype=float)
    q = np.array(joints.msg, dtype=float)
    if wall_time is None:
        wall_time = time.time()
    age = wall_time - stamps
    if (q.shape != (7,) or not np.isfinite(np.r_[q, torque, stamps]).all()
            or np.any(age < 0.) or np.any(age > .5)):
        return None
    return dict(q_deg=np.rad2deg(q), torque=torque, ages_ms=age[1:] * 1000.,
                stamps=tuple(stamps[1:]), skew_ms=float(np.ptp(stamps[1:]) * 1000.))


def summarize_samples(samples, baseline_seconds):
    t = np.array([s['t_s'] for s in samples])
    torque = np.array([s['torque'] for s in samples])
    q = np.array([s['q_deg'] for s in samples])
    rest = t < baseline_seconds
    if rest.sum() < 5 or not np.any(~rest):
        raise ValueError('Need at least five baseline samples and post-baseline observations')
    baseline = np.median(torque[rest], axis=0)
    delta = torque - baseline
    report = dict(
        semantics='Motor feedback Nm minus initial held-object baseline; not external force or an automatic release decision',
        samples=len(samples), baseline_samples=int(rest.sum()), baseline_seconds=baseline_seconds,
        baseline_torque_nm=baseline.tolist(), baseline_std_nm=np.std(torque[rest], axis=0).tolist(),
        baseline_mad_nm=np.median(np.abs(torque[rest] - baseline), axis=0).tolist(),
        peak_abs_delta_nm=np.max(np.abs(delta[~rest]), axis=0).tolist(),
        maximum_joint_drift_deg=np.max(np.abs(q - np.median(q[rest], axis=0)), axis=0).tolist(),
        maximum_feedback_age_ms=max(float(np.max(s['ages_ms'])) for s in samples),
        maximum_interjoint_feedback_skew_ms=max(s['skew_ms'] for s in samples),
        labels={},
    )
    for label in ('REST', 'PULL'):
        mask = (~rest) & np.array([s['label'] == label for s in samples])
        if np.any(mask):
            report['labels'][label] = dict(samples=int(mask.sum()),
                delta_rms_nm=np.sqrt(np.mean(delta[mask] ** 2, axis=0)).tolist(),
                peak_abs_delta_nm=np.max(np.abs(delta[mask]), axis=0).tolist())
    return report, delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--firmware', choices=('V111', 'V112', 'V120'), default='V111')
    parser.add_argument('--duration', type=float, default=60.)
    parser.add_argument('--baseline-seconds', type=float, default=5.)
    parser.add_argument('--hz', type=float, default=20.)
    parser.add_argument('--plot', action='store_true', help='Live raw/delta plot; P labels pull, R labels rest, Q exits')
    parser.add_argument('--save-dir', type=Path, default=Path('logs/pull_torque'))
    args = parser.parse_args()
    if (not np.isfinite([args.duration, args.baseline_seconds, args.hz]).all()
            or not 0. < args.baseline_seconds < args.duration or not 0. < args.hz <= 100.):
        parser.error('Require 0 < baseline-seconds < duration and 0 < hz <= 100')

    import matplotlib
    matplotlib.use('TkAgg' if args.plot else 'Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['pdf.fonttype'] = 42
    # P/R label data; Matplotlib's default pan/home shortcuts must not also run.
    for name in ('keymap.pan', 'keymap.home'):
        plt.rcParams[name] = [key for key in plt.rcParams[name] if key.lower() not in ('p', 'r')]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    lines = [[ax.plot([], [], label=f'J{i + 1}')[0] for i in range(7)] for ax in axes]
    for ax, title in zip(axes, ('Motor feedback torque', 'Change from initial held-object baseline')):
        ax.set(title=title, ylabel='N m'); ax.grid(alpha=.25); ax.legend(ncol=7, loc='upper left')
        ax.axvspan(0., args.baseline_seconds, alpha=.1, color='gray')
    axes[1].set_xlabel('Seconds'); fig.tight_layout()
    state = dict(stop=False, label='REST')
    if args.plot:
        def on_key(event):
            key = (event.key or '').lower()
            if key in ('p', 'r'):
                state['label'] = 'PULL' if key == 'p' else 'REST'
            elif key in ('q', 'escape'):
                state['stop'] = True
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('close_event', lambda event: state.update(stop=True))
        plt.ion(); plt.show(block=False)

    # Direct SDK construction, not the grasp adapter whose context enables motors.
    from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
    config = create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=getattr(NeroFW, args.firmware),
                                  interface='socketcan', channel=args.channel, bitrate=1000000)
    robot = AgxArmFactory.create_arm(config)
    output = args.save_dir / datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    fields = ['t_s', 'label'] + [f'{kind}{i}{unit}' for kind, unit in
        (('q', '_deg'), ('tau', '_nm'), ('delta', '_nm'), ('age', '_ms')) for i in range(1, 8)] + ['skew_ms']
    samples = []; baseline = None; previous_stamps = None
    start = None; connected_at = time.monotonic(); last_fresh = connected_at
    last_print = last_plot = 0.; skipped = 0
    print('READ ONLY: existing arm/gripper hold stays in control; no motor or gripper commands.')
    print(f'Keep the held object untouched for the first {args.baseline_seconds:g}s. After BASELINE READY, gently pull/release repeatedly.')
    print('Plot: P marks PULL, R marks REST, Q/Esc exits. Labels are human annotations, not a force detector.')
    print('Log:', output, flush=True)
    robot.connect()
    # Tk catches KeyboardInterrupt raised inside its drawing callback. Request
    # loop exit instead so a single Ctrl+C still closes CAN and saves the data.
    previous_sigint = signal.signal(signal.SIGINT, lambda *_: state.update(stop=True))
    try:
        with (output / 'samples.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(fields)
            while not state['stop']:
                now = time.monotonic()
                if start is not None and now - start >= args.duration:
                    break
                sample = read_feedback(robot)
                if sample is None or sample['stamps'] == previous_stamps:
                    skipped += 1
                    if start is None and now - connected_at > 5.:
                        raise TimeoutError('No fresh seven-joint feedback; check CAN. No enable/motion command was sent.')
                    if start is not None and now - last_fresh > 1.:
                        raise TimeoutError('Seven-joint feedback stopped updating; raw CSV retained. Recording stopped without changing control.')
                else:
                    last_fresh = now
                    previous_stamps = sample['stamps']
                    if start is None:
                        start = now
                    sample['t_s'] = elapsed = now - start
                    sample['label'] = 'BASELINE' if elapsed < args.baseline_seconds else state['label']
                    if baseline is None and elapsed >= args.baseline_seconds:
                        baseline = np.median([s['torque'] for s in samples], axis=0)
                        print('\nBASELINE READY: tau0 Nm =', np.round(baseline, 3).tolist(), flush=True)
                    delta = np.full(7, np.nan) if baseline is None else sample['torque'] - baseline
                    writer.writerow([round(elapsed, 4), sample['label'], *sample['q_deg'],
                                     *sample['torque'], *delta, *sample['ages_ms'], sample['skew_ms']])
                    samples.append(sample)
                    if now - last_print >= 1.:
                        last_print = now
                        readiness = 'BASELINE COLLECTING' if baseline is None else 'BASELINE READY'
                        print(f"t={elapsed:5.1f}s {readiness} | label={sample['label']} | J1..J7 delta Nm={np.round(delta, 3).tolist()}", flush=True)
                    if args.plot and now - last_plot >= .1:
                        last_plot = now
                        recent = samples[-int(args.hz * 30):]
                        ts = [s['t_s'] for s in recent]; raw = np.array([s['torque'] for s in recent])
                        change = raw * np.nan if baseline is None else raw - baseline
                        for panel, values in zip(lines, (raw, change)):
                            for i, line in enumerate(panel): line.set_data(ts, values[:, i])
                        for ax in axes: ax.relim(); ax.autoscale_view()
                        readiness = 'BASELINE COLLECTING' if baseline is None else 'BASELINE READY'
                        fig.suptitle(f"{readiness} | label={sample['label']} | P: pull / R: rest / Q: quit")
                        fig.canvas.draw_idle()
                if args.plot:
                    fig.canvas.flush_events()
                time.sleep(1. / args.hz)
    except KeyboardInterrupt:
        print('\nStopped recording; arm and gripper commands were not changed.')
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        robot.disconnect()
    print()
    if not samples or baseline is None:
        print('No complete baseline; raw CSV retained:', output); plt.close(fig); return 1
    report, delta = summarize_samples(samples, args.baseline_seconds)
    report.update(skipped_missing_stale_or_duplicate=skipped, channel=args.channel, firmware=args.firmware)
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    t = [s['t_s'] for s in samples]; raw = np.array([s['torque'] for s in samples])
    for panel, values in zip(lines, (raw, delta)):
        for i, line in enumerate(panel): line.set_data(t, values[:, i])
    for ax in axes: ax.relim(); ax.autoscale_view()
    fig.suptitle('Read-only held-object pull test; feedback delta is not an external torque estimate')
    fig.tight_layout(); fig.savefig(output / 'torque.png', dpi=180); fig.savefig(output / 'torque.pdf')
    plt.close(fig)
    print('Peak |delta| J1..J7 Nm:', np.round(report['peak_abs_delta_nm'], 3).tolist())
    print('Baseline noise std Nm:', np.round(report['baseline_std_nm'], 4).tolist())
    print('Max joint drift deg:', np.round(report['maximum_joint_drift_deg'], 3).tolist())
    print('Saved samples.csv, summary.json, torque.png and torque.pdf to', output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
