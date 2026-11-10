#!/usr/bin/env python3
"""Fit a small action classifier from separately labeled fusion shadow sessions.

Requires disjoint whole train/test sessions. Report is sample-level action
classification, NOT validation of support or closed-loop release reliability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handoff.fusion_handover import FEATURE_VERSION, LinearPullModel

LABELS = {'TAKE_PULL': 1, 'NO_CONTACT': 0, 'BUMP_PUSH': 0, 'POST_RELEASE': 0}


def load_sessions(paths):
    features, labels, manifests = [], [], []
    window = None
    for name in paths:
        directory = Path(name).expanduser().resolve()
        meta = json.loads((directory/'metadata.json').read_text())
        if meta.get('dry_run') is not False or meta.get('control_mode') != 'mit_cartesian_hold':
            raise ValueError(f'{directory}: require real, same-mode shadow recordings')
        if meta.get('mode') != 'shadow':
            raise ValueError('train initial action classifier on shadow-only data')
        duration = float(meta['config']['window_seconds'])
        if window is not None and duration != window:
            raise ValueError('all training/test feature windows must match')
        window = duration
        source = directory/'samples.jsonl'
        previous = None
        count = 0
        with source.open() as stream:
            for line in stream:
                row = json.loads(line)
                policy, visual, feedback = row.get('policy') or {}, row.get('visual') or {}, row.get('feedback') or {}
                x, label = policy.get('features'), visual.get('label')
                # Terminal snapshots can retain their last features; do not
                # turn those repeated values into new training observations.
                if policy.get('phase') not in ('FUSION_WAIT_TAKE', 'FUSION_CONFIRM') or x is None or label not in LABELS:
                    continue
                stamps = np.asarray(feedback.get('motor_timestamps', []), float)
                if stamps.shape != (7,) or not np.isfinite(stamps).all():
                    continue
                if previous is not None and not np.all(stamps > previous):
                    continue
                previous = stamps
                now = float(row['monotonic'])
                if not 0 <= now-float(visual.get('received_monotonic', -1e20)) <= .25:
                    continue
                value = np.asarray(x, float)
                if value.shape != (28,) or not np.isfinite(value).all():
                    raise ValueError('invalid recorded feature vector')
                features.append(value)
                labels.append(LABELS[label])
                count += 1
        manifests.append(dict(path=str(directory), sha256=hashlib.sha256(source.read_bytes()).hexdigest(), samples=count))
    if not features or set(labels) != {0, 1}:
        raise ValueError('each split needs labeled TAKE_PULL and confirmed negative samples')
    return np.array(features), np.array(labels), manifests, window


def fit_binary(x, y, *, joints=tuple(range(1, 8)), regularization=.01):
    from scipy.optimize import minimize
    if not joints or len(set(joints)) != len(joints) or any(j not in range(1, 8) for j in joints):
        raise ValueError('joints must be distinct indices in 1..7')
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError('regularization must be finite and positive')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-6] = 1.
    active = np.zeros(28, bool)
    for group in range(4):
        active[group*7+np.array(joints)-1] = True
    a = ((x-mean)/scale)[:, active]
    sample_weight = np.where(y == 1, .5/np.count_nonzero(y), .5/np.count_nonzero(y == 0))
    def objective(beta):
        z = a @ beta[:-1]+beta[-1]
        p = 1/(1+np.exp(-np.clip(z, -60, 60)))
        cost = sample_weight @ (np.logaddexp(0, z)-y*z) + regularization*(beta[:-1] @ beta[:-1])/2
        residual = sample_weight*(p-y)
        gradient = np.r_[a.T @ residual+regularization*beta[:-1], residual.sum()]
        return cost, gradient
    result = minimize(objective, np.zeros(a.shape[1]+1), jac=True, method='L-BFGS-B', options={'maxiter': 1000})
    if not result.success:
        raise ValueError(f'action fit failed: {result.message}')
    weights = np.zeros(28)
    weights[active] = result.x[:-1]
    return dict(version=1, feature_version=FEATURE_VERSION, mean=mean.tolist(), scale=scale.tolist(),
                weights=weights.tolist(), intercept=float(result.x[-1]), joints=list(joints),
                task='TAKE_PULL_action_candidate_not_support', regularization=regularization)


def metrics(model, x, y, threshold=.8):
    predicted = np.array([model.score(row) >= threshold for row in x])
    return dict(samples=len(y), true_positive=int(np.sum(predicted & (y == 1))),
                false_positive=int(np.sum(predicted & (y == 0))),
                true_negative=int(np.sum(~predicted & (y == 0))),
                false_negative=int(np.sum(~predicted & (y == 1))), threshold=threshold,
                unit='sample_level_action_only_not_handover_success')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', nargs='+', required=True, help='Whole training session directories')
    p.add_argument('--test', nargs='+', required=True, help='Whole untouched holdout session directories')
    p.add_argument('--joints', default='1,2,3,4,5,6,7', help='Use 2 or 1,2,3,5,6,7 for axis ablations')
    p.add_argument('--regularization', type=float, default=.01)
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    training = {Path(name).expanduser().resolve() for name in args.train}
    testing = {Path(name).expanduser().resolve() for name in args.test}
    if training & testing:
        p.error('train/test must contain disjoint whole sessions')
    x, y, train_meta, window = load_sessions(sorted(training))
    xt, yt, test_meta, test_window = load_sessions(sorted(testing))
    if window != test_window or {m['sha256'] for m in train_meta} & {m['sha256'] for m in test_meta}:
        p.error('feature mismatch or copied train/test source data')
    data = fit_binary(x, y, joints=tuple(int(j) for j in args.joints.split(',')), regularization=args.regularization)
    data.update(window_seconds=window, training_sessions=train_meta, test_sessions=test_meta,
                control_mode='mit_cartesian_hold', support_verified=None,
                validation_scope='action classification only; replay state machine and verify physical outcomes separately')
    model = LinearPullModel(data)
    data['test_action_metrics'] = metrics(model, xt, yt)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(data, stream, indent=2)
    print(json.dumps(data['test_action_metrics'], indent=2))
    print(f'Saved action candidate model: {out}; no hardware was accessed.')


if __name__ == '__main__':
    main()
