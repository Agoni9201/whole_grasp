"""Verify held-object dispatch without running preflight, cameras or arm control."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / 'scripts/collect_handover_shadow.sh'


@pytest.fixture
def launcher(tmp_path):
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    conda_bin = tmp_path / 'conda/bin'
    conda_bin.mkdir(parents=True)
    captured = tmp_path / 'arguments.txt'
    preflight = tmp_path / 'preflight.txt'
    for path, variable in ((fake_bin / 'bash', 'LAUNCH_CAPTURE'),
                           (conda_bin / 'python', 'PREFLIGHT_CAPTURE')):
        path.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$' + variable + '"\n')
        path.chmod(0o755)
    env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ['PATH'],
               CONDA_PREFIX=str(conda_bin.parent), LAUNCH_CAPTURE=str(captured),
               PREFLIGHT_CAPTURE=str(preflight))

    def run(arguments):
        result = subprocess.run(['/bin/bash', str(SCRIPT), *arguments],
                                cwd=tmp_path, env=env, capture_output=True, text=True)
        return result, captured, preflight
    return run


@pytest.mark.parametrize('arguments,side,phase', [
    ([], 'Left', 'all'),
    (['Right'], 'Right', 'all'),
    (['Left', '--held'], 'Left', 'servo'),
    (['Right', '--held'], 'Right', 'servo'),
    (['--held', 'Right'], 'Right', 'servo'),
    (['--held'], 'Left', 'servo'),
])
def test_held_dispatch_and_production_cli_validation(launcher, monkeypatch, arguments, side, phase):
    result, captured, preflight = launcher(arguments)
    assert result.returncode == 0, result.stderr
    forwarded = captured.read_text().splitlines()
    assert forwarded[0] == 'scripts/run_grasp_safe.sh'
    assert preflight.read_text().splitlines() == [
        'tools/check_handover_usb.py', '--wrist-serial', 'CP0T963000S4']
    monkeypatch.chdir(PROJECT)
    monkeypatch.setattr(sys, 'argv', ['main.py', *forwarded[1:]])
    from real_nero_grasp.cli import parse_args
    args = parse_args()  # Parsing only; runtime.main is never called.
    assert args.phase == phase
    assert args.fusion_receiver_side == side
    assert args.fusion_handover == 'shadow'
    assert args.fusion_pull_model is None
    assert args.delivery_distance == .25
    assert args.fusion_ready_distance == .28
    assert args.fusion_ready_seconds == .15


@pytest.mark.parametrize('arguments', [
    ['Left', 'Right'], ['Left', 'Left'], ['right'], ['--held', '--held'],
    ['Right', '--unknown'], ['Right', '--held', 'unexpected'],
])
def test_invalid_arguments_exit_before_preflight_or_control(launcher, arguments):
    result, captured, preflight = launcher(arguments)
    assert result.returncode == 2
    assert not captured.exists()
    assert not preflight.exists()


def test_help_documents_held_mode_without_preflight_or_control(launcher):
    result, captured, preflight = launcher(['--help'])
    assert result.returncode == 0
    assert '--held' in result.stdout and 'skipping initial pose, opening and grasp' in result.stdout
    assert not captured.exists()
    assert not preflight.exists()


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
def test_physical_grasp_and_partition_reach_production_parser(launcher, monkeypatch, split):
    result, captured, _ = launcher(['Right', '--held', '--group', 'unit_test_grasp_A', '--split', split])
    assert result.returncode == 0, result.stderr
    forwarded = captured.read_text().splitlines()[1:]
    monkeypatch.chdir(PROJECT)
    monkeypatch.setattr(sys, 'argv', ['main.py', *forwarded])
    from real_nero_grasp.cli import parse_args
    args = parse_args()
    assert args.phase == 'servo' and args.fusion_handover == 'shadow'
    assert args.fusion_collection_group == 'unit_test_grasp_A'
    assert args.fusion_collection_split == split
    assert args.save_dir == f'logs/handover_dataset_20261109/{split}/unit_test_grasp_A'
    assert args.fusion_pull_model is None


@pytest.mark.parametrize('arguments', [
    ['--group'], ['--split'], ['--group', 'grasp01'], ['--split', 'train'],
    ['--group', '../grasp01', '--split', 'train'],
    ['--group', 'grasp01', '--split', 'testing'],
    ['--group', 'grasp01', '--split', 'train', '--group', 'grasp02'],
    ['--group', 'grasp01', '--split', 'train', '--split', 'test'],
])
def test_bad_collection_metadata_never_reaches_hardware(launcher, arguments):
    result, captured, preflight = launcher(arguments)
    assert result.returncode == 2
    assert not captured.exists() and not preflight.exists()


def test_existing_physical_grasp_in_another_partition_exits_before_preflight(launcher, tmp_path, monkeypatch):
    copied = tmp_path/'project/scripts/collect_handover_shadow.sh'
    copied.parent.mkdir(parents=True)
    copied.write_text(SCRIPT.read_text())
    (tmp_path/'project/logs/handover_dataset_20261109/train/same_grasp').mkdir(parents=True)
    monkeypatch.setattr(sys.modules[__name__], 'SCRIPT', copied)
    result, captured, preflight = launcher([
        'Right', '--held', '--group', 'same_grasp', '--split', 'test'])
    assert result.returncode == 2
    assert 'already exists in train' in result.stderr
    assert not captured.exists() and not preflight.exists()
