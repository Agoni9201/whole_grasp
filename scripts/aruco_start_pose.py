"""Shared ArUco visual-servo start posture."""

from __future__ import annotations

import argparse

import numpy as np


DEFAULT_ARUCO_START_Q_DEG = np.array(
    [3.73, -3.32, 7.29, 101.74, -3.76, 10.53, 29.42],
    dtype=np.float64,
)


def parse_deg7(text: str) -> np.ndarray:
    values = [float(item) for item in text.split(",")]
    if len(values) != 7:
        raise argparse.ArgumentTypeError("expected seven comma-separated degree values")
    return np.asarray(values, dtype=np.float64)
