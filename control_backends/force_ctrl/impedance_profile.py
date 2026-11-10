"""Single source of truth for the validated Nero Cartesian impedance profile."""

from __future__ import annotations

import numpy as np


VALIDATED_CARTESIAN_KP = np.array(
    [200.0, 200.0, 200.0, 5.0, 5.0, 5.0], dtype=np.float64
)
VALIDATED_CARTESIAN_KD = np.array(
    [5.0, 5.0, 5.0, 0.2, 0.2, 0.2], dtype=np.float64
)
VALIDATED_JOINT_WEIGHTS = np.array(
    [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=np.float64
)
VALIDATED_COULOMB = np.array(
    [0.0, 0.4, 0.0, 0.4, 0.0, 0.0, 0.0], dtype=np.float64
)
VALIDATED_GRAVITY_FACTORS = np.array(
    [1.0, 1.25, 1.0, 1.25, 1.0, 1.0, 1.5], dtype=np.float64
)
VALIDATED_TASK_TORQUE_LIMIT = 10.0
VALIDATED_TOTAL_TORQUE_LIMIT = 14.0
VALIDATED_VELOCITY_SIGNS = np.array(
    [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0], dtype=np.float64
)
