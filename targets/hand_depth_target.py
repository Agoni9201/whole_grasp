"""Hand depth target source.

Uses depth camera + hand detection to publish hand position in base frame.

Target:
    target_tcp = hand_center - approach_axis * desired_distance

Approach axis priority:
    1. line_of_sight: normalize(hand_center - tcp_now)
    2. tcp_x_legacy: current TCP-X axis
    3. palm_normal: MediaPipe/depth estimated palm normal

Reference: REFACTOR_PLAN.md Section 7.3
TODO: Implement in Step 10 after ArUco servo is stable.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class HandDepthTarget:
    """Hand detection target source using depth camera.

    TODO: Implement in Step 10.
    """

    def __init__(self) -> None:
        pass

    def detect(
        self,
        image: npt.NDArray[np.uint8],
        depth: Optional[npt.NDArray[np.uint16]] = None,
    ) -> Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Detect hand and return (hand_center_base, approach_axis_base).

        TODO: Implement in Step 10.
        """
        raise NotImplementedError("HandDepthTarget not yet implemented")
