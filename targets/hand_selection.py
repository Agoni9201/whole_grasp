"""Conservative RGB candidate checks and fixed left/right hand roles."""
from __future__ import annotations

import numpy as np


def filter_hand_observations(observations, image_shape, *, fingertips=False):
    """Reject unusable candidates BEFORE counting hands; never choose by size.

    These are geometry/side-confidence checks, not proof that an image contains
    a real hand. Two plausible hands remain two hands, even at different sizes.
    """
    h, w = image_shape[:2]
    accepted, rejected = [], []
    for index, observation in enumerate(observations):
        reason = None
        points = np.asarray(observation.landmarks_normalized, dtype=float)
        score = float(observation.handedness_score)
        if observation.handedness not in ('Left', 'Right') or not np.isfinite(score) or score < .7:
            reason = 'hand side uncertain'
        elif points.shape != (21, 3) or not np.isfinite(points).all():
            reason = 'invalid hand landmarks'
        else:
            indices = [0, 5, 9, 13, 17] + ([4, 8] if fingertips else [])
            xy = points[indices, :2]
            if np.any(xy < 0.) or np.any(xy > 1.):
                reason = 'fingers or palm outside image'
            elif np.linalg.norm((points[5, :2] - points[17, :2]) * [w-1, h-1]) < 15.:
                reason = 'palm too small or edge-on'
        if reason:
            rejected.append(dict(index=index, reason=reason, side=observation.handedness))
        else:
            accepted.append(observation)
    return accepted, rejected


class HandRoles:
    """Lock the first sole, consistent hand as receiver; never swap on dropout."""

    def __init__(self):
        self.receiver = None
        self.candidate = None
        self.count = 0
        self.presence = 'unknown'
        self.control_observations = []

    @property
    def control(self):
        return None if self.receiver is None else ('Right' if self.receiver == 'Left' else 'Left')

    def select(self, accepted, rejected):
        self.control_observations = []
        if self.receiver is None:
            if len(accepted) != 1:
                self.candidate, self.count, self.presence = None, 0, 'unknown'
                return None
            observation = accepted[0]
            side = observation.handedness
            self.count = self.count + 1 if self.candidate == side else 1
            self.candidate = side
            self.presence = 'acquiring'
            if self.count < 3:
                return None
            self.receiver = side
        receivers = [o for o in accepted if o.handedness == self.receiver]
        self.control_observations = [o for o in accepted if o.handedness == self.control]
        if len(receivers) == 1:
            self.presence = 'visible'
            return receivers[0]
        # Rejected/uncertain receiver detections are not proof of occlusion.
        ambiguous = len(receivers) > 1 or any(
            r['side'] == self.receiver or r['reason'] == 'hand side uncertain' for r in rejected)
        self.presence = 'unknown' if ambiguous else 'absent'
        return None
