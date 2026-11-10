import os
import sys

import numpy as np
import numpy.testing as npt
from cameras.sdk_loop import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from aruco_camera_jacobian_servo_sdk_dryrun import image_msg_to_bgr


def _image(width, height, encoding, data, step):
    msg = Image()
    msg.width = width
    msg.height = height
    msg.encoding = encoding
    msg.step = step
    msg.data = bytes(data)
    return msg


def test_bgr8_image_msg_to_bgr():
    msg = _image(1, 1, "bgr8", [1, 2, 3], 3)
    out = image_msg_to_bgr(msg)
    npt.assert_array_equal(out, np.array([[[1, 2, 3]]], dtype=np.uint8))


def test_rgb8_image_msg_to_bgr():
    msg = _image(1, 1, "rgb8", [1, 2, 3], 3)
    out = image_msg_to_bgr(msg)
    npt.assert_array_equal(out, np.array([[[3, 2, 1]]], dtype=np.uint8))


def test_mono8_image_msg_to_bgr():
    msg = _image(1, 1, "mono8", [7], 1)
    out = image_msg_to_bgr(msg)
    npt.assert_array_equal(out, np.array([[[7, 7, 7]]], dtype=np.uint8))
