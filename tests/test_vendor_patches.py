"""Guards for the local patches listed in third_party/i2rt.VERSION."""

import struct

import can
from i2rt.motor_drivers.dm_driver import PassiveEncoderReader


def _parse(counts: int):
    data = struct.pack("!B H h B", 0, counts, 0, 0)
    return PassiveEncoderReader._parse_encoder_message(None, can.Message(arbitration_id=0x50F, data=data))


def test_encoder_counts_wrap_negative():
    pos, _, _ = _parse(4091)  # -5 counts
    assert abs(pos - (-5 * 2 * 3.141592653589793 / 4096)) < 1e-6
    pos, _, _ = _parse(20)
    assert pos > 0


def test_wrap_correction_uses_joint_limits():
    """A base parked at +183° that reads -177° (inside ±π, but past the -150° stop) is unwrapped."""
    import numpy as np
    from i2rt.robots.get_robot import wrap_correction

    two_pi = 2 * np.pi
    base = np.array([-2.618 - 0.15, 3.1416 + 0.15])  # YAM joint 1 limits with the SDK's safety buffer
    assert wrap_correction(-3.0885, base) == two_pi  # -177° -> +183°, which is inside the limits
    assert wrap_correction(3.19, base) == 0.0  # +183° already fine
    assert wrap_correction(-2.5, base) == 0.0  # legal reading, untouched
    elbow = np.array([0.0 - 0.15, 3.1416 + 0.15])
    assert wrap_correction(-3.0, elbow) == two_pi  # -172° on a 0..180° joint can only be +188° (inside the buffered limit)
    assert wrap_correction(-1.0, elbow) == 0.0  # below the limit and shifting a turn does not help: leave it (the SDK then reports the violation)
    # no limits known (e.g. the gripper motor): original ±π behaviour
    assert wrap_correction(-3.3) == two_pi and wrap_correction(3.3) == -two_pi and wrap_correction(3.0) == 0.0
