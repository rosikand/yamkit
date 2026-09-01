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
