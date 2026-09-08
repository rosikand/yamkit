"""Hardware-free hostile-input and exact-pixel tests for the binary HTTP envelope."""

import json
import math
import struct
import unittest

from yamkit.inference.http_wire import (
    MAGIC,
    MAX_ATTACHMENTS,
    MAX_DEPTH,
    MAX_HEADER_BYTES,
    MAX_WIRE_BYTES,
    VERSION,
    WireError,
    decode_message,
    encode_message,
)

_PREFIX = struct.Struct("!4sBII")


class TestHttpWire(unittest.TestCase):
    @staticmethod
    def raw(tree, lengths=(), body=b""):
        header = json.dumps({"tree": tree, "attachments": list(lengths)}, separators=(",", ":")).encode()
        return _PREFIX.pack(MAGIC, VERSION, len(header), len(body)) + header + body

    def test_roundtrip_preserves_types_and_marker_keys(self):
        value = {"bytes": b"\x00\xffbinary", "empty": b"", "array": [None, True, False, 17, -42, -0.0],
                 "tuple": ("top", "left", "right"), "unicode": "caf\u00e9 \U0001f916", "tree": {"b": ["b", 0]}}
        result = decode_message(encode_message(value))
        self.assertEqual(result, value)
        self.assertIs(type(result["bytes"]), bytes)
        self.assertIs(type(result["tuple"]), tuple)
        self.assertEqual(math.copysign(1, result["array"][-1]), -1)

    def test_realistic_raw_image_sizes(self):
        value = {"images": {name: {"data": bytes(640 * 480 * 3), "height": 480, "width": 640,
                                    "encoding": "rgb8"} for name in ("top", "left_wrist", "right_wrist")},
                 "state": [0.0] * 14, "chunk": [[0.125] * 14] * 30}
        self.assertEqual(decode_message(encode_message(value)), value)

    def test_rejects_encoding_types_nonfinite_and_oversize(self):
        for value in ({"v": float("nan")}, {"v": float("inf")}, {"v": object()}, {1: "value"},
                      {"v": bytearray(b"x")}, {"v": bytes(MAX_WIRE_BYTES)}, {"v": "x" * MAX_HEADER_BYTES},
                      {"v": "\ud800"}, {"v": [b""] * (MAX_ATTACHMENTS + 1)}):
            with self.subTest(kind=str(type(value.get("v")))), self.assertRaises(WireError):
                encode_message(value)

    def test_rejects_version_truncation_trailing_and_invalid_header(self):
        valid = encode_message({"a": b"hello"})
        for wire in (b"", valid[:-1], valid + b"extra", b"NOPE" + valid[4:],
                     valid[:4] + b"\x02" + valid[5:], bytearray(valid), bytes(MAX_WIRE_BYTES + 1)):
            with self.assertRaises(WireError):
                decode_message(wire)
        for header in (b'{"tree":["d",[]],"tree":["d",[]],"attachments":[]}',
                       b'{"tree":["d",[["x",NaN]]],"attachments":[]}',
                       b'{"tree":["d",[["x",1e999]]],"attachments":[]}', b'{}', b'\xff'):
            with self.assertRaises(WireError):
                decode_message(_PREFIX.pack(MAGIC, VERSION, len(header), 0) + header)

    def test_rejects_duplicate_keys_attachments_and_malformed_tags(self):
        invalid = [
            self.raw(["d", [["a", 1], ["a", 2]]]),
            self.raw(["d", []], [1], b"x"),
            self.raw(["d", [["a", ["b", 0]], ["b", ["b", 0]]]], [1], b"x"),
            self.raw(["d", [["a", ["b", True]]]], [1], b"x"),
            self.raw(["d", [["a", ["b", 1]]]], [1], b"x"),
            self.raw(["d", [["a", ["b", 0]]]], [2], b"x"),
            self.raw(["d", [["a", ["x", []]]]]),
            self.raw(["d", [["a", {"not": "tagged"}]]]),
            self.raw(["l", []]), self.raw(["d", [["\ud800", 1]]]),
        ]
        for wire in invalid:
            with self.assertRaises(WireError):
                decode_message(wire)

    def test_rejects_excessive_depth(self):
        value = 0
        tree = 0
        for _ in range(MAX_DEPTH + 2):
            value = [value]
            tree = ["l", [tree]]
        with self.assertRaises(WireError):
            encode_message({"v": value})
        with self.assertRaises(WireError):
            decode_message(self.raw(["d", [["v", tree]]]))

