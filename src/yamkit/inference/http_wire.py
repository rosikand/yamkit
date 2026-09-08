"""Bounded inert binary messages for the authenticated HTTP inference transport.

Public interface: ``encode_message(dict) -> bytes`` and ``decode_message(bytes)
-> dict``. Supports inert JSON values, bytes and explicitly tagged tuples (the
existing readiness metadata contains tuples). No objects are instantiated from
wire-supplied names. HTTP and authentication belong to the caller; image bytes are never compressed.

Framing: magic, version, JSON-header length, attachment-body length, JSON header,
then concatenated raw attachments. Total wire size is at most 4 MiB. The tagged
JSON tree makes ordinary dictionaries unambiguous even when they contain keys
that resemble codec markers. Each attachment is referenced exactly once.
"""

from __future__ import annotations

import json
import math
import struct

WIRE_VERSION = VERSION = 1
WIRE_CODEC = "yamkit-binary-v1"
MAX_MESSAGE_BYTES = MAX_WIRE_BYTES = 4 * 1024 * 1024
MAX_HEADER_BYTES = 256 * 1024
MAX_ATTACHMENTS = 128
MAX_DEPTH = 32
MAX_NODES = 100_000
MAGIC = b"YAMW"
_PREFIX = struct.Struct("!4sBII")


class WireError(ValueError):
    """Malformed or unsupported wire data; messages never echo payloads."""


def _reject(message: str) -> None:
    raise WireError(message)


def _finite(value: float) -> float:
    if not math.isfinite(value):
        _reject("Nonfinite numbers are forbidden")
    return value


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _reject("Duplicate JSON keys are forbidden")
        result[key] = value
    return result


def encode_message(message: dict) -> bytes:
    """Encode one dictionary without pickle, implicit coercions or compression."""
    if type(message) is not dict:
        _reject("Message root must be a dictionary")
    attachments = []
    body_bytes = 0
    nodes = 0
    string_bytes = 0

    def string(value: str) -> str:
        nonlocal string_bytes
        if len(value) > MAX_HEADER_BYTES:
            _reject("String exceeds header bound")
        try:
            string_bytes += len(value.encode("utf-8"))
        except UnicodeError:
            _reject("Strings must contain valid Unicode")
        if string_bytes > MAX_HEADER_BYTES:
            _reject("Strings exceed header bound")
        return value

    def visit(value, depth=0):
        nonlocal nodes, body_bytes
        nodes += 1
        if depth > MAX_DEPTH or nodes > MAX_NODES:
            _reject("Message structure exceeds bounds")
        kind = type(value)
        if value is None or kind in (bool, int):
            return value
        if kind is float:
            return _finite(value)
        if kind is str:
            return string(value)
        if kind is bytes:
            body_bytes += len(value)
            if body_bytes > MAX_WIRE_BYTES or len(attachments) >= MAX_ATTACHMENTS:
                _reject("Binary attachments exceed bounds")
            index = len(attachments)
            attachments.append(value)
            return ["b", index]
        if kind in (list, tuple):
            if len(value) > MAX_NODES - nodes:
                _reject("Message structure exceeds bounds")
            return ["l" if kind is list else "t", [visit(item, depth + 1) for item in value]]
        if kind is dict:
            if len(value) > MAX_NODES - nodes:
                _reject("Message structure exceeds bounds")
            pairs = []
            for key, item in value.items():
                if type(key) is not str:
                    _reject("Dictionary keys must be strings")
                pairs.append([string(key), visit(item, depth + 1)])
            return ["d", pairs]
        _reject("Unsupported value type")

    tree = visit(message)
    try:
        header = json.dumps({"tree": tree, "attachments": [len(item) for item in attachments]},
                            ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (ValueError, OverflowError, UnicodeError):
        _reject("Header cannot be encoded")
    if len(header) > MAX_HEADER_BYTES or _PREFIX.size + len(header) + body_bytes > MAX_WIRE_BYTES:
        _reject("Encoded message exceeds bounds")
    return _PREFIX.pack(MAGIC, VERSION, len(header), body_bytes) + header + b"".join(attachments)


def decode_message(wire: bytes) -> dict:
    """Decode exactly one bounded message; reject missing or trailing data."""
    if type(wire) is not bytes or not _PREFIX.size <= len(wire) <= MAX_WIRE_BYTES:
        _reject("Wire message type or length is invalid")
    magic, version, header_bytes, body_bytes = _PREFIX.unpack_from(wire)
    if magic != MAGIC or version != VERSION:
        _reject("Wire magic or version is unsupported")
    if not 1 <= header_bytes <= MAX_HEADER_BYTES or body_bytes > MAX_WIRE_BYTES:
        _reject("Wire section lengths exceed bounds")
    if _PREFIX.size + header_bytes + body_bytes != len(wire):
        _reject("Wire message is truncated or has trailing data")
    header_end = _PREFIX.size + header_bytes
    try:
        header = json.loads(wire[_PREFIX.size:header_end].decode("utf-8"), object_pairs_hook=_object,
                            parse_constant=lambda value: _reject("Nonfinite JSON constants are forbidden"))
    except (ValueError, UnicodeError, RecursionError):
        _reject("JSON header is malformed")
    if type(header) is not dict or set(header) != {"tree", "attachments"}:
        _reject("JSON header fields are invalid")
    lengths = header["attachments"]
    if type(lengths) is not list or len(lengths) > MAX_ATTACHMENTS or any(
        type(length) is not int or not 0 <= length <= MAX_WIRE_BYTES for length in lengths
    ) or sum(lengths) != body_bytes:
        _reject("Attachment lengths are invalid")
    attachments = []
    offset = header_end
    for length in lengths:
        attachments.append(wire[offset:offset + length])
        offset += length
    used = set()
    nodes = 0

    def visit(value, depth=0):
        nonlocal nodes
        nodes += 1
        if depth > MAX_DEPTH or nodes > MAX_NODES:
            _reject("Message structure exceeds bounds")
        kind = type(value)
        if value is None or kind in (bool, int):
            return value
        if kind is str:
            try:
                value.encode("utf-8")
            except UnicodeError:
                _reject("Strings must contain valid Unicode")
            return value
        if kind is float:
            return _finite(value)
        if kind is not list or len(value) != 2 or type(value[0]) is not str:
            _reject("Tagged value is malformed")
        tag, payload = value
        if tag == "b":
            if type(payload) is not int or not 0 <= payload < len(attachments) or payload in used:
                _reject("Attachment reference is invalid or duplicated")
            used.add(payload)
            return attachments[payload]
        if tag in ("l", "t"):
            if type(payload) is not list:
                _reject("Sequence payload is malformed")
            items = [visit(item, depth + 1) for item in payload]
            return items if tag == "l" else tuple(items)
        if tag == "d":
            if type(payload) is not list:
                _reject("Dictionary payload is malformed")
            result = {}
            for pair in payload:
                if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str:
                    _reject("Dictionary pair is malformed")
                key, item = pair
                try:
                    key.encode("utf-8")
                except UnicodeError:
                    _reject("Dictionary keys must contain valid Unicode")
                if key in result:
                    _reject("Duplicate dictionary keys are forbidden")
                result[key] = visit(item, depth + 1)
            return result
        _reject("Tagged value is unsupported")

    result = visit(header["tree"])
    if type(result) is not dict or used != set(range(len(attachments))):
        _reject("Message root or attachment coverage is invalid")
    return result

