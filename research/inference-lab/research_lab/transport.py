"""Local codec/replica experiment. Byte counts exclude TCP/TLS framing."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import cbor2
import zstandard as zstd

from .common import canonical, digest


class ResyncRequired(ValueError):
    pass


def state_hash(state):
    return digest(sorted(state.items()))


@dataclass
class Replica:
    context: str
    state: dict = field(default_factory=dict)
    sequence: int = 0
    last_packet: str | None = None
    allowed_ids: set | None = None

    def apply(self, packet):
        packet_digest = digest(packet)
        if self.last_packet == packet_digest:
            return "duplicate"
        if packet["context"] != self.context:
            raise ResyncRequired("context mismatch: install the verified dictionary first")
        if packet["kind"] == "snapshot":
            if packet["sequence"] < self.sequence or (self.last_packet is not None and packet["sequence"] == self.sequence):
                raise ResyncRequired("stale/conflicting snapshot")
            state = {int(k): v for k, v in packet["state"].items()}
            if state_hash(state) != packet["result_hash"]:
                raise ResyncRequired("snapshot checksum mismatch")
        else:
            if packet["kind"] != "delta":
                raise ResyncRequired("unknown packet kind")
            if packet["context"] != self.context or packet["base_hash"] != state_hash(self.state):
                raise ResyncRequired("context/base mismatch")
            if packet["sequence"] != self.sequence + 1:
                raise ResyncRequired("out-of-order sequence")
            state = copy.deepcopy(self.state)
            for k, value in packet["updates"]:
                if self.allowed_ids is not None and int(k) not in self.allowed_ids:
                    raise ResyncRequired("unknown identifier")
                state[int(k)] = value
            for k in packet["deletes"]:
                state.pop(int(k), None)
            if state_hash(state) != packet["result_hash"]:
                raise ResyncRequired("delta checksum mismatch")
        if self.allowed_ids is not None and not set(state).issubset(self.allowed_ids):
            raise ResyncRequired("unknown identifier")
        if any(type(value) is not int or value not in (0, 1, 2) for value in state.values()):
            raise ResyncRequired("invalid health code")
        # Commit only after all checks. No partially applied state on failure.
        self.state = state
        self.sequence = packet["sequence"]
        self.last_packet = packet_digest
        return "applied"


def snapshot(state, context, sequence=0):
    return {"kind": "snapshot", "context": context, "sequence": sequence,
            "state": state, "result_hash": state_hash(state)}


def delta(before, after, context, sequence):
    return {"kind": "delta", "context": context, "sequence": sequence,
            "base_hash": state_hash(before), "result_hash": state_hash(after),
            "updates": [[k, after[k]] for k in sorted(after) if k not in before or before[k] != after[k]],
            "deletes": sorted(set(before) - set(after))}


def encode(packet, codec, compressed=True):
    raw = canonical(packet) if codec == "json" else cbor2.dumps(packet, canonical=True)
    return zstd.ZstdCompressor(level=3).compress(raw) if compressed else raw


def decode(payload, codec, compressed=True):
    import json
    raw = zstd.ZstdDecompressor().decompress(payload, max_output_size=128 * 1024 * 1024) if compressed else payload
    return json.loads(raw) if codec == "json" else cbor2.loads(raw)
