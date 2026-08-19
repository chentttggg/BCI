"""LSL outlets for EEG stream and experiment markers."""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

logger = logging.getLogger("frontend.lsl")


class NullMarkerOutlet:
    def push(self, marker: str) -> float:
        return time.monotonic()

    def close(self) -> None:
        pass


class NullEEGOutlet:
    def push_chunk(self, samples: Any) -> None:
        pass

    def close(self) -> None:
        pass


class MarkerOutlet:
    def __init__(self, outlet: Any, local_clock: Any) -> None:
        self.outlet = outlet
        self.local_clock = local_clock

    def push(self, marker: str) -> float:
        timestamp = self.local_clock()
        self.outlet.push_sample([str(marker)], timestamp)
        return float(timestamp)

    def close(self) -> None:
        pass


class EEGOutlet:
    def __init__(self, outlet: Any, local_clock: Any) -> None:
        self.outlet = outlet
        self.local_clock = local_clock

    def push_chunk(self, samples: Any) -> None:
        n = samples.shape[1]
        stamps = [self.local_clock() for _ in range(n)]
        rows = [[float(v) for v in samples[:, i]] for i in range(n)]
        self.outlet.push_chunk(rows, stamps)

    def close(self) -> None:
        pass


def create_marker_outlet(name: str = "GuessNumberMarkers", source_id: str | None = None) -> Any:
    try:
        from pylsl import StreamInfo, StreamOutlet, local_clock
    except Exception:
        logger.warning("pylsl unavailable; markers are only written to EDF/JSONL")
        return NullMarkerOutlet()

    info = StreamInfo(name=name, type="Markers", channel_count=1,
                      nominal_srate=0.0, channel_format="string",
                      source_id=source_id or f"guess-number-{uuid.uuid4().hex[:12]}")
    return MarkerOutlet(StreamOutlet(info, chunk_size=1, max_buffered=360), local_clock)


def create_eeg_outlet(sfreq: float, channels: list[str],
                      name: str = "BrainSync-EEG", source_id: str | None = None) -> Any:
    try:
        from pylsl import StreamInfo, StreamOutlet, local_clock
    except Exception:
        logger.warning("pylsl unavailable; EEG LSL forwarding disabled")
        return NullEEGOutlet()

    info = StreamInfo(name=name, type="EEG", channel_count=len(channels),
                      nominal_srate=float(sfreq), channel_format="float32",
                      source_id=source_id or f"brainsync-guess-{uuid.uuid4().hex[:12]}")
    ch = info.desc().append_child("channels")
    for i, label in enumerate(channels):
        c = ch.append_child("channel")
        c.append_child_value("label", label)
        c.append_child_value("unit", "microvolts")
        c.append_child_value("type", "EEG")
        c.append_child_value("index", str(i))
    return EEGOutlet(StreamOutlet(info, chunk_size=0, max_buffered=360), local_clock)
