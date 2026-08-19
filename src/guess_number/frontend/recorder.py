"""Raw EDF+ recording with annotations and JSON sidecars.

The frontend records the unprocessed packet stream in microvolts.  No filtering
is applied before saving.  Filtering belongs to the backend derived pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .utils import atomic_write_json, sha256_file, utc_now_iso

logger = logging.getLogger("frontend.recorder")


@dataclass
class RawEDFRecorder:
    path: Path
    sfreq: float
    channels: list[str]
    participant_id: str = "P01"
    session_id: str = "001"
    recording_additional: str = "GuessNumber-P300-raw-EEG-unfiltered"
    _chunks: list[np.ndarray] = field(default_factory=list, repr=False)
    _annotations: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _sample_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _closed: bool = False
    _data: np.ndarray | None = None

    def write(self, samples: np.ndarray) -> None:
        """samples: (n_channels, n_samples) float32 microvolts."""
        if self._closed:
            raise RuntimeError("recorder already closed")
        samples = np.asarray(samples, dtype=np.float64)
        with self._lock:
            self._chunks.append(samples.copy())
            self._sample_count += int(samples.shape[1])

    def add_annotation(self, onset_sec: float, duration_sec: float, description: str) -> None:
        if self._closed:
            return
        with self._lock:
            self._annotations.append({
                "onset_sec": float(onset_sec),
                "duration_sec": float(duration_sec),
                "description": str(description),
            })

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._sample_count

    def get_data(self) -> np.ndarray:
        if self._data is None:
            raise RuntimeError("recorder has not been closed yet")
        return self._data.copy()

    def close(self) -> dict[str, Any]:
        if self._closed:
            return {}
        self._closed = True
        import pyedflib
        from pyedflib import highlevel

        if not self._chunks:
            raise RuntimeError(f"No EEG samples recorded: {self.path}")
        data = np.concatenate(self._chunks, axis=1)
        self._data = data
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        writer = pyedflib.EdfWriter(str(tmp_path), len(self.channels),
                                    pyedflib.FILETYPE_EDFPLUS)
        headers = []
        for label in self.channels:
            headers.append(highlevel.make_signal_header(
                label=label, dimension="uV", sample_frequency=float(self.sfreq),
                physical_min=-2000.0, physical_max=2000.0))
        writer.setSignalHeaders(headers)
        writer.setPatientCode(self.participant_id)
        writer.setPatientName(self.participant_id)
        writer.setRecordingAdditional(self.recording_additional)
        writer.setStartdatetime(datetime_from_utcnow())
        # pyedflib attaches an annotation to the currently written data record.
        # Therefore samples are written up to each annotation onset, then the
        # annotation is inserted, then the remaining samples are written.
        n_total = int(data.shape[1])
        cursor = 0
        for ann in sorted(self._annotations, key=lambda x: x["onset_sec"]):
            next_pos = int(round(float(ann["onset_sec"]) * self.sfreq))
            next_pos = max(cursor, min(n_total, next_pos))
            if next_pos > cursor:
                writer.writeSamples(data[:, cursor:next_pos].astype(np.float64))
                cursor = next_pos
            writer.writeAnnotation(float(ann["onset_sec"]), float(ann["duration_sec"]),
                                   str(ann["description"]))
        if cursor < n_total:
            writer.writeSamples(data[:, cursor:].astype(np.float64))
        writer.close()
        os.replace(tmp_path, self.path)

        sidecar = {
            "edf_path": str(self.path.resolve()),
            "sha256": sha256_file(self.path),
            "sfreq": float(self.sfreq),
            "unit": "uV",
            "channels": self.channels,
            "n_samples": int(data.shape[1]),
            "n_annotations": int(len(self._annotations)),
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "created_utc": utc_now_iso(),
            "note": "Unfiltered packet stream. Reference and ground labels are in session.json.",
        }
        atomic_write_json(self.path.with_suffix(".edf.sidecar.json"), sidecar)
        logger.info("EDF saved: %s (%d samples, %.1f s)",
                    self.path, sidecar["n_samples"], sidecar["n_samples"] / self.sfreq)
        return sidecar


def datetime_from_utcnow() -> Any:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)
