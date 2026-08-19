"""Raw EDF+ recording with annotations and JSON sidecars.

The frontend records the unprocessed packet stream in microvolts.  No filtering
is applied before saving.  Filtering belongs to the backend derived pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from guess_number.utils import atomic_write_json, sha256_file, utc_now_iso

logger = logging.getLogger("frontend.recorder")


def _edf_header_bound(value: float, *, lower: bool) -> float:
    """Round an EDF physical min/max outward so it fits the 8-char header field.

    Values like the -8388608 ADC rail are stored exactly; ordinary EEG values
    are rounded to centi-microvolts outward, which keeps the physical bounds
    valid without clipping the raw samples.
    """
    for decimals in (2, 1, 0):
        factor = 10 ** decimals
        if lower:
            rounded = np.floor(value * factor) / factor
        else:
            rounded = np.ceil(value * factor) / factor
        text = f"{rounded:.{decimals}f}"
        if len(text) <= 8:
            return float(rounded)
    return float(value)


def _physical_bounds(data: np.ndarray) -> tuple[float, float]:
    """Pick EDF physical min/max that covers the actual signal.

    The old fixed -2000..2000 uV range clipped railed channels (for example the
    -8388608 ADC rail seen during hardware diagnosis), which made EDF channel QC
    lie about the original samples.  Raw data must be preserved exactly.
    """
    lo = float(np.min(data))
    hi = float(np.max(data))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = -2000.0, 2000.0
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    return _edf_header_bound(lo, lower=True), _edf_header_bound(hi, lower=False)


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
        # 20 ms data records keep annotation padding tiny and let markers that
        # are >= 20 ms apart live in their own EDF+ record.  At the fixed
        # 250 Hz device rate, one record contains 5 samples.
        record_duration = 0.02
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Forcing a specific record_duration.*")
            writer.setDatarecordDuration(record_duration)
        headers = []
        phys_lo, phys_hi = _physical_bounds(data)
        for label in self.channels:
            headers.append(highlevel.make_signal_header(
                label=label, dimension="uV", sample_frequency=float(self.sfreq),
                physical_min=phys_lo, physical_max=phys_hi))
        writer.setSignalHeaders(headers)
        writer.setPatientCode(self.participant_id)
        writer.setPatientName(self.participant_id)
        writer.setRecordingAdditional(self.recording_additional)
        writer.setStartdatetime(datetime_from_utcnow())
        # pyedflib's writeSamples pads every call to a whole data record.
        # Writing one small chunk per annotation (as the first implementation
        # did) therefore inflated a 10 s file to 40 s.  EDF+ annotations attach
        # to the currently written data record, so we only emit *complete*
        # records up to each annotation's record, then insert the annotation.
        record_samples = max(1, int(round(self.sfreq * record_duration)))
        n_total = int(data.shape[1])
        cursor = 0

        def _write_full_record_until(sample: int) -> None:
            nonlocal cursor
            while cursor + record_samples <= min(n_total, sample):
                writer.writeSamples(data[:, cursor:cursor + record_samples].astype(np.float64))
                cursor += record_samples

        # Multiple project markers can legitimately share one 20 ms record
        # (for example session_start and target/n at t=0).  pyedflib stores one
        # TAL per data record, and a 20 ms record has little annotation-string
        # capacity, so the highest-priority marker (stim_on first) is written.
        # events.jsonl remains the authoritative marker table and keeps every
        # marker individually.
        _priority = {"stim_on": 4, "stim_off": 3, "block_end": 2, "block_start": 1,
                     "fixation_on": 1, "fixation_off": 1}
        grouped: list[tuple[int, list[dict[str, Any]]]] = []
        for ann in sorted(self._annotations, key=lambda x: x["onset_sec"]):
            ann_pos = max(0, min(n_total, int(round(float(ann["onset_sec"]) * self.sfreq))))
            record_start = (ann_pos // record_samples) * record_samples
            if grouped and grouped[-1][0] == record_start:
                grouped[-1][1].append(ann)
            else:
                grouped.append((record_start, [ann]))

        written_annotations = 0
        for record_start, anns in grouped:
            _write_full_record_until(record_start)
            if len(anns) > 1:
                top = max(_priority.get(str(a["description"]).split("/", 1)[0], 0)
                          for a in anns)
                chosen = [a for a in anns
                          if _priority.get(str(a["description"]).split("/", 1)[0], 0) == top]
                ann = chosen[0]
            else:
                ann = anns[0]
            writer.writeAnnotation(float(ann["onset_sec"]), float(ann["duration_sec"]),
                                   str(ann["description"]))
            written_annotations += 1
        _write_full_record_until(n_total)
        if cursor < n_total:
            # Last partial record is padded with zeros by pyedflib; the source
            # sample count and all event indices remain based on the original data.
            writer.writeSamples(data[:, cursor:].astype(np.float64))
            cursor = n_total
        writer.close()
        os.replace(tmp_path, self.path)

        edf_samples = int(np.ceil(n_total / record_samples) * record_samples)

        sidecar = {
            "edf_path": str(self.path.resolve()),
            "sha256": sha256_file(self.path),
            "sfreq": float(self.sfreq),
            "unit": "uV",
            "channels": self.channels,
            "n_samples": int(edf_samples),
            "n_source_samples": int(data.shape[1]),
            "trailing_padding_samples": int(edf_samples - data.shape[1]),
            "n_annotations": int(len(self._annotations)),
            "n_edf_annotations": int(written_annotations),
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "created_utc": utc_now_iso(),
            "note": "Unfiltered packet stream. Reference and ground labels are in session.json.",
        }
        atomic_write_json(self.path.with_suffix(".edf.sidecar.json"), sidecar)
        logger.info("EDF saved: %s (%d EDF samples, %.1f s)",
                    self.path, sidecar["n_samples"], sidecar["n_samples"] / self.sfreq)
        return sidecar


def datetime_from_utcnow() -> Any:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)
