"""Experiment controller: acquisition, stimulus timing, markers, raw recording."""
from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .paradigm import Paradigm, TimelineEvent, TimelineRunner
from .recorder import RawEDFRecorder
from .utils import append_jsonl, atomic_write_json, sha256_file, utc_now_iso

logger = logging.getLogger("frontend.experiment")

VisualCallback = Callable[[str | None], None]
StatusCallback = Callable[[dict[str, Any]], None]


def git_commit_short() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL, text=True)
        return out.strip() or None
    except Exception:
        return None


@dataclass
class ExperimentConfig:
    participant_id: str = "P01"
    session_id: str = "001"
    run_id: str = "001"
    target_number: int = 7
    sfreq: int = 500
    gain: str = "Gain24"
    channels: list[str] = field(default_factory=lambda: ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"])
    ref_label: str = "A1"
    gnd_label: str = "A1"
    acquisition_mode: str = "mock"
    output_dir: str | Path = "data/raw"
    seed: int = 0


@dataclass
class SessionPaths:
    edf: Path
    events: Path
    session: Path


def make_session_paths(cfg: ExperimentConfig, output_dir: Path) -> SessionPaths:
    stem = f"sub-{cfg.participant_id}_ses-{cfg.session_id}_task-guessnumber_run-{cfg.run_id}_eeg"
    return SessionPaths(edf=output_dir / (stem + ".edf"),
                        events=output_dir / (stem + "_events.jsonl"),
                        session=output_dir / (stem + "_session.json"))


class ExperimentController:
    """Owns one recording session and drives the paradigm timeline."""

    def __init__(self, cfg: ExperimentConfig, paradigm: Paradigm,
                 acquirer: Any, marker_outlet: Any, eeg_outlet: Any) -> None:
        self.cfg = cfg
        self.paradigm = paradigm
        self.acquirer = acquirer
        self.marker_outlet = marker_outlet
        self.eeg_outlet = eeg_outlet
        self.paths = make_session_paths(cfg, Path(cfg.output_dir))
        self.paths.edf.parent.mkdir(parents=True, exist_ok=True)
        self.recorder = RawEDFRecorder(self.paths.edf, float(cfg.sfreq), cfg.channels,
                                       participant_id=cfg.participant_id,
                                       session_id=cfg.session_id)
        self.visual_callback: VisualCallback | None = None
        self.status_callback: StatusCallback | None = None
        self._start_monotonic: float | None = None
        self._runner: TimelineRunner | None = None
        self._stopped = False
        self._finalized = False
        self._stop_reason = ""
        self._event_seq = 0
        self._sample_count_at_start = 0
        self._stim_count = 0
        self._current_visual: str | None = None
        self._latest_samples: list[np.ndarray] = []
        self._latest_lock = __import__("threading").Lock()
        self.session_metadata: dict[str, Any] = {}
        self._initialise_session_metadata()

    def _initialise_session_metadata(self) -> None:
        self.session_metadata = {
            "participant_id": self.cfg.participant_id,
            "session_id": self.cfg.session_id,
            "run_id": self.cfg.run_id,
            "target_number": int(self.cfg.target_number),
            "sfreq": int(self.cfg.sfreq),
            "gain": self.cfg.gain,
            "channels": list(self.cfg.channels),
            "ref_label": self.cfg.ref_label,
            "gnd_label": self.cfg.gnd_label,
            "acquisition_mode": self.cfg.acquisition_mode,
            "paradigm": {
                "blocks": self.paradigm.cfg.blocks,
                "repetitions": self.paradigm.cfg.repetitions,
                "stimulus_s": self.paradigm.cfg.stimulus_s,
                "blank_s": self.paradigm.cfg.blank_s,
                "fixation_s": self.paradigm.cfg.fixation_s,
                "inter_block_s": self.paradigm.cfg.inter_block_s,
                "seed": self.paradigm.cfg.seed,
            },
            "created_utc": utc_now_iso(),
            "git_commit": git_commit_short(),
        }
        atomic_write_json(self.paths.session, self.session_metadata)

    def start(self) -> None:
        if self._start_monotonic is not None:
            return
        self._sample_count_at_start = int(self.acquirer.sample_count)
        self._runner = TimelineRunner(self.paradigm, self._on_timeline_event)
        self.acquirer.start(self._on_eeg_chunk)
        self._start_monotonic = time.monotonic()
        self._push_marker("session_start")
        self._push_marker(f"target/{self.cfg.target_number}")
        self._set_visual(None)
        self._update_status("RUNNING")

    def _on_eeg_chunk(self, samples: np.ndarray, start_sample: int) -> None:
        try:
            self.recorder.write(samples)
        except Exception:
            logger.exception("raw recorder write failed")
        try:
            self.eeg_outlet.push_chunk(samples)
        except Exception:
            logger.debug("LSL EEG push failed", exc_info=True)
        with self._latest_lock:
            self._latest_samples.append(np.asarray(samples, dtype=np.float32))
            # Keep only the most recent ~2 seconds for the GUI.
            total = sum(int(x.shape[1]) for x in self._latest_samples)
            while len(self._latest_samples) > 1 and total > int(self.cfg.sfreq * 2):
                total -= int(self._latest_samples[0].shape[1])
                self._latest_samples.pop(0)

    def get_recent_samples(self, max_samples: int | None = None) -> np.ndarray:
        with self._latest_lock:
            if not self._latest_samples:
                return np.zeros((len(self.cfg.channels), 1), dtype=np.float32)
            data = np.concatenate(self._latest_samples, axis=1)
        if max_samples is not None and data.shape[1] > max_samples:
            data = data[:, -max_samples:]
        return data

    def _set_visual(self, visual: str | None) -> None:
        if visual == self._current_visual:
            return
        self._current_visual = visual
        if self.visual_callback is not None:
            try:
                self.visual_callback(visual)
            except Exception:
                logger.exception("visual callback failed")

    def _push_marker(self, text: str, duration_sec: float = 0.0,
                     sample_onset: int | None = None) -> float:
        try:
            lsl_sec = self.marker_outlet.push(text)
        except Exception:
            lsl_sec = float("nan")
        if sample_onset is None:
            sample_onset = int(self.acquirer.sample_count)
        self.recorder.add_annotation(
            max(0.0, sample_onset / self.cfg.sfreq), duration_sec, text)
        return lsl_sec

    def _on_timeline_event(self, event: TimelineEvent, now_sec: float) -> None:
        marker_text = ""
        duration = 0.0
        visual: str | None = None
        if event.kind == "session_start":
            marker_text = "session_start"
        elif event.kind == "block_start":
            marker_text = f"block_start/{event.block}"
        elif event.kind == "fixation_on":
            marker_text = f"fixation_on/{event.block}"
            visual = "+"
            duration = event.duration_sec
        elif event.kind == "fixation_off":
            marker_text = f"fixation_off/{event.block}"
            visual = None
        elif event.kind == "stim_on":
            marker_text = f"stim_on/{event.number}"
            visual = str(event.number)
            duration = event.duration_sec
            self._stim_count += 1
        elif event.kind == "stim_off":
            marker_text = f"stim_off/{event.number}"
            visual = None
        elif event.kind == "block_end":
            marker_text = f"block_end/{event.block}"
            visual = None
        elif event.kind == "session_end":
            marker_text = "session_end"
            visual = None
        elif event.kind == "session_stop":
            marker_text = "session_stop"
            visual = None
            self._request_stop("paradigm_completed")
            return

        # Visual change first, marker immediately afterwards.
        event_abs_mono = (self._start_monotonic or 0.0) + now_sec
        estimated = getattr(self.acquirer, "estimated_sample_count", None)
        sample = int(estimated(event_abs_mono)) if estimated is not None else int(self.acquirer.sample_count)
        self._set_visual(visual)
        lsl_sec = self._push_marker(marker_text, duration, sample_onset=sample)
        self._event_seq += 1
        append_jsonl(self.paths.events, {
            "seq": self._event_seq,
            "type": marker_text,
            "number": event.number,
            "block": event.block,
            "trial": event.trial,
            "duration_sec": duration,
            "expected_onset_sec": event.time_sec,
            "actual_onset_sec": now_sec,
            "monotonic_sec": now_sec,
            "lsl_sec": lsl_sec,
            "eeg_sample": sample,
            "recording_sample": max(0, sample - self._sample_count_at_start),
            "recording_onset_sec": max(0.0, (sample - self._sample_count_at_start) / self.cfg.sfreq),
            **getattr(self.acquirer, "last_status", {}),
        })

    def _request_stop(self, reason: str) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop_reason = reason

    def tick(self, now_monotonic: float) -> None:
        if self._start_monotonic is None or self._runner is None or self._stopped:
            return
        self._runner.tick(now_monotonic - self._start_monotonic)
        if self._runner.finished:
            self._request_stop("paradigm_completed")
        self._update_status("RUNNING" if not self._stopped else "STOPPING")

    def _update_status(self, state: str) -> None:
        if self.status_callback is None:
            return
        try:
            self.status_callback({
                "state": state,
                "stimuli_done": self._stim_count,
                "stimuli_total": len(self.paradigm.stim_on_events),
                "samples_acquired": int(self.acquirer.sample_count),
                "samples_recorded": self.recorder.sample_count,
            })
        except Exception:
            logger.exception("status callback failed")

    @property
    def finished(self) -> bool:
        return self._stopped and self._start_monotonic is not None

    def stop(self, reason: str = "user_stop") -> dict[str, Any]:
        if self._finalized:
            return self.session_metadata
        self._finalized = True
        self._request_stop(reason)
        if self._start_monotonic is None:
            return {}
        try:
            self._set_visual(None)
            self._push_marker("session_stop")
        except Exception:
            logger.exception("failed to write session_stop marker")
        try:
            self.acquirer.stop()
        except Exception:
            logger.exception("acquirer stop failed")
        try:
            recorder_sidecar = self.recorder.close()
        except Exception:
            logger.exception("recorder close failed")
            recorder_sidecar = {"error": "EDF close failed"}
        try:
            self.marker_outlet.close()
            self.eeg_outlet.close()
        except Exception:
            logger.debug("LSL close failed", exc_info=True)

        loss_stats = getattr(self.acquirer, "loss_stats", {})
        final = dict(self.session_metadata)
        final.update({
            "stopped_utc": utc_now_iso(),
            "stop_reason": self._stop_reason,
            "duration_s_expected": self.paradigm.duration_sec,
            "samples_expected": int(round(self.paradigm.duration_sec * self.cfg.sfreq)),
            "samples_acquired": int(self.acquirer.sample_count - self._sample_count_at_start),
            "samples_recorded": int(self.recorder.sample_count),
            "n_events_logged": self._event_seq,
            "n_stimuli": self._stim_count,
            "edf_sha256": recorder_sidecar.get("sha256"),
            "events_jsonl_sha256": sha256_file(self.paths.events) if self.paths.events.exists() else None,
            "loss_stats": loss_stats,
            "device_batch_size": getattr(self.acquirer, "batch_size", None),
            "sdk_channel_config": getattr(self.acquirer, "channel_config_summary", {}),
            "device_quality": getattr(self.acquirer, "quality_stats", {}),
            "last_packet_status": getattr(self.acquirer, "last_status", {}),
            "last_error": str(getattr(self.acquirer, "last_error", None) or ""),
        })
        atomic_write_json(self.paths.session, final)
        self._update_status("FINISHED")
        return final
