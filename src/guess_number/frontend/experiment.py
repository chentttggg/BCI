"""Experiment controller: acquisition, stimulus timing, markers, raw recording."""
from __future__ import annotations

import logging
import subprocess
import threading
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
    unit: str = "uV"
    channels: list[str] = field(default_factory=lambda: ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"])
    ref_label: str = "A1"
    gnd_label: str = "A2"
    acquisition_mode: str = "mock"
    output_dir: str | Path = "Data"
    seed: int = 0
    subject_guess: int | None = None


@dataclass
class SessionPaths:
    edf: Path
    events: Path
    session: Path


def make_session_paths(cfg: ExperimentConfig, run_dir: Path) -> SessionPaths:
    stem = f"sub-{cfg.participant_id}_ses-{cfg.session_id}_task-guessnumber_run-{cfg.run_id}_eeg"
    return SessionPaths(edf=run_dir / (stem + ".edf"),
                        events=run_dir / (stem + "_events.jsonl"),
                        session=run_dir / (stem + "_session.json"))


class ExperimentController:
    """Owns one recording session and drives the paradigm timeline."""

    def __init__(self, cfg: ExperimentConfig, paradigm: Paradigm,
                 acquirer: Any, marker_outlet: Any, eeg_outlet: Any) -> None:
        self.cfg = cfg
        self.paradigm = paradigm
        self.acquirer = acquirer
        self.marker_outlet = marker_outlet
        self.eeg_outlet = eeg_outlet
        self.started_at_local = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = self._make_run_dir()
        self.paths = make_session_paths(cfg, self.run_dir)
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
        self._acquisition_started = False
        self._event_seq = 0
        self._sample_count_at_start = 0
        self._stim_count = 0
        self._current_visual: str | None = None
        self._latest_samples: list[np.ndarray] = []
        self._latest_lock = threading.Lock()
        self._eeg_clock_refs: list[tuple[int, float]] = []
        self._clock_lock = threading.Lock()
        self.session_metadata: dict[str, Any] = {}
        self._initialise_session_metadata()

    def _make_run_dir(self) -> Path:
        root = Path(self.cfg.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        candidate = root / self.started_at_local
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        for i in range(1, 100):
            candidate = root / f"{self.started_at_local}_{i:02d}"
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
        raise RuntimeError("cannot allocate a run directory")

    def _initialise_session_metadata(self) -> None:
        self.session_metadata = {
            "participant_id": self.cfg.participant_id,
            "session_id": self.cfg.session_id,
            "run_id": self.cfg.run_id,
            "target_number": int(self.cfg.target_number),
            "sfreq": int(self.cfg.sfreq),
            "gain": self.cfg.gain,
            "unit": self.cfg.unit,
            "channels": list(self.cfg.channels),
            "ref_label": self.cfg.ref_label,
            "gnd_label": self.cfg.gnd_label,
            "acquisition_mode": self.cfg.acquisition_mode,
            "paradigm": {
                "blocks": self.paradigm.cfg.blocks,
                "repetitions": self.paradigm.cfg.repetitions,
                "stimulus_s": self.paradigm.cfg.stimulus_s,
                "blank_s": self.paradigm.cfg.blank_s,
                "baseline_black_s": self.paradigm.cfg.baseline_black_s,
                "fixation_s": self.paradigm.cfg.fixation_s,
                "inter_block_s": self.paradigm.cfg.inter_block_s,
                "seed": self.paradigm.cfg.seed,
            },
            "created_utc": utc_now_iso(),
            "git_commit": git_commit_short(),
            "run_dir": str(self.run_dir),
            "started_at_local": self.started_at_local,
            "subject_guess": self.cfg.subject_guess,
        }
        atomic_write_json(self.paths.session, self.session_metadata)

    def start(self) -> None:
        if self._start_monotonic is not None:
            return
        self._sample_count_at_start = int(self.acquirer.sample_count)
        self._runner = TimelineRunner(self.paradigm, self._on_timeline_event)
        # Acquisition starts immediately when the experiment starts. The first
        # two seconds are a black-screen resting baseline.
        self.acquirer.start(self._on_eeg_chunk)
        self._acquisition_started = True
        self._start_monotonic = time.monotonic()
        self._push_marker("session_start")
        self._push_marker(f"target/{self.cfg.target_number}")
        self._set_visual(None)
        self._update_status("RUNNING")

    def _start_acquisition(self, event: TimelineEvent) -> None:
        if self._acquisition_started:
            return
        self._sample_count_at_start = int(self.acquirer.sample_count)
        # Mock stream must jump to the current timeline position because it has
        # pre-generated the whole session. Real device starts at its live cursor.
        start_sample = int(round(event.time_sec * self.cfg.sfreq))
        try:
            self.acquirer.start(self._on_eeg_chunk, start_sample=start_sample)
        except TypeError:
            self.acquirer.start(self._on_eeg_chunk)
        if self.cfg.acquisition_mode == "mock":
            # Mock raw data was pre-generated for the whole timeline; the local
            # recording starts at `start_sample`, so that is the new zero point.
            self._sample_count_at_start = start_sample
        self._acquisition_started = True
        logger.info("Acquisition started at first digit (timeline %.3fs, sample %d)",
                    event.time_sec, start_sample)

    def _on_eeg_chunk(self, samples: np.ndarray, start_sample: int) -> None:
        try:
            self.recorder.write(samples)
        except Exception:
            logger.exception("raw recorder write failed")
        lsl_ts = None
        try:
            lsl_ts = self.eeg_outlet.push_chunk(samples)
        except Exception:
            logger.debug("LSL EEG push failed", exc_info=True)
        if lsl_ts is not None:
            rel_sample = max(0, int(start_sample - self._sample_count_at_start))
            with self._clock_lock:
                self._eeg_clock_refs.append((rel_sample, float(lsl_ts)))
                if len(self._eeg_clock_refs) > 12:
                    self._eeg_clock_refs = self._eeg_clock_refs[-12:]
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

    def record_subject_guess(self, value: int | None) -> None:
        if value is not None and not 1 <= int(value) <= 9:
            raise ValueError("subject_guess must be in 1..9")
        self.cfg.subject_guess = value
        self.session_metadata["subject_guess"] = value
        atomic_write_json(self.paths.session, self.session_metadata)
        self._write_experiment_summary()

    def _write_split_files(self) -> list[dict[str, Any]]:
        """Save one EDF file per block, ordered by time, next to the total EDF."""
        data = self.recorder.get_data()
        events = []
        if self.paths.events.exists():
            import json as _json
            with open(self.paths.events, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(_json.loads(line))
        starts = {}
        ends = {}
        for ev in events:
            if ev.get("type", "").startswith("block_start"):
                starts[int(ev.get("block", -1))] = int(ev.get("recording_sample", 0))
            elif ev.get("type", "").startswith("block_end"):
                ends[int(ev.get("block", -1))] = int(ev.get("recording_sample", 0))
        manifest = []
        n_total = int(data.shape[1])
        for block in sorted(starts):
            start = max(0, int(starts[block]))
            if block in ends:
                end = min(n_total, max(start + 1, int(ends[block])))
            else:
                end = n_total
            if end <= start:
                continue
            block_data = data[:, start:end].astype(np.float64)
            path = self.run_dir / f"eeg_block_{int(block):03d}.edf"
            self._write_edf_segment(path, block_data)
            manifest.append({
                "block": int(block),
                "start_sample": int(start),
                "end_sample": int(end),
                "n_samples": int(end - start),
                "duration_sec": round((end - start) / self.cfg.sfreq, 3),
                "path": str(path),
            })
        atomic_write_json(self.run_dir / "split_manifest.json",
                          {"n_total_samples": n_total, "segments": manifest})
        return manifest

    def _write_edf_segment(self, path: Path, data: np.ndarray) -> None:
        from pyedflib import highlevel
        headers = [highlevel.make_signal_header(
            label=label, dimension="uV", sample_frequency=float(self.cfg.sfreq),
            physical_min=-2000.0, physical_max=2000.0) for label in self.cfg.channels]
        highlevel.write_edf(str(path), data, headers,
                            file_type=1)  # EDF+

    def _write_experiment_summary(self) -> None:
        counts: dict[int, int] = {}
        for event in self.paradigm.stim_on_events:
            counts[event.number] = counts.get(event.number, 0) + 1
        summary = {
            "started_at_local": self.started_at_local,
            "run_dir": str(self.run_dir),
            "participant_id": self.cfg.participant_id,
            "session_id": self.cfg.session_id,
            "run_id": self.cfg.run_id,
            "target_number": int(self.cfg.target_number),
            "subject_guess": self.cfg.subject_guess,
            "n_stimuli": self._stim_count,
            "stimulus_count_by_digit": counts,
            "n_samples_recorded": int(self.recorder.sample_count),
            "sfreq": int(self.cfg.sfreq),
            "unit": self.cfg.unit,
            "total_edf": str(self.paths.edf),
            "events_jsonl": str(self.paths.events),
            "session_json": str(self.paths.session),
        }
        atomic_write_json(self.run_dir / "experiment_summary.json", summary)

    def _set_visual(self, visual: str | None) -> None:
        if visual == self._current_visual:
            return
        self._current_visual = visual
        if self.visual_callback is not None:
            try:
                self.visual_callback(visual)
            except Exception:
                logger.exception("visual callback failed")

    def _sample_from_lsl(self, lsl_sec: float, fallback: int) -> tuple[int, str]:
        if not np.isfinite(lsl_sec):
            return int(fallback), "monotonic_estimate"
        with self._clock_lock:
            refs = list(self._eeg_clock_refs)
        if len(refs) >= 2:
            xs = np.asarray([r[1] for r in refs], dtype=np.float64)
            ys = np.asarray([r[0] for r in refs], dtype=np.float64)
            slope, intercept = np.polyfit(xs, ys, 1)
            sample = int(round(intercept + slope * float(lsl_sec)))
            return max(0, sample), "lsl_linear_fit"
        if refs:
            rel, ref_lsl = refs[-1]
            sample = int(round(rel + (float(lsl_sec) - ref_lsl) * self.cfg.sfreq))
            return max(0, sample), "lsl_last_anchor"
        return int(fallback), "monotonic_estimate"

    def _push_marker(self, text: str, duration_sec: float = 0.0,
                     sample_onset: int | None = None) -> tuple[float, int, str]:
        try:
            lsl_sec = self.marker_outlet.push(text)
        except Exception:
            lsl_sec = float("nan")
        if sample_onset is None:
            sample_onset = int(self.acquirer.sample_count)
        sample, source = self._sample_from_lsl(lsl_sec, sample_onset)
        self.recorder.add_annotation(
            max(0.0, sample / self.cfg.sfreq), duration_sec, text)
        return float(lsl_sec), int(sample), source

    def _on_timeline_event(self, event: TimelineEvent, now_sec: float) -> None:
        marker_text = ""
        duration = 0.0
        visual: str | None = None
        if event.kind == "session_start":
            marker_text = "session_start"
        elif event.kind == "baseline_start":
            marker_text = "baseline_start"
            visual = None
        elif event.kind == "baseline_end":
            marker_text = "baseline_end"
            visual = None
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
        fallback_sample = int(estimated(event_abs_mono)) if estimated is not None else int(self.acquirer.sample_count)
        self._set_visual(visual)
        lsl_sec, sample, alignment_source = self._push_marker(
            marker_text, duration, sample_onset=fallback_sample)
        recording_sample = max(0, sample - self._sample_count_at_start)
        recording_onset_sec = max(0.0, recording_sample / self.cfg.sfreq)
        self._event_seq += 1
        append_jsonl(self.paths.events, {
            "seq": self._event_seq,
            "type": marker_text,
            "number": event.number,
            "digit": event.number,
            "block": event.block,
            "trial": event.trial,
            "duration_sec": duration,
            "expected_onset_sec": event.time_sec,
            "actual_onset_sec": now_sec,
            "monotonic_sec": now_sec,
            "lsl_sec": lsl_sec,
            "alignment_source": alignment_source,
            "eeg_sample": sample,
            "recording_sample": recording_sample,
            "recording_onset_sec": recording_onset_sec,
            "edf_annotation_onset_sec": recording_onset_sec,
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
        split_manifest = []
        if self.recorder._data is not None:
            try:
                split_manifest = self._write_split_files()
            except Exception:
                logger.exception("block split save failed")
        self._write_experiment_summary()
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
            "event_alignment": {
                "method": "lsl-marker-to-eeg-sample linear fit",
                "sfreq": int(self.cfg.sfreq),
                "clock_anchors": [{"recording_sample": int(smp), "lsl_sec": float(ts)}
                                  for smp, ts in self._eeg_clock_refs[-5:]],
            },
            "split_files": split_manifest,
            "subject_guess": self.cfg.subject_guess,
            "last_error": str(getattr(self.acquirer, "last_error", None) or ""),
        })
        self.session_metadata = final
        atomic_write_json(self.paths.session, final)
        self._update_status("FINISHED")
        return final
