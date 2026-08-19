"""Generate a labeled synthetic guess-number dataset for end-to-end testing.

The synthetic P300 is intentionally easy but non-trivial: target stimuli evoke a
larger P3-like deflection at Cz/Pz.  This script writes the exact same files as
the real frontend (EDF+, events.jsonl, session.json), so the backend path is
identical.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from frontend.mock_eeg import build_stimulus_list, generate_session_data
from frontend.paradigm import Paradigm, ParadigmConfig
from frontend.recorder import RawEDFRecorder
from frontend.utils import append_jsonl, atomic_write_json, sha256_file, utc_now_iso


def make_session(output_dir: Path, participant: str, session: str, target: int,
                 sfreq: int, seed: int, blocks: int, repetitions: int,
                 stimulus_ms: int, blank_ms: int) -> dict:
    channels = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]
    paradigm = Paradigm(ParadigmConfig(
        blocks=blocks, repetitions=repetitions,
        fixation_s=0.5, stimulus_s=stimulus_ms / 1000.0, blank_s=blank_ms / 1000.0,
        inter_block_s=2.0, start_delay_s=1.0, end_delay_s=1.0, seed=seed))
    schedule = paradigm.schedule_records()
    stimuli = build_stimulus_list(schedule, sfreq)
    raw = generate_session_data(sfreq, channels, stimuli, target, seed=seed)

    stem = f"sub-{participant}_ses-{session}_task-guessnumber_run-001_eeg"
    edf_path = output_dir / (stem + ".edf")
    events_path = output_dir / (stem + "_events.jsonl")
    session_path = output_dir / (stem + "_session.json")

    recorder = RawEDFRecorder(edf_path, float(sfreq), channels,
                              participant_id=participant, session_id=session)
    recorder.write(raw)
    seq = 0
    for ev in schedule:
        onset = float(ev["time_sec"])
        sample = int(round(onset * sfreq))
        if ev["kind"] == "stim_on":
            seq += 1
            marker = f"stim_on/{ev['number']}"
        else:
            marker = ev["kind"]
            if ev["block"] is not None:
                marker = f"{ev['kind']}/{ev['block']}"
        recorder.add_annotation(onset, float(ev.get("duration_sec", 0.0)), marker)
        append_jsonl(events_path, {
            "seq": seq,
            "type": marker,
            "number": ev.get("number"),
            "block": ev.get("block"),
            "trial": ev.get("trial"),
            "duration_sec": ev.get("duration_sec", 0.0),
            "expected_onset_sec": onset,
            "actual_onset_sec": onset,
            "monotonic_sec": onset,
            "lsl_sec": float("nan"),
            "eeg_sample": sample,
            "recording_sample": sample,
            "recording_onset_sec": onset,
        })
    sidecar = recorder.close()

    session_meta = {
        "participant_id": participant,
        "session_id": session,
        "run_id": "001",
        "target_number": int(target),
        "sfreq": int(sfreq),
        "gain": "Gain24",
        "channels": channels,
        "ref_label": "A1",
        "gnd_label": "Fpz",
        "acquisition_mode": "mock",
        "paradigm": {
            "blocks": blocks, "repetitions": repetitions,
            "stimulus_s": stimulus_ms / 1000.0, "blank_s": blank_ms / 1000.0,
            "seed": seed,
        },
        "created_utc": utc_now_iso(),
        "synthetic": True,
        "edf_sha256": sidecar["sha256"],
        "events_jsonl_sha256": sha256_file(events_path),
    }
    atomic_write_json(session_path, session_meta)
    return {"edf": str(edf_path), "events": str(events_path), "session": str(session_path),
            "target": target, "n_stim": len([e for e in schedule if e["kind"] == "stim_on"])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data/raw")
    p.add_argument("--sessions", type=int, default=4)
    p.add_argument("--subject", default="P01")
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--sfreq", type=int, default=500)
    p.add_argument("--stimulus-ms", type=int, default=200)
    p.add_argument("--blank-ms", type=int, default=1300)
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for i in range(1, args.sessions + 1):
        target = rng.randint(1, 9)
        info = make_session(out, args.subject, f"{i:03d}", target, args.sfreq,
                            args.seed + i, args.blocks, args.repetitions,
                            args.stimulus_ms, args.blank_ms)
        print(info)


if __name__ == "__main__":
    main()
