"""End-to-end smoke tests for guess-number P300 frontend/backend."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.config import ChannelConfig, PreprocessConfig, TrainConfig
from backend.io import load_session
from backend.model import build_shallow_convnet
from backend.preprocess import prepare_session
from backend.scoring import aggregate_number_scores
from frontend.mock_eeg import build_stimulus_list, generate_session_data
from frontend.paradigm import Paradigm, ParadigmConfig


def test_paradigm_is_balanced() -> None:
    paradigm = Paradigm(ParadigmConfig(blocks=2, repetitions=3, seed=1))
    stims = paradigm.stim_on_events
    assert len(stims) == 2 * 9 * 3
    for block in range(2):
        nums = [e.number for e in stims if e.block == block]
        assert sorted(nums) == sorted(list(range(1, 10)) * 3)


def test_synthetic_prepare_and_scoring(tmp_path: Path) -> None:
    channels = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]
    paradigm = Paradigm(ParadigmConfig(blocks=1, repetitions=1, seed=2))
    stimuli = build_stimulus_list(paradigm.schedule_records(), 500)
    raw = generate_session_data(500, channels, stimuli, target_number=5, seed=2)

    cfg = PreprocessConfig(raw_sfreq=500.0, downsample_sfreq=250.0)
    X, meta, sidecar = prepare_session_arrays(raw, channels, stimuli, 5, cfg)
    assert X.shape[0] == 9
    assert X.shape[1] == 8
    assert X.shape[2] == cfg.n_times
    assert meta["is_target"].sum() == 1

    probs = np.full(len(meta), 0.1)
    probs[meta["is_target"] == 1] = 0.9
    scores, ranking = aggregate_number_scores(meta, probs)
    assert ranking[0] == 5


def prepare_session_arrays(raw, channels, stimuli, target, cfg):
    """Small helper to bypass file IO in tests."""
    from backend.preprocess import _resample, _butter_bandpass, _notch_filter, \
        apply_reref, epoch_data, detect_artifacts
    filt = _butter_bandpass(raw, 500.0, cfg.highpass_hz, cfg.lowpass_hz)
    filt = _notch_filter(filt, 500.0, [cfg.notch_hz] + cfg.notch_harmonics)
    filt = _resample(filt, 500.0, cfg.downsample_sfreq)
    filt = apply_reref(filt, "car")
    events = pd.DataFrame([{
        "onset_sample": int(round(s.onset_sample * cfg.downsample_sfreq / 500.0)),
        "onset_sec": s.onset_sample / 500.0,
        "number": s.number,
        "block": s.block,
        "trial": s.trial,
    } for s in stimuli])
    X, meta, dropped = epoch_data(filt, cfg.downsample_sfreq, events, cfg.tmin_s, cfg.tmax_s,
                                  tuple(cfg.baseline_s))
    art = detect_artifacts(X, cfg)
    meta["bad_trial"] = art.bad_trial.astype(int)
    meta["is_target"] = (meta["number"] == target).astype(int)
    return X, meta, {"artifact": art.metrics}


def test_shallow_convnet_forward_shape() -> None:
    torch = pytest.importorskip("torch")
    model = build_shallow_convnet(8, 300, 2, TrainConfig(), model_sfreq=250.0)
    x = torch.randn(3, 1, 8, 300)
    y = model(x)
    assert tuple(y.shape) == (3, 2)


def test_edf_roundtrip(tmp_path: Path) -> None:
    pyedflib = pytest.importorskip("pyedflib")
    from frontend.recorder import RawEDFRecorder

    rec = RawEDFRecorder(tmp_path / "test.edf", 500.0,
                         ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"],
                         participant_id="P01", session_id="001")
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 5, size=(8, 1000)).astype(np.float32)
    rec.write(raw)
    rec.add_annotation(0.5, -1, "stim_on/7")
    rec.close()
    session = load_session(tmp_path / "test.edf")
    assert session.raw.shape[1] >= 1000
    assert session.ch_names[0] == "Fz"


def test_xdawn_projector_shapes() -> None:
    from backend.xdawn import XdawnProjector

    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, size=(30, 8, 300)).astype(np.float32)
    X[:, 3, 50:90] += 5.0
    y = np.array([1] * 3 + [0] * 27)
    p = XdawnProjector(reg=1e-6).fit(X, y, 2, 1)
    out = p.transform(X)
    assert p.n_output_channels == 11
    assert out.shape == (30, 11, 300)
    restored = XdawnProjector.from_dict(p.to_dict())
    assert restored.n_output_channels == 11
    assert restored.transform(X[:2]).shape == (2, 11, 300)


def test_leadoff_status_marks_bad_trial(tmp_path: Path) -> None:
    from backend.io import load_session
    from backend.preprocess import prepare_session
    from frontend.recorder import RawEDFRecorder

    channels = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]
    rng = np.random.default_rng(4)
    raw = rng.normal(0, 5, size=(8, 1500)).astype(np.float32)
    rec = RawEDFRecorder(tmp_path / "leadoff.edf", 500.0, channels,
                         participant_id="P01", session_id="001")
    rec.write(raw)
    rec.add_annotation(0.5, -1, "stim_on/7")
    rec.close()
    events_path = tmp_path / "leadoff.edf_events.jsonl"
    events_path.write_text(
        '{"type":"stim_on/7","number":7,"block":0,"trial":0,'
        '"recording_sample":250,"leadoff_status":0,"is_impedance_mode":false}\n'
        '{"type":"stim_on/3","number":3,"block":0,"trial":1,'
        '"recording_sample":750,"leadoff_status":255,"is_impedance_mode":false}\n',
        encoding="utf-8")
    session = load_session(tmp_path / "leadoff.edf", events_path=events_path)
    cfg = PreprocessConfig(raw_sfreq=500.0, downsample_sfreq=250.0, xdawn_enable=False)
    X, meta, sidecar = prepare_session(session, cfg, channels)
    bad_by_status = meta.loc[meta["number"] == 7, "bad_trial"].iloc[0]
    assert int(bad_by_status) == 1
    # With only two synthetic trials the global 30% bad-ratio gate is exceeded,
    # which proves the gate is enforced rather than silently skipped.
    assert sidecar["qc_pass"] is False
    assert "exceeds gate" in str(sidecar["qc"]["issues"])
