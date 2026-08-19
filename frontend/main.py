"""Guess-number P300 experiment frontend.

Examples
--------
    # Fast headless mock session (for pipeline development)
    python -m frontend.main --mock --headless --target 7 \
        --blocks 1 --repetitions 1 --stimulus-ms 20 --blank-ms 30 \
        --output-dir data/raw

    # Real device, full GUI
    python -m frontend.main --device --target 7 --blocks 6 --repetitions 5
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .acquisition import create_acquirer
from .channel_config import build_sdk_channel_config
from .experiment import ExperimentConfig, ExperimentController
from .lsl_bridge import create_eeg_outlet, create_marker_outlet
from .mock_eeg import build_stimulus_list
from .paradigm import Paradigm, ParadigmConfig


def _read_channels(path: str | Path) -> tuple[list[str], str, str]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    channels = [str(item["label"]) for item in obj["eeg_channels"]]
    ref = str(obj.get("ref_label", "A1"))
    gnd = str(obj.get("gnd_label", "A2"))
    return channels, ref, gnd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Guess-number P300 experiment frontend")
    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument("--mock", action="store_true", help="synthetic EEG mode (no hardware)")
    mode.add_argument("--device", action="store_true", help="BrainSync BS8A real-device mode")
    p.add_argument("--subject", default="P01")
    p.add_argument("--session", default="001")
    p.add_argument("--run", default="001")
    p.add_argument("--target", type=int, required=True, choices=range(1, 10))
    p.add_argument("--sfreq", type=int, default=500, choices=[250, 500, 1000, 2000, 4000, 8000])
    p.add_argument("--gain", default="Gain24")
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--fixation-ms", type=int, default=0)
    p.add_argument("--baseline-black-ms", type=int, default=2000,
                   help="第一个数字前的黑屏静息基线时长")
    p.add_argument("--stimulus-ms", type=int, default=200)
    p.add_argument("--blank-ms", type=int, default=1300)
    p.add_argument("--inter-block-ms", type=int, default=2000)
    p.add_argument("--start-delay-ms", type=int, default=0)
    p.add_argument("--end-delay-ms", type=int, default=1000)
    p.add_argument("--output-dir", default="Data")
    p.add_argument("--subject-guess", type=int, choices=range(1, 10), default=None,
                   help="受试者在数字播放结束后自己报出的数字 (1-9)")
    p.add_argument("--channel-config", default="config/channel_config.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action="store_true",
                   help="run without Qt GUI (useful for mock recording and tests)")
    p.add_argument("--disable-lsl", action="store_true")
    p.add_argument("--model-dir", default=None,
                   help="trained backend model dir; after recording, run an offline prediction")
    return p


def _post_session_predict(paths, args: argparse.Namespace) -> None:
    if not args.model_dir:
        return
    try:
        from backend.predict import process_session
        from backend.config import ChannelConfig
        from backend.model import ModelBundle

        bundle = ModelBundle.load(args.model_dir)
        ch = ChannelConfig.from_json(args.channel_config)
        result = process_session(paths.edf, bundle, ch, paths.events, paths.session)
        print("ONLINE-OFFLINE PREDICTION:")
        print("  prediction:", result["prediction"], "| truth:", result["target_number_truth"],
              "| correct:", result["correct"])
        print("  top3:", result["top3"])
    except Exception:
        logging.exception("post-session prediction failed")


def _run_headless(args: argparse.Namespace) -> int:
    channels, ref, gnd = _read_channels(args.channel_config)
    cfg = ExperimentConfig(
        participant_id=args.subject, session_id=args.session, run_id=args.run,
        target_number=args.target, sfreq=args.sfreq, gain=args.gain,
        channels=channels, ref_label=ref, gnd_label=gnd,
        acquisition_mode="mock" if args.mock else "device",
        output_dir=args.output_dir, seed=args.seed,
        subject_guess=args.subject_guess,
    )
    paradigm_cfg = ParadigmConfig(
        blocks=args.blocks, repetitions=args.repetitions,
        fixation_s=args.fixation_ms / 1000.0,
        stimulus_s=args.stimulus_ms / 1000.0,
        baseline_black_s=args.baseline_black_ms / 1000.0,
        blank_s=args.blank_ms / 1000.0,
        inter_block_s=args.inter_block_ms / 1000.0,
        start_delay_s=args.start_delay_ms / 1000.0,
        end_delay_s=args.end_delay_ms / 1000.0,
        seed=args.seed,
    )
    paradigm = Paradigm(paradigm_cfg)
    stimuli = build_stimulus_list(paradigm.schedule_records(), args.sfreq)
    sdk_channel_config = build_sdk_channel_config(args.channel_config)
    acquirer = create_acquirer(cfg.acquisition_mode, args.sfreq, channels, stimuli,
                               args.target, seed=args.seed,
                               sdk_channel_config=sdk_channel_config)
    marker_outlet = None if args.disable_lsl else create_marker_outlet()
    eeg_outlet = None if args.disable_lsl else create_eeg_outlet(args.sfreq, channels)
    if marker_outlet is None:
        from .lsl_bridge import NullMarkerOutlet
        marker_outlet = NullMarkerOutlet()
    if eeg_outlet is None:
        from .lsl_bridge import NullEEGOutlet
        eeg_outlet = NullEEGOutlet()

    controller = ExperimentController(cfg, paradigm, acquirer, marker_outlet, eeg_outlet)
    controller.start()
    logging.info("Headless session started: target=%s, duration=%.1fs",
                 args.target, paradigm.duration_sec)
    try:
        while not controller.finished:
            controller.tick(time.monotonic())
            time.sleep(0.005)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        result = controller.stop("keyboard_interrupt" if controller._stop_reason == "" else controller._stop_reason)
    if getattr(args, "subject_guess", None) is not None:
        controller.record_subject_guess(args.subject_guess)
        print({"subject_guess_saved": args.subject_guess})
    print(result)
    _post_session_predict(controller.paths, args)
    return 0 if result.get("n_stimuli", 0) > 0 else 2


def _run_gui(args: argparse.Namespace) -> int:
    try:
        from PySide6.QtCore import QPoint, Qt, QTimer
        from PySide6.QtGui import QColor, QFont, QPainter, QPen
        from PySide6.QtWidgets import QApplication, QInputDialog, QLabel, QVBoxLayout, QWidget
    except Exception as exc:
        print("PySide6 is required for GUI mode. Use --headless for data recording only.", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        return 3

    channels, ref, gnd = _read_channels(args.channel_config)
    cfg = ExperimentConfig(
        participant_id=args.subject, session_id=args.session, run_id=args.run,
        target_number=args.target, sfreq=args.sfreq, gain=args.gain,
        channels=channels, ref_label=ref, gnd_label=gnd,
        acquisition_mode="mock" if args.mock else "device",
        output_dir=args.output_dir, seed=args.seed,
        subject_guess=args.subject_guess,
    )
    paradigm_cfg = ParadigmConfig(
        blocks=args.blocks, repetitions=args.repetitions,
        fixation_s=args.fixation_ms / 1000.0,
        stimulus_s=args.stimulus_ms / 1000.0,
        baseline_black_s=args.baseline_black_ms / 1000.0,
        blank_s=args.blank_ms / 1000.0,
        inter_block_s=args.inter_block_ms / 1000.0,
        start_delay_s=args.start_delay_ms / 1000.0,
        end_delay_s=args.end_delay_ms / 1000.0,
        seed=args.seed,
    )
    paradigm = Paradigm(paradigm_cfg)
    stimuli = build_stimulus_list(paradigm.schedule_records(), args.sfreq)
    sdk_channel_config = build_sdk_channel_config(args.channel_config)
    acquirer = create_acquirer(cfg.acquisition_mode, args.sfreq, channels, stimuli,
                               args.target, seed=args.seed,
                               sdk_channel_config=sdk_channel_config)
    marker_outlet = None if args.disable_lsl else create_marker_outlet()
    eeg_outlet = None if args.disable_lsl else create_eeg_outlet(args.sfreq, channels)
    if marker_outlet is None:
        from .lsl_bridge import NullMarkerOutlet
        marker_outlet = NullMarkerOutlet()
    if eeg_outlet is None:
        from .lsl_bridge import NullEEGOutlet
        eeg_outlet = NullEEGOutlet()

    app = QApplication(sys.argv)

    class WaveformWidget(QWidget):
        def __init__(self, channels: list[str]) -> None:
            super().__init__()
            self.channels = channels
            self.samples = np.zeros((len(channels), 1), dtype=np.float32)
            self.setMinimumHeight(180)

        def set_samples(self, samples: np.ndarray) -> None:
            self.samples = np.asarray(samples, dtype=np.float32)

        def paintEvent(self, event: Any) -> None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(5, 8, 12))
            n_ch = self.samples.shape[0]
            if n_ch == 0:
                return
            h = max(1, self.height() // n_ch)
            pen = QPen(QColor(90, 220, 170))
            pen.setWidth(1)
            painter.setPen(pen)
            for i in range(n_ch):
                y0 = int((i + 0.5) * h)
                signal = self.samples[i, -min(len(self.samples[i]), 500):]
                if len(signal) < 2:
                    continue
                mid = np.nanmedian(signal)
                amp = max(1.0, float(np.nanmax(np.abs(signal - mid))))
                n = len(signal)
                pts = []
                for j, val in enumerate(signal):
                    x = int(10 + j * (self.width() - 20) / max(1, n - 1))
                    y = int(y0 - (val - mid) / (2.2 * amp) * (h - 14))
                    pts.append((x, y))
                painter.drawPolyline([QPoint(int(x), int(y)) for x, y in pts])
                painter.setPen(QColor(140, 160, 175))
                painter.drawText(6, y0 - 3, self.channels[i])
                painter.setPen(pen)

    class MainWindow(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Guess Number P300")
            self.controller = ExperimentController(cfg, paradigm, acquirer,
                                                   marker_outlet, eeg_outlet)
            self.controller.visual_callback = self.set_stimulus
            self.controller.status_callback = self.set_status
            self._status = ""
            layout = QVBoxLayout(self)
            self.status_label = QLabel("Preparing...")
            self.status_label.setStyleSheet("color: #8fc9ff; font-size: 18px;")
            layout.addWidget(self.status_label)
            self.stimulus_label = QLabel("")
            self.stimulus_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.stimulus_label.setStyleSheet("color: white; background-color: black;")
            layout.addWidget(self.stimulus_label, 8)
            self.wave = WaveformWidget(channels)
            layout.addWidget(self.wave, 2)
            self.setStyleSheet("background-color: black;")
            self.controller.start()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.on_tick)
            self.timer.start(5)
            self.showFullScreen()

        def set_stimulus(self, text: str | None) -> None:
            if text is None:
                self.stimulus_label.setText("")
            elif text == "+":
                self.stimulus_label.setText("+")
                font = QFont()
                font.setPixelSize(max(48, int(self.height() * 0.10)))
                self.stimulus_label.setFont(font)
            else:
                self.stimulus_label.setText(text)
                font = QFont()
                font.setPixelSize(max(80, int(self.height() * 0.22)))
                font.setBold(True)
                self.stimulus_label.setFont(font)
            self.stimulus_label.repaint()

        def set_status(self, status: dict[str, Any]) -> None:
            self._status = (
                f"{status.get('state')} | trial {status.get('stimuli_done')}/{status.get('stimuli_total')} | "
                f"recorded {status.get('samples_recorded', 0)} samples"
            )

        def on_tick(self) -> None:
            now = time.monotonic()
            self.controller.tick(now)
            self.status_label.setText(self._status)
            self.wave.set_samples(self.controller.get_recent_samples(max_samples=500))
            self.wave.update()
            if self.controller.finished:
                result = self.controller.stop("gui_completed")
                self._ask_subject_guess()
                print(result)
                _post_session_predict(self.controller.paths, args)
                QApplication.quit()

        def _ask_subject_guess(self) -> None:
            if getattr(args, "subject_guess", None) is not None:
                self.controller.record_subject_guess(args.subject_guess)
                return
            guess, ok = QInputDialog.getInt(
                self, "记录受试者猜测", "受试者猜的数字是？(1-9)",
                value=int(self.controller.cfg.target_number), min=1, max=9)
            if ok:
                self.controller.record_subject_guess(int(guess))
                print({"subject_guess_saved": int(guess)})

        def keyPressEvent(self, event: Any) -> None:
            if event.key() == Qt.Key.Key_Escape:
                result = self.controller.stop("user_escape")
                self._ask_subject_guess()
                print(result)
                _post_session_predict(self.controller.paths, args)
                QApplication.quit()

    window = MainWindow()
    return app.exec()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.mock and not args.device:
        args.mock = True
        logging.info("No mode selected; using --mock")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    code = _run_headless(args) if args.headless else _run_gui(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
