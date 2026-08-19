"""Researcher-friendly graphical application for the guess-number P300 experiment.

This GUI wraps every operational step that would otherwise be run from the
command line:

- device/port check
- experiment parameter configuration
- stimulus playback and synchronous EEG acquisition
- automatic Data/<timestamp> saving
- subject-guess recording
- backend ingest / QC report / training / prediction

It is also the entry point used by PyInstaller to build the Windows exe.
"""
from __future__ import annotations

import contextlib
import io
import logging
import sys
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from guess_number.frontend.acquisition import create_acquirer
from guess_number.frontend.channel_config import read_montage
from guess_number.frontend.experiment import ExperimentConfig, ExperimentController
from guess_number.frontend.lsl_bridge import create_eeg_outlet, create_marker_outlet
from guess_number.frontend.mock_eeg import build_stimulus_list
from guess_number.frontend.paradigm import Paradigm, ParadigmConfig

logger = logging.getLogger("guess_number.gui")


class BackendWorker(QThread):
    log = Signal(str)
    done = Signal(int)

    def __init__(self, command: str, argv: list[str]) -> None:
        super().__init__()
        self.command = command
        self.argv = argv

    def run(self) -> None:
        buf = io.StringIO()
        code = 0
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                if self.command == "ingest":
                    from guess_number.backend.ingest import main
                    main(self.argv)
                elif self.command == "report":
                    from guess_number.backend.qc_report import main
                    main(self.argv)
                elif self.command == "train":
                    from guess_number.backend.train import main
                    main(self.argv)
                elif self.command == "predict":
                    from guess_number.backend.predict import main
                    main(self.argv)
                else:
                    raise ValueError(f"unknown backend command: {self.command}")
        except SystemExit as exc:
            code = int(exc.code or 0)
        except Exception:
            logger.exception("backend command failed")
            code = 1
        text = buf.getvalue()
        if text:
            self.log.emit(text)
        self.done.emit(code)


class StimulusWindow(QWidget):
    """Full-screen black stimulus surface used during the experiment."""

    stop_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Guess Number P300 - Stimulus")
        self.setStyleSheet("background-color: black;")
        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: white; background-color: black;")
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self.showFullScreen()

    def show_visual(self, text: str | None) -> None:
        if text is None:
            self.label.setText("")
            self.label.repaint()
            return
        self.label.setText(text)
        font = QFont()
        if text == "+":
            font.setPixelSize(max(48, int(self.height() * 0.10)))
        else:
            font.setPixelSize(max(80, int(self.height() * 0.22)))
            font.setBold(True)
        self.label.setFont(font)
        self.label.repaint()

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.stop_requested.emit()
        else:
            super().keyPressEvent(event)


class ResearcherWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Guess Number P300 - Researcher Console")
        self.resize(1100, 760)
        self.controller: ExperimentController | None = None
        self.stimulus: StimulusWindow | None = None
        self.experiment_timer: QTimer | None = None
        self._build_ui()
        self._log("GUI ready. 请先检查设备，再填写实验参数。")

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_experiment_tab(), "实验")
        tabs.addTab(self._build_backend_tab(), "数据后端")
        self.setCentralWidget(tabs)

    def _group(self, title: str) -> tuple[QGroupBox, QFormLayout]:
        box = QGroupBox(title)
        form = QFormLayout(box)
        return box, form

    def _build_experiment_tab(self) -> QWidget:
        widget = QWidget()
        root = QVBoxLayout(widget)

        device_box, device_form = self._group("设备")
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["真实设备(USB)", "蓝牙设备", "Mock 模拟"])
        self.ed_ble_name = QLineEdit("BrainSync")
        self.cb_mode.currentIndexChanged.connect(
            lambda _: self.ed_ble_name.setEnabled(self.cb_mode.currentIndex() == 1))
        self.btn_check = QPushButton("检查设备")
        self.btn_check.clicked.connect(self.check_device)
        device_form.addRow("采集模式", self.cb_mode)
        device_form.addRow("BLE 设备名", self.ed_ble_name)
        device_form.addRow("", self.btn_check)
        self.lbl_device = QLabel("尚未检查")
        device_form.addRow("状态", self.lbl_device)
        root.addWidget(device_box)

        params_box, params = self._group("实验参数")
        self.ed_subject = QLineEdit("P01")
        self.ed_session = QLineEdit("001")
        self.ed_run = QLineEdit("001")
        self.sp_target = QSpinBox(); self.sp_target.setRange(1, 9); self.sp_target.setValue(7)
        self.sp_blocks = QSpinBox(); self.sp_blocks.setRange(1, 20); self.sp_blocks.setValue(6)
        self.sp_reps = QSpinBox(); self.sp_reps.setRange(1, 20); self.sp_reps.setValue(5)
        self.sp_sr = QComboBox(); self.sp_sr.addItems(["250", "500", "1000", "2000", "4000", "8000"]); self.sp_sr.setCurrentText("500")
        self.cb_gain = QComboBox(); self.cb_gain.addItems(["Gain1", "Gain2", "Gain4", "Gain6", "Gain8", "Gain12", "Gain24"]); self.cb_gain.setCurrentText("Gain24")
        self.sp_stim_ms = QSpinBox(); self.sp_stim_ms.setRange(50, 1000); self.sp_stim_ms.setValue(200)
        self.sp_blank_ms = QSpinBox(); self.sp_blank_ms.setRange(100, 5000); self.sp_blank_ms.setValue(1300)
        self.sp_baseline_ms = QSpinBox(); self.sp_baseline_ms.setRange(0, 10000); self.sp_baseline_ms.setValue(2000)
        self.ed_output = QLineEdit(str(Path.cwd() / "data" / "recordings"))
        btn_out = QPushButton("选择...")
        btn_out.clicked.connect(self.choose_output_dir)
        out_row = QHBoxLayout(); out_row.addWidget(self.ed_output); out_row.addWidget(btn_out)
        params.addRow("受试者 ID", self.ed_subject)
        params.addRow("Session", self.ed_session)
        params.addRow("Run", self.ed_run)
        params.addRow("默想数字", self.sp_target)
        params.addRow("Blocks", self.sp_blocks)
        params.addRow("每数字重复", self.sp_reps)
        params.addRow("采样率 (Hz)", self.sp_sr)
        params.addRow("增益", self.cb_gain)
        params.addRow("刺激时长 (ms)", self.sp_stim_ms)
        params.addRow("空白时长 (ms)", self.sp_blank_ms)
        params.addRow("黑屏基线 (ms)", self.sp_baseline_ms)
        params.addRow("数据目录", out_row)
        root.addWidget(params_box)

        ctrl = QHBoxLayout()
        self.btn_start = QPushButton("开始实验")
        self.btn_start.setStyleSheet("background-color:#1e6f5c;color:white;font-weight:bold;")
        self.btn_start.clicked.connect(self.start_experiment)
        self.btn_stop = QPushButton("停止实验")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_experiment)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        root.addLayout(ctrl)

        root.addWidget(QLabel("日志"))
        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumBlockCount(5000)
        root.addWidget(self.log_widget, 1)
        return widget

    def _build_backend_tab(self) -> QWidget:
        widget = QWidget()
        root = QVBoxLayout(widget)
        box, form = self._group("后端处理")
        self.ed_data_dir = QLineEdit(str(Path.cwd() / "data" / "raw"))
        btn_data = QPushButton("选择...")
        btn_data.clicked.connect(lambda: self.choose_dir(self.ed_data_dir))
        data_row = QHBoxLayout(); data_row.addWidget(self.ed_data_dir); data_row.addWidget(btn_data)
        form.addRow("原始数据目录", data_row)
        self.ed_model_dir = QLineEdit(str(Path.cwd() / "data" / "derived" / "models" / "guess_number"))
        btn_model = QPushButton("选择...")
        btn_model.clicked.connect(lambda: self.choose_dir(self.ed_model_dir))
        model_row = QHBoxLayout(); model_row.addWidget(self.ed_model_dir); model_row.addWidget(btn_model)
        form.addRow("模型目录", model_row)
        self.ed_predict_edf = QLineEdit("")
        btn_edf = QPushButton("选择 EDF...")
        btn_edf.clicked.connect(self.choose_edf)
        edf_row = QHBoxLayout(); edf_row.addWidget(self.ed_predict_edf); edf_row.addWidget(btn_edf)
        form.addRow("预测 EDF", edf_row)
        root.addWidget(box)

        buttons = QHBoxLayout()
        btn_ingest = QPushButton("1. 完整性检查")
        btn_report = QPushButton("2. QC 报告")
        btn_train = QPushButton("3. 训练模型")
        btn_predict = QPushButton("4. 预测数字")
        for btn, cmd in [(btn_ingest, "ingest"), (btn_report, "report"),
                         (btn_train, "train"), (btn_predict, "predict")]:
            btn.clicked.connect(lambda checked=False, c=cmd: self.run_backend(c))
            buttons.addWidget(btn)
        root.addLayout(buttons)
        root.addWidget(QLabel("后端日志"))
        self.backend_log = QPlainTextEdit()
        self.backend_log.setReadOnly(True)
        self.backend_log.setMaximumBlockCount(5000)
        root.addWidget(self.backend_log, 1)
        return widget

    def _log(self, text: str) -> None:
        self.log_widget.appendPlainText(text)

    def choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据目录", self.ed_output.text())
        if path:
            self.ed_output.setText(path)

    def choose_dir(self, line_edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录", line_edit.text())
        if path:
            line_edit.setText(path)

    def choose_edf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 EDF 文件", "", "EDF Files (*.edf *.bdf)")
        if path:
            self.ed_predict_edf.setText(path)

    def check_device(self) -> None:
        def work() -> None:
            try:
                import brainsync_sdk as sdk
                if self.cb_mode.currentIndex() == 1:
                    sdk.ble_init_adapter()
                    self.lbl_device.setText(
                        f"蓝牙适配器已初始化；开始实验时将连接 {self.ed_ble_name.text().strip() or 'BrainSync'}")
                    return
                ports = sdk.list_brainsync_ports()
                if not ports:
                    self.lbl_device.setText("未发现 BrainSync USB 设备")
                    return
                self.lbl_device.setText(f"发现设备: {', '.join(map(str, ports))}")
            except Exception as exc:
                self.lbl_device.setText(f"检查失败: {exc}")
        # Keep the UI responsive.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, work)

    def _experiment_config(self) -> tuple[ExperimentConfig, ParadigmConfig]:
        montage = read_montage()
        cfg = ExperimentConfig(
            participant_id=self.ed_subject.text().strip() or "P01",
            session_id=self.ed_session.text().strip() or "001",
            run_id=self.ed_run.text().strip() or "001",
            target_number=int(self.sp_target.value()),
            sfreq=int(self.sp_sr.currentText()),
            gain=self.cb_gain.currentText(),
            channels=montage["channels"],
            ref_label=montage["ref_label"],
            gnd_label=montage["gnd_label"],
            acquisition_mode=("device" if self.cb_mode.currentIndex() == 0
                              else "ble" if self.cb_mode.currentIndex() == 1 else "mock"),
            output_dir=self.ed_output.text().strip() or str(Path.cwd() / "data" / "recordings"),
        )
        pcfg = ParadigmConfig(
            blocks=int(self.sp_blocks.value()),
            repetitions=int(self.sp_reps.value()),
            fixation_s=0.0,
            stimulus_s=self.sp_stim_ms.value() / 1000.0,
            blank_s=self.sp_blank_ms.value() / 1000.0,
            baseline_black_s=self.sp_baseline_ms.value() / 1000.0,
            inter_block_s=2.0,
            start_delay_s=0.0,
            end_delay_s=1.0,
            seed=0,
        )
        return cfg, pcfg

    def start_experiment(self) -> None:
        if self.controller is not None and not self.controller.finished:
            QMessageBox.information(self, "提示", "实验已经在运行中")
            return
        try:
            cfg, pcfg = self._experiment_config()
            paradigm = Paradigm(pcfg)
            stimuli = build_stimulus_list(paradigm.schedule_records(), cfg.sfreq)
            if cfg.acquisition_mode == "ble" and cfg.sfreq > 1000:
                raise ValueError("蓝牙模式最高只支持 1000 Hz，请把采样率改为 250/500/1000")
            acquirer = create_acquirer(cfg.acquisition_mode, cfg.sfreq, cfg.channels,
                                       stimuli, cfg.target_number, seed=0,
                                       ble_name=self.ed_ble_name.text().strip() or "BrainSync")
            marker_outlet = create_marker_outlet()
            eeg_outlet = create_eeg_outlet(cfg.sfreq, cfg.channels)
            controller = ExperimentController(cfg, paradigm, acquirer, marker_outlet, eeg_outlet)
            stimulus = StimulusWindow()
            controller.visual_callback = stimulus.show_visual
            stimulus.stop_requested.connect(lambda: controller.stop("user_escape"))
            self.controller = controller
            self.stimulus = stimulus
            controller.start()
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.experiment_timer = QTimer(self)
            self.experiment_timer.timeout.connect(self._poll_experiment)
            self.experiment_timer.start(20)
            self._log(f"实验已开始，数据目录: {controller.run_dir}")
        except Exception as exc:
            logger.exception("start experiment failed")
            QMessageBox.critical(self, "启动失败", str(exc))

    def _poll_experiment(self) -> None:
        if self.controller is None:
            return
        self.controller.tick(time.monotonic())
        if self.controller.finished:
            self.finish_experiment()

    def stop_experiment(self) -> None:
        if self.controller is not None:
            self.controller.stop("user_stop")
            self.finish_experiment()

    def finish_experiment(self) -> None:
        if self.experiment_timer is not None:
            self.experiment_timer.stop()
        if self.controller is not None and self.controller.finished:
            self.controller.stop("experiment_completed")
        if self.stimulus is not None:
            self.stimulus.close()
            self.stimulus = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.controller is not None:
            guess, ok = QInputDialog.getInt(
                self, "记录受试者猜测", "受试者猜的数字是？(1-9)",
                value=int(self.controller.cfg.target_number), min=1, max=9)
            if ok:
                self.controller.record_subject_guess(int(guess))
            self._log(f"实验完成，文件位于: {self.controller.run_dir}")
            self.controller = None

    def run_backend(self, command: str) -> None:
        data_dir = self.ed_data_dir.text().strip()
        model_dir = self.ed_model_dir.text().strip()
        report_dir = str(Path(model_dir).parents[1] / "reports") if model_dir else "data/derived/reports"
        manifest = str(Path(data_dir).parent / "manifest.jsonl") if data_dir else "data/manifest.jsonl"
        if command == "ingest":
            argv = ["--data-dir", data_dir, "--manifest", manifest]
        elif command == "report":
            argv = ["--data-dir", data_dir, "--output-dir", report_dir]
        elif command == "train":
            argv = ["--data-dir", data_dir, "--output-dir", model_dir, "--cv", "--production"]
        elif command == "predict":
            edf = self.ed_predict_edf.text().strip()
            if not edf:
                QMessageBox.warning(self, "提示", "请先选择预测 EDF")
                return
            argv = ["--edf", edf, "--model-dir", model_dir]
        else:
            return
        self.backend_log.appendPlainText(f"$ guess-number-backend {command} " + " ".join(argv))
        self.worker = BackendWorker(command, argv)
        self.worker.log.connect(self.backend_log.appendPlainText)
        self.worker.done.connect(lambda code: self.backend_log.appendPlainText(f"[exit={code}]"))
        self.worker.start()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    app = QApplication(sys.argv)
    window = ResearcherWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
