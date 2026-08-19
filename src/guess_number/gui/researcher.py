"""Researcher-friendly graphical application for the guess-number P300 experiment.

This GUI is intentionally thin: it handles device checking, experiment setup,
the stimulus window and researcher interaction.  Heavy backend work
(ingest / QC / training / prediction) is delegated to a local Python
interpreter with ``python -m guess_number.backend.main ...`` so the frozen exe
does not need to bundle MNE / PyTorch / SciPy / scikit-learn / matplotlib.

The stimulus surface defaults to a **window** (not a full-screen black screen),
so the researcher can drag it onto a secondary monitor in Windows extended
desktop mode.  Full-screen mode is still available from the experiment tab.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QFont
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
from guess_number.frontend.channel_config import build_sdk_channel_config, read_montage
from guess_number.frontend.experiment import ExperimentConfig, ExperimentController
from guess_number.frontend.lsl_bridge import create_eeg_outlet, create_marker_outlet
from guess_number.frontend.mock_eeg import build_stimulus_list
from guess_number.frontend.paradigm import Paradigm, ParadigmConfig

logger = logging.getLogger("guess_number.gui")


def project_root() -> Path:
    """Project data root in both source and PyInstaller frozen modes."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        parent = exe_dir.parent
        if (parent / "src" / "guess_number").exists() or (parent / "pyproject.toml").exists():
            return parent
        return exe_dir
    return Path(__file__).resolve().parents[3]


def find_python_interpreter() -> str:
    """Find the Python environment used for heavy backend subprocesses.

    Precedence: GUESS_NUMBER_PYTHON -> current interpreter (source mode) ->
    project-local .venv311 -> PATH python/python3/py -> sys.executable.
    """
    env_python = os.environ.get("GUESS_NUMBER_PYTHON")
    if env_python and Path(env_python).exists():
        return env_python

    if not getattr(sys, "frozen", False):
        current = Path(sys.executable)
        if current.exists():
            return str(current)

    root = project_root()
    exe_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else None
    roots = [root] + ([exe_dir] if exe_dir is not None else [])
    win_candidates = [
        ".venv311/Scripts/python.exe", ".venv/Scripts/python.exe", "venv/Scripts/python.exe",
    ]
    posix_candidates = [".venv311/bin/python", ".venv/bin/python", "venv/bin/python"]
    for base in roots:
        for rel in win_candidates + posix_candidates:
            candidate = base / rel
            if candidate.exists():
                return str(candidate)

    for command in ("python", "python3", "py"):
        found = shutil.which(command)
        if found:
            return found
    return sys.executable


class BackendWorker(QThread):
    """Run one backend CLI command in a local Python process."""

    log = Signal(str)
    done = Signal(int)

    def __init__(self, command: str, argv: list[str], python_exe: str) -> None:
        super().__init__()
        self.command = command
        self.argv = list(argv)
        self.python_exe = python_exe

    def run(self) -> None:
        root = project_root()
        env = os.environ.copy()
        src_dir = root / "src"
        if src_dir.exists():
            old_path = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(src_dir) + (os.pathsep + old_path if old_path else "")

        cmd = [self.python_exe, "-m", "guess_number.backend.main", self.command, *self.argv]
        self.log.emit("$ " + subprocess.list2cmdline(cmd))
        kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "cwd": str(root),
            "env": env,
        }
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            proc = subprocess.Popen(cmd, **kwargs)
        except Exception as exc:
            self.log.emit(f"无法启动 Python 后端: {exc}")
            self.done.emit(1)
            return

        assert proc.stdout is not None
        for line in proc.stdout:
            self.log.emit(line.rstrip("\r\n"))
        code = proc.wait()
        self.done.emit(int(code or 0))


class DeviceCheckWorker(QThread):
    result = Signal(str)

    def __init__(self, mode: str, ble_name: str) -> None:
        super().__init__()
        self.mode = mode
        self.ble_name = ble_name

    def run(self) -> None:
        try:
            if self.mode == "ble":
                self.result.emit(self._check_ble())
            else:
                import brainsync_sdk as sdk
                ports = sdk.list_brainsync_ports()
                if not ports:
                    self.result.emit("未发现 BrainSync USB 设备")
                else:
                    self.result.emit(f"发现设备: {', '.join(map(str, ports))}")
        except Exception as exc:
            self.result.emit(f"检查失败: {exc}")

    def _check_ble(self) -> str:
        import asyncio

        import brainsync_sdk as sdk

        async def run() -> str:
            sdk.ble_init_adapter()
            await asyncio.sleep(1.0)
            handle = await sdk.open_brainsync_ble(self.ble_name)
            try:
                version = await sdk.get_firmware_version(handle)
                if isinstance(version, dict):
                    return (f"BLE 已连接 {self.ble_name} | {version.get('device_type', 'BrainSync')} "
                            f"SW:{version.get('sw_version', 'N/A')}")
                return f"BLE 已连接 {self.ble_name}"
            finally:
                try:
                    await sdk.close_device(handle)
                except Exception:
                    pass

        return asyncio.run(run())


class StimulusWindow(QWidget):
    """Black stimulus surface shown during the experiment.

    Default mode is a normal top-level window so it can be moved to a second
    monitor. ``display_mode="fullscreen"`` restores the old full-screen mode.
    """

    stop_requested = Signal()

    def __init__(self, display_mode: str = "window", screen_index: int = 0) -> None:
        super().__init__()
        self.display_mode = "fullscreen" if display_mode == "fullscreen" else "window"
        self.setWindowTitle("Guess Number P300 - Stimulus")
        self.setStyleSheet("background-color: black;")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: white; background-color: black;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

        self._place_on_screen(screen_index)
        if self.display_mode == "fullscreen":
            self.showFullScreen()
        else:
            self.show()

    def _screen(self, index: int) -> Any:
        screens = QApplication.screens() or [QApplication.primaryScreen()]
        return screens[max(0, min(int(index), len(screens) - 1))]

    def _place_on_screen(self, screen_index: int) -> None:
        screen = self._screen(screen_index)
        geometry = screen.availableGeometry()
        width = max(640, min(1280, int(geometry.width() * 0.78)))
        height = max(480, min(800, int(geometry.height() * 0.78)))
        x = geometry.x() + (geometry.width() - width) // 2
        y = geometry.y() + (geometry.height() - height) // 2
        self.resize(width, height)
        self.move(x, y)
        try:
            self.setScreen(screen)
        except Exception:
            logger.debug("QWidget.setScreen unavailable; falling back to geometry", exc_info=True)

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

    def closeEvent(self, event: Any) -> None:
        self.stop_requested.emit()
        event.accept()


class ResearcherWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Guess Number P300 - Researcher Console")
        self.resize(1100, 780)
        self.controller: ExperimentController | None = None
        self.stimulus: StimulusWindow | None = None
        self._preview_stimulus: StimulusWindow | None = None
        self.experiment_timer: QTimer | None = None
        self.project_root = project_root()
        self._build_ui()
        self._log(f"项目数据根目录: {self.project_root}")
        self._log("GUI ready. 请先检查设备，再填写实验参数。")

    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_experiment_tab(), "实验")
        self.tabs.addTab(self._build_backend_tab(), "数据后端")
        self.setCentralWidget(self.tabs)

    def _group(self, title: str) -> tuple[QGroupBox, QFormLayout]:
        box = QGroupBox(title)
        form = QFormLayout(box)
        return box, form
    def _build_experiment_tab(self) -> QWidget:
        widget = QWidget()
        self._experiment_tab = widget
        root = QVBoxLayout(widget)
        self._experiment_layout = root

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
        self.sp_sr = QComboBox(); self.sp_sr.addItems(["250"]); self.sp_sr.setCurrentText("250")
        self.cb_gain = QComboBox(); self.cb_gain.addItems(["Gain1", "Gain2", "Gain4", "Gain6", "Gain8", "Gain12", "Gain24"]); self.cb_gain.setCurrentText("Gain24")
        self.sp_stim_ms = QSpinBox(); self.sp_stim_ms.setRange(50, 1000); self.sp_stim_ms.setValue(200)
        self.sp_blank_ms = QSpinBox(); self.sp_blank_ms.setRange(100, 5000); self.sp_blank_ms.setValue(1300)
        self.sp_baseline_ms = QSpinBox(); self.sp_baseline_ms.setRange(0, 10000); self.sp_baseline_ms.setValue(2000)
        self.ed_output = QLineEdit(str(self.project_root / "data" / "recordings"))
        btn_out = QPushButton("选择...")
        btn_out.clicked.connect(self.choose_output_dir)
        out_container = QWidget()
        out_row = QHBoxLayout(out_container); out_row.setContentsMargins(0, 0, 0, 0)
        out_row.addWidget(self.ed_output); out_row.addWidget(btn_out)
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
        params.addRow("数据目录", out_container)
        self.lbl_duration = QLabel("")
        self.lbl_duration.setStyleSheet("color:#00ffcc;font-weight:bold;")
        params.addRow("预计实验时长", self.lbl_duration)
        for name in ["sp_blocks", "sp_reps", "sp_stim_ms", "sp_blank_ms", "sp_baseline_ms"]:
            field = getattr(self, name)
            field.valueChanged.connect(
                lambda _value, w=field: self._update_estimated_duration())
        root.addWidget(params_box)
        self._update_estimated_duration()

        display_box, display_form = self._group("刺激显示")
        self.cb_display_mode = QComboBox()
        self.cb_display_mode.addItems(["窗口（可拖到扩展屏第二屏）", "全屏"])
        self.cb_screen = QComboBox()
        self._refresh_screen_list()
        self.btn_preview = QPushButton("预览/定位刺激窗口")
        self.btn_preview.clicked.connect(self.preview_stimulus)
        display_form.addRow("显示模式", self.cb_display_mode)
        display_form.addRow("目标屏幕", self.cb_screen)
        display_form.addRow("", self.btn_preview)
        root.addWidget(display_box)

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
        self._backend_tab = widget
        root = QVBoxLayout(widget)
        box, form = self._group("后端处理")
        self.ed_data_dir = QLineEdit(str(self.project_root / "data" / "raw"))
        btn_data = QPushButton("选择...")
        btn_data.clicked.connect(lambda: self.choose_dir(self.ed_data_dir))
        data_row = QHBoxLayout(); data_row.addWidget(self.ed_data_dir); data_row.addWidget(btn_data)
        form.addRow("原始数据目录", data_row)
        self.ed_model_dir = QLineEdit(str(self.project_root / "data" / "derived" / "models" / "guess_number"))
        btn_model = QPushButton("选择...")
        btn_model.clicked.connect(lambda: self.choose_dir(self.ed_model_dir))
        model_row = QHBoxLayout(); model_row.addWidget(self.ed_model_dir); model_row.addWidget(btn_model)
        form.addRow("模型目录", model_row)
        self.ed_predict_edf = QLineEdit("")
        btn_edf = QPushButton("选择 EDF...")
        btn_edf.clicked.connect(self.choose_edf)
        edf_row = QHBoxLayout(); edf_row.addWidget(self.ed_predict_edf); edf_row.addWidget(btn_edf)
        form.addRow("预测 EDF", edf_row)
        self.ed_python = QLineEdit(find_python_interpreter())
        btn_python = QPushButton("选择...")
        btn_python.clicked.connect(self.choose_python)
        python_row = QHBoxLayout(); python_row.addWidget(self.ed_python); python_row.addWidget(btn_python)
        form.addRow("Python 解释器", python_row)
        form.addRow("", QLabel("后端（完整性/QC/训练/预测）会在该 Python 环境中运行，exe 不打包 torch/mne"))
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

    def _refresh_screen_list(self) -> None:
        current = self.cb_screen.currentIndex()
        screens = QApplication.screens() or [QApplication.primaryScreen()]
        self.cb_screen.clear()
        for i, screen in enumerate(screens, 1):
            geo = screen.geometry()
            self.cb_screen.addItem(
                f"屏幕 {i}: {screen.name()} ({geo.width()}x{geo.height()})")
        if screens:
            self.cb_screen.setCurrentIndex(max(0, min(current, len(screens) - 1)))

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

    def choose_python(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 Python 解释器", self.ed_python.text(),
            "Python (*.exe python python3);;All Files (*)")
        if path:
            self.ed_python.setText(path)

    def check_device(self) -> None:
        mode = "ble" if self.cb_mode.currentIndex() == 1 else "usb"
        ble_name = self.ed_ble_name.text().strip() or "BrainSync"
        self.lbl_device.setText("检查中...")
        self._device_check = DeviceCheckWorker(mode, ble_name)
        self._device_check.result.connect(self.lbl_device.setText)
        self._device_check.start()
    def _update_estimated_duration(self) -> None:
        blocks = int(self.sp_blocks.value())
        reps = int(self.sp_reps.value())
        pcfg = ParadigmConfig(
            blocks=blocks,
            repetitions=reps,
            fixation_s=0.0,
            stimulus_s=self.sp_stim_ms.value() / 1000.0,
            blank_s=self.sp_blank_ms.value() / 1000.0,
            baseline_black_s=self.sp_baseline_ms.value() / 1000.0,
            inter_block_s=2.0,
            start_delay_s=0.0,
            end_delay_s=1.0,
            seed=0,
        )
        paradigm = Paradigm(pcfg)
        total = blocks * 9 * reps
        minutes = int(paradigm.duration_sec // 60)
        seconds = int(round(paradigm.duration_sec % 60))
        self.lbl_duration.setText(
            f"{minutes:02d}:{seconds:02d} | 总数字 {total} 个 | "
            f"每个数字出现 {blocks * reps} 次")

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
            display_mode="fullscreen" if self.cb_display_mode.currentIndex() == 1 else "window",
            display_screen=int(self.cb_screen.currentIndex()),
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
    def preview_stimulus(self) -> None:
        if self._preview_stimulus is not None:
            self._preview_stimulus.close()
        mode = "fullscreen" if self.cb_display_mode.currentIndex() == 1 else "window"
        preview = StimulusWindow(display_mode=mode, screen_index=int(self.cb_screen.currentIndex()))
        preview.setWindowTitle("Guess Number P300 - 刺激窗口预览")
        preview.show_visual("7")
        self._preview_stimulus = preview

    def start_experiment(self) -> None:
        if self.controller is not None and not self.controller.finished:
            QMessageBox.information(self, "提示", "实验已经在运行中")
            return
        if self._preview_stimulus is not None:
            self._preview_stimulus.close()
            self._preview_stimulus = None
        try:
            cfg, pcfg = self._experiment_config()
            paradigm = Paradigm(pcfg)
            stimuli = build_stimulus_list(paradigm.schedule_records(), cfg.sfreq)
            # D-006: explicit project ChannelConfig must be sent to the SDK.
            sdk_channel_config = (None if cfg.acquisition_mode == "mock"
                                  else build_sdk_channel_config())
            acquirer = create_acquirer(
                cfg.acquisition_mode, cfg.sfreq, cfg.channels, stimuli, cfg.target_number,
                seed=0,
                sdk_channel_config=sdk_channel_config,
                ble_name=self.ed_ble_name.text().strip() or "BrainSync")
            marker_outlet = create_marker_outlet()
            eeg_outlet = create_eeg_outlet(cfg.sfreq, cfg.channels)
            controller = ExperimentController(cfg, paradigm, acquirer, marker_outlet, eeg_outlet)
            stimulus = StimulusWindow(display_mode=cfg.display_mode,
                                      screen_index=cfg.display_screen)
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
        report_dir = str(self.project_root / "data" / "derived" / "reports")
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
        python_exe = self.ed_python.text().strip() or find_python_interpreter()
        self.ed_python.setText(python_exe)
        if not Path(python_exe).exists() and not shutil.which(python_exe):
            QMessageBox.warning(self, "提示", f"找不到 Python 解释器: {python_exe}")
            return
        self.backend_log.appendPlainText(f"$ guess-number-backend {command} " + " ".join(argv))
        self.worker = BackendWorker(command, argv, python_exe)
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
