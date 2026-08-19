# -*- coding: utf-8 -*-
"""
BrainSync Multimodal Research Station - Main Window.
Split and refactored from brainsync_multimodal_hub.py.
"""

import os
import sys
import math
import time
import asyncio
from datetime import datetime
from pathlib import Path
import numpy as np

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, QSize
import pyqtgraph as pg

# Local imports
from multimodal_hub.i18n import I18N
from multimodal_hub.gui.widgets import StatCard
from multimodal_hub.core.dsp_filter import SoftwareFilter, RingBuffer
from multimodal_hub.core.edf_recorder import EDFRecorder
from multimodal_hub.core.channel_config import (
    active_mask_from_booleans,
    default_channel_config,
    load_gui_config,
    save_gui_config,
    trigger_hub_dict_to_runtime,
    trigger_hub_runtime_to_dict,
    with_active_mask,
)
from multimodal_hub.core.mock_source import (
    MockEegPacket, MockCesPacket, MockImuPacket,
    MockMagPacket, MockBatteryPacket, MockMeasurementPacket,
    generate_mock_eeg_samples, generate_mock_ces_samples
)
from multimodal_hub.gui.dialogs.triggerbox_dialog import TriggerBoxDialog, _TRB_AVAILABLE
from multimodal_hub.gui.dialogs.montage_dialog import MontageDialog

import brainsync_sdk as sdk

class SuppressNativeStderr:
    """Context manager to temporarily silence C/C++ native library stderr outputs (fd 2)."""
    def __enter__(self):
        try:
            self.devnull = os.open(os.devnull, os.O_WRONLY)
            self.old_stderr_fd = os.dup(2)
            os.dup2(self.devnull, 2)
        except Exception:
            self.devnull = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if getattr(self, "devnull", None) is not None:
            try:
                os.dup2(self.old_stderr_fd, 2)
                os.close(self.old_stderr_fd)
                os.close(self.devnull)
            except Exception:
                pass

# Check pylsl availability
try:
    with SuppressNativeStderr():
        from pylsl import StreamInfo, StreamOutlet
    _LSL_AVAILABLE = True
except Exception:
    StreamInfo, StreamOutlet = None, None
    _LSL_AVAILABLE = False

class CompleteSensorGUI(QtWidgets.QMainWindow):
    sig_device_disconnected = QtCore.Signal(str)

    def __init__(self, mock_mode=False, default_port=None):
        super().__init__()
        self.mock_mode = mock_mode
        self.default_port = default_port
        self.handle = None
        self.connected_method = "serial"
        self.running_streams = {}

        # Multimodal LSL Outlets
        self.lsl_eeg_outlet = None
        self.lsl_marker_outlet = None
        self.lsl_enabled = True

        # Event marking hotkeys & tags
        self.event_tags = ["Rest", "Task/Trigger", "Eye Blink", "Arm Movement", "Custom Marker"]
        self.marker_history_list = []

        # Set default language
        self.lang = "en"
        self.sdk_version = getattr(sdk, "__version__", "0.2.0")

        self.setWindowTitle(f"BrainSync SDK - Advanced Control & Visualization Hub (v{self.sdk_version})")
        self.resize(1340, 850)

        # Apply QSS
        self.apply_qss()

        # Shared data buffers
        self.capacity_eeg = 2000
        self.capacity_ces = 2000
        self.capacity_imu = 500
        self.capacity_mag = 500

        self.eeg_fs = 250
        self.channel_config_path = Path(__file__).resolve().parents[1] / "channel_config.json"
        self.trigger_hub_config = {}
        self.channel_config = self.load_gui_channel_config()
        self.eeg_ch_names = list(self.channel_config.labels)
        self.eeg_buffers = [RingBuffer(self.capacity_eeg) for _ in range(8)]
        self.eeg_filter = SoftwareFilter(fs=self.eeg_fs)
        self.eeg_offset_gap = 300

        self.ces_buffers = [RingBuffer(self.capacity_ces) for _ in range(2)]

        self.imu_acc_buffers = [RingBuffer(self.capacity_imu) for _ in range(3)]
        self.imu_gyro_buffers = [RingBuffer(self.capacity_imu) for _ in range(3)]
        self.mag_buffers = [RingBuffer(self.capacity_mag) for _ in range(3)]

        # Packets count and loss statistics (seq tracking)
        self.total_eeg_pkts = 0
        self.lost_eeg_pkts = 0
        self.last_eeg_seq = None
        self.eeg_conversion_errors = 0

        self.total_ces_pkts = 0
        self.lost_ces_pkts = 0
        self.last_ces_seq = None

        # EEG visible channel checkboxes reference list
        self.chk_channels = []

        # TriggerBox Integration
        self.trb_device = None
        self.trb_polling_task = None
        self.trb_aliases, self.trb_thresholds, self.trb_enabled_status = trigger_hub_dict_to_runtime(
            self.trigger_hub_config
        )
        self.recording_start_time = None
        self.edf_recorder = None

        # i18n state persistence for runtime updates
        self.ct_state_key = "ct_status_idle"
        self.ct_state_args = {}
        self.ct_state_raw_text = ""
        self.stim_state_key = "stim_state_disarmed"

        self.init_ui()

        # Initialize LSL
        self.init_lsl_outlets()

        # Periodic update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(40) # 25 FPS update

        # Unexpected disconnection handling
        self.sig_device_disconnected.connect(self.on_device_disconnected_signal)
        try:
            sdk.set_connection_state_callback(self.on_connection_state_changed)
        except Exception as e:
            self.write_log(f"Warning: set_connection_state_callback not supported: {e}")

        # Mock simulation task variable
        self.mock_task = None

        # Allow key event interception on window
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def apply_qss(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #08080c;
            }
            QTabWidget::pane {
                border: 1px solid #14172a;
                background-color: #0b0c13;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #121424;
                color: #8f9bb3;
                padding: 10px 20px;
                min-width: 140px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 3px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #0b0c13;
                color: #00ffcc;
                border-bottom: 2px solid #00ffcc;
            }
            QLabel {
                color: #e2e8f0;
            }
            QPushButton {
                background-color: #1a1b2e;
                border: 1px solid #363757;
                border-radius: 5px;
                color: #e2e8f0;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #272846;
                border-color: #00ffcc;
            }
            QPushButton:pressed {
                background-color: #0f101f;
            }
            QPushButton:disabled {
                background-color: #0e0f17;
                color: #475569;
                border-color: #1c1d29;
            }
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #121321;
                border: 1px solid #2a2d48;
                border-radius: 5px;
                color: #ffffff;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: #121321;
                color: #ffffff;
                selection-background-color: #272846;
                selection-color: #00ffcc;
                border: 1px solid #2a2d48;
            }
            QTableWidget {
                background-color: #0b0c13;
                border: 1px solid #23253b;
                gridline-color: #1a1a26;
                color: #ffffff;
            }
            QScrollBar:vertical {
                background: #0b0c13;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #23253b;
                min-height: 20px;
                border-radius: 5px;
            }
            QProgressBar {
                background-color: #121321;
                border: 1px solid #2a2d48;
                border-radius: 5px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #00ffcc;
                width: 10px;
            }
            QGroupBox {
                border: 1px solid #1c1d2e;
                border-radius: 6px;
                margin-top: 15px;
                font-weight: bold;
                color: #00ffcc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QListWidget {
                background-color: #05060b;
                border: 1px solid #1c1d2e;
                color: #ffaa00;
                font-family: monospace;
                font-size: 11px;
                border-radius: 5px;
            }
            QCheckBox {
                color: #e2e8f0;
            }
        """)

    def tr(self, key, **kwargs):
        text = I18N[self.lang].get(key, key)
        if kwargs:
            return text.format(**kwargs)
        return text

    def init_lsl_outlets(self):
        if not _LSL_AVAILABLE or not self.lsl_enabled:
            return

        try:
            with SuppressNativeStderr():
                # Create LSL EEG Outlet
                info = StreamInfo("BrainSync-EEG", "EEG", 8, self.eeg_fs, "float32", "brainsync_eeg_source")
                self.apply_channel_config_to_lsl_info(info)
                self.lsl_eeg_outlet = StreamOutlet(info)

                # Create LSL Marker Outlet
                marker_info = StreamInfo("BrainSync-Markers", "Markers", 1, 0.0, "string", "brainsync_markers_source")
                self.lsl_marker_outlet = StreamOutlet(marker_info)

            self.write_log("LSL Outlets successfully registered (EEG & Markers).")
        except Exception as e:
            self.write_log(f"LSL initialization error: {e}")

    def load_gui_channel_config(self):
        try:
            gui_config = load_gui_config(self.channel_config_path)
            self.trigger_hub_config = gui_config["trigger_hub"]
            return gui_config["channel_config"]
        except Exception as e:
            print(f"Failed to load channel_config.json, using defaults: {e}", flush=True)
            self.trigger_hub_config = trigger_hub_runtime_to_dict({}, {}, {})
            return default_channel_config()

    def current_active_mask(self):
        if not self.chk_channels:
            return self.channel_config.active_mask
        return active_mask_from_booleans(chk.isChecked() for chk in self.chk_channels)

    def current_channel_config(self):
        self.channel_config = with_active_mask(self.channel_config, self.current_active_mask())
        return self.channel_config

    def persist_channel_config(self, quiet=False):
        try:
            self.trigger_hub_config = trigger_hub_runtime_to_dict(
                self.trb_aliases,
                self.trb_thresholds,
                self.trb_enabled_status,
            )
            save_gui_config(self.current_channel_config(), self.trigger_hub_config, self.channel_config_path)
            if not quiet:
                self.write_log(f"Channel config saved: {self.channel_config_path}")
        except Exception as e:
            self.write_log(f"Failed to save channel config: {e}")

    def reload_channel_config(self):
        try:
            gui_config = load_gui_config(self.channel_config_path)
            self.channel_config = gui_config["channel_config"]
            self.trigger_hub_config = gui_config["trigger_hub"]
            self.trb_aliases, self.trb_thresholds, self.trb_enabled_status = trigger_hub_dict_to_runtime(
                self.trigger_hub_config
            )
            self.eeg_ch_names = list(self.channel_config.labels)
            for index, chk in enumerate(self.chk_channels):
                chk.blockSignals(True)
                chk.setText(self.eeg_ch_names[index])
                chk.setChecked(self.channel_config.active_mask & (1 << index) != 0)
                chk.blockSignals(False)
            self.update_channel_visibility()
            if self.lsl_enabled:
                self.init_lsl_outlets()
            self.write_log(f"Channel config loaded: {self.channel_config_path}")
        except Exception as e:
            self.write_log(f"Failed to load channel config: {e}")

    def edit_montage_config(self):
        dialog = MontageDialog(self, self.current_channel_config(), self.trigger_hub_config)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        self.channel_config = dialog.channel_config
        self.trigger_hub_config = dialog.trigger_hub
        self.trb_aliases, self.trb_thresholds, self.trb_enabled_status = trigger_hub_dict_to_runtime(
            self.trigger_hub_config
        )
        self.eeg_ch_names = list(self.channel_config.labels)
        for index, chk in enumerate(self.chk_channels):
            chk.blockSignals(True)
            chk.setText(self.eeg_ch_names[index])
            chk.setChecked(self.channel_config.active_mask & (1 << index) != 0)
            chk.blockSignals(False)
        self.update_channel_visibility()
        self.persist_channel_config()
        if self.lsl_enabled:
            self.init_lsl_outlets()
        self.write_log("Montage configuration updated.")

    def apply_channel_config_to_lsl_info(self, info):
        try:
            config = self.current_channel_config()
            desc = info.desc()
            if config.ref_label:
                desc.append_child_value("reference", config.ref_label)
            if config.gnd_label:
                desc.append_child_value("ground", config.gnd_label)

            stim_channels = desc.append_child("tes_channels")
            for assignment in config.stim:
                node = stim_channels.append_child("channel")
                node.append_child_value("label", assignment.label)
                node.append_child_value("slot", str(assignment.slot))
                node.append_child_value("index", str(assignment.channel))
                node.append_child_value("polarity", assignment.polarity)

            channels = desc.append_child("channels")
            labels = list(config.labels)
            for index, label in enumerate(labels[:8]):
                node = channels.append_child("channel")
                node.append_child_value("label", label)
                node.append_child_value("unit", "microvolts")
                node.append_child_value("type", "EEG")
                node.append_child_value("index", str(index))
                node.append_child_value("active", "true" if config.active_mask & (1 << index) else "false")
        except Exception as e:
            self.write_log(f"Failed to attach channel metadata to LSL: {e}")

    def init_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # 1. Header Area
        header_layout = QtWidgets.QHBoxLayout()
        self.lbl_main_title = QtWidgets.QLabel(self.tr("title"))
        self.lbl_main_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff; font-family: 'Outfit', 'Inter', sans-serif;")
        # Hidden title widget to prevent duplicate text in GUI, keeping only the Window Title.
        # header_layout.addWidget(self.lbl_main_title)

        self.lbl_mock_indicator = QtWidgets.QLabel("")
        if self.mock_mode:
            self.lbl_mock_indicator.setText(self.tr("mock_indicator"))
            self.lbl_mock_indicator.setStyleSheet("color: #ffaa00; font-weight: bold; background-color: #3b2800; border: 1px solid #ffaa00; border-radius: 4px; padding: 4px 8px;")
        header_layout.addWidget(self.lbl_mock_indicator)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # 2. Main Tab widget
        self.tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tabs)

        self.setup_tab_control()
        self.setup_tab_eeg()
        self.setup_tab_motion()
        self.setup_tab_stimulation()

        # 3. Footer statusbar info
        footer_layout = QtWidgets.QHBoxLayout()
        self.lbl_connection_status = QtWidgets.QLabel(self.tr("status_disconnected"))
        self.lbl_connection_status.setStyleSheet("color: #ff5555; font-weight: bold;")
        footer_layout.addWidget(self.lbl_connection_status)
        footer_layout.addStretch()

        # Language Switcher (QPushButton style matching revo-sdk)
        btn_text = "🌐 EN" if self.lang == "en" else "🌐 中"
        self.lang_btn = QtWidgets.QPushButton(btn_text)
        self.lang_btn.setFixedWidth(60)
        self.lang_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a1b2e;
                border: 1px solid #363757;
                border-radius: 4px;
                color: #e2e8f0;
                padding: 2px 6px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #272846;
                border-color: #00ffcc;
            }
        """)
        self.lang_btn.clicked.connect(self.toggle_language)
        footer_layout.addWidget(self.lang_btn)

        footer_layout.addSpacing(15)

        self.lbl_battery = QtWidgets.QLabel("Battery: -- %")
        self.lbl_battery.setStyleSheet("color: #888899;")
        footer_layout.addWidget(self.lbl_battery)

        main_layout.addLayout(footer_layout)
        self.set_device_controls_enabled(False)
        self.set_connection_controls_enabled(True)
        self.refresh_i18n_ui()

    def set_connection_controls_enabled(self, enabled):
        self.cb_conn_method.setEnabled(enabled)
        if enabled:
            self.on_connection_method_changed()
        else:
            self.lbl_port_label.setEnabled(False)
            self.cb_ports.setEnabled(False)
            self.btn_refresh.setEnabled(False)
            self.lbl_ble_target.setEnabled(False)
            self.cb_ble_names.setEnabled(False)

    def on_connection_method_changed(self):
        is_serial = self.cb_conn_method.currentIndex() == 0
        
        # Serial port controls visibility & enabled state
        self.lbl_port_label.setVisible(is_serial)
        self.cb_ports.setVisible(is_serial)
        self.btn_refresh.setVisible(is_serial)
        self.lbl_port_label.setEnabled(is_serial)
        self.cb_ports.setEnabled(is_serial)
        self.btn_refresh.setEnabled(is_serial)

        # BLE target controls visibility & enabled state
        self.lbl_ble_target.setVisible(not is_serial)
        self.cb_ble_names.setVisible(not is_serial)
        self.lbl_ble_target.setEnabled(not is_serial)
        self.cb_ble_names.setEnabled(not is_serial)

    def set_device_controls_enabled(self, enabled):
        self.box_edf.setEnabled(enabled)
        self.box_markers.setEnabled(enabled)
        self.box_stream_ctrl.setEnabled(enabled)
        self.box_eeg_cfg.setEnabled(enabled)
        self.box_filter.setEnabled(enabled)
        self.box_perf.setEnabled(enabled)
        self.box_motion_ctrl.setEnabled(enabled)
        self.box_ct.setEnabled(enabled)
        self.box_stim.setEnabled(enabled)
        self.btn_refresh_status.setEnabled(enabled)

    def toggle_language(self):
        if self.lang == "en":
            self.lang = "zh"
            self.lang_btn.setText("🌐 中")
        else:
            self.lang = "en"
            self.lang_btn.setText("🌐 EN")
        self.refresh_i18n_ui()

    def update_ct_state(self, key, raw_text=None, **kwargs):
        self.ct_state_key = key
        self.ct_state_args = kwargs
        if raw_text is not None:
            self.ct_state_raw_text = raw_text

        if key == "raw_status":
            self.lbl_ct_state.setText(self.ct_state_raw_text)
        else:
            self.lbl_ct_state.setText(self.tr(key, **kwargs))

    def update_stim_state(self, key):
        self.stim_state_key = key
        self.lbl_stim_state.setText(self.tr(key))

    def refresh_i18n_ui(self):
        # Update Window titles & indicator
        title_text = f"{self.tr('title')} (v{self.sdk_version})"
        self.lbl_main_title.setText(title_text)
        self.setWindowTitle(title_text)
        if self.mock_mode:
            self.lbl_mock_indicator.setText(self.tr("mock_indicator"))

        # Connection status label
        if self.handle is None:
            self.lbl_connection_status.setText(self.tr("status_disconnected"))
            self.lbl_connection_status.setStyleSheet("color: #ff5555; font-weight: bold;")
        elif self.mock_mode:
            self.lbl_connection_status.setText(self.tr("status_connected_mock"))
            self.lbl_connection_status.setStyleSheet("color: #00ffcc; font-weight: bold;")
        else:
            if self.connected_method == "ble":
                self.lbl_connection_status.setText(self.tr("status_connected_ble", target=self.cb_ble_names.currentText()))
            else:
                self.lbl_connection_status.setText(self.tr("status_connected", port=self.cb_ports.currentText()))
            self.lbl_connection_status.setStyleSheet("color: #00ffcc; font-weight: bold;")

        # Tab 1 translation
        self.tabs.setTabText(0, self.tr("tab_conn"))
        self.box_conn.setTitle(self.tr("conn_title"))
        self.lbl_conn_method.setText(self.tr("conn_method_lbl"))

        curr_method_idx = self.cb_conn_method.currentIndex()
        self.cb_conn_method.blockSignals(True)
        self.cb_conn_method.clear()
        self.cb_conn_method.addItems([
            self.tr("conn_method_serial"),
            self.tr("conn_method_ble")
        ])
        if curr_method_idx >= 0:
            self.cb_conn_method.setCurrentIndex(curr_method_idx)
        self.cb_conn_method.blockSignals(False)
        self.on_connection_method_changed()

        self.lbl_port_label.setText(self.tr("port_lbl"))
        self.btn_refresh.setText(self.tr("btn_refresh"))

        self.lbl_ble_target.setText(self.tr("ble_target_lbl"))
        self.cb_ble_names.setPlaceholderText(self.tr("ble_target_placeholder"))

        if self.handle is None:
            self.btn_connect.setText(self.tr("btn_connect"))
        self.btn_disconnect.setText(self.tr("btn_disconnect"))

        self.box_stats.setTitle(self.tr("dev_status_title"))
        self.card_fw.lbl_title.setText(self.tr("fw_card"))
        self.card_bat.lbl_title.setText(self.tr("bat_card"))
        self.card_charge.lbl_title.setText(self.tr("charge_card"))
        self.btn_refresh_status.setText(self.tr("btn_refresh_status"))

        self.box_edf.setTitle(self.tr("edf_title"))
        self.lbl_filename_label.setText(self.tr("filename_lbl"))
        self.btn_edf_start.setText(self.tr("btn_edf_start"))
        self.btn_edf_stop.setText(self.tr("btn_edf_stop"))

        self.box_ann.setTitle(self.tr("ann_grp"))
        self.txt_annotation.setPlaceholderText(self.tr("ann_placeholder"))
        self.btn_send_ann.setText(self.tr("btn_ann_send"))

        self.box_markers.setTitle(self.tr("marker_title"))
        for i in range(5):
            self.marker_buttons[i].setText(self.tr("btn_marker", tag=f"F{i+1} - {self.event_tags[i]}"))
        self.lbl_history_label.setText(self.tr("history_lbl"))

        self.box_lsl.setTitle(self.tr("lsl_bridge_title"))
        self.chk_lsl.setText(self.tr("lsl_chk_lbl"))
        self.update_lsl_indicator()

        self.box_log.setTitle(self.tr("log_title"))

        # Tab 2 translation
        self.tabs.setTabText(1, self.tr("tab_eeg"))
        self.eeg_group.setTitle(self.tr("eeg_plot_title"))
        self.ces_group.setTitle(self.tr("ces_plot_title"))
        self.box_stream_ctrl.setTitle(self.tr("stream_switches_title"))
        self.btn_eeg_start.setText(self.tr("btn_eeg_stop") if "eeg" in self.running_streams else self.tr("btn_eeg_start"))
        self.btn_ces_start.setText(self.tr("btn_ces_stop") if "ces" in self.running_streams else self.tr("btn_ces_start"))
        self.box_eeg_cfg.setTitle(self.tr("eeg_config_title"))
        self.btn_apply_eeg.setText(self.tr("apply_settings"))
        self.btn_edit_montage.setText(self.tr("btn_edit_montage"))
        self.btn_load_channel_config.setText(self.tr("btn_load_montage"))
        self.btn_save_channel_config.setText(self.tr("btn_save_montage"))
        self.lbl_eeg_offset_lbl.setText(self.tr("plot_offset_lbl"))
        self.box_filter.setTitle(self.tr("sw_filters_title"))
        self.box_perf.setTitle(self.tr("sig_quality_title"))

        # Tab 3 translation
        self.tabs.setTabText(2, self.tr("tab_motion"))
        self.box_acc.setTitle(self.tr("imu_acc_title"))
        self.box_gyro.setTitle(self.tr("imu_gyro_title"))
        self.box_mag.setTitle(self.tr("mag_title"))
        self.box_motion_ctrl.setTitle(self.tr("motion_switches_title"))
        self.btn_imu_start.setText(self.tr("btn_imu_stop") if "imu" in self.running_streams else self.tr("btn_imu_start"))
        self.btn_mag_start.setText(self.tr("btn_mag_stop") if "mag" in self.running_streams else self.tr("btn_mag_start"))

        # Tab 4 translation
        self.tabs.setTabText(3, self.tr("tab_stim"))
        self.box_ct.setTitle(self.tr("ct_title"))
        self.lbl_ct_freq.setText(self.tr("ct_freq_lbl"))
        self.lbl_ct_curr.setText(self.tr("ct_curr_lbl"))
        self.lbl_ct_mask.setText(self.tr("ct_mask_lbl"))

        curr_mask_idx = self.cb_ct_mask.currentIndex()
        self.cb_ct_mask.blockSignals(True)
        self.cb_ct_mask.clear()
        self.cb_ct_mask.addItems([
            self.tr("ct_mask_ch0"),
            self.tr("ct_mask_ch1"),
            self.tr("ct_mask_both")
        ])
        if curr_mask_idx >= 0:
            self.cb_ct_mask.setCurrentIndex(curr_mask_idx)
        self.cb_ct_mask.blockSignals(False)

        self.card_ces1_res.set_title(self.tr("ces1_res_title"))
        self.card_ces2_res.set_title(self.tr("ces2_res_title"))

        self.lbl_ct_dur.setText(self.tr("ct_dur_lbl"))
        self.btn_ct_start.setText(self.tr("btn_ct_start"))
        self.btn_ct_stop.setText(self.tr("btn_ct_stop"))
        self.lbl_ct_progress.setText(self.tr("ct_progress_lbl"))

        self.box_stim.setTitle(self.tr("stim_title"))
        self.lbl_stim_chan.setText(self.tr("stim_chan_lbl"))

        curr_stim_chan_idx = self.cb_stim_chan.currentIndex()
        self.cb_stim_chan.blockSignals(True)
        self.cb_stim_chan.clear()
        self.cb_stim_chan.addItems([
            self.tr("stim_chan_0"),
            self.tr("stim_chan_1")
        ])
        if curr_stim_chan_idx >= 0:
            self.cb_stim_chan.setCurrentIndex(curr_stim_chan_idx)
        self.cb_stim_chan.blockSignals(False)

        self.lbl_stim_mode.setText(self.tr("stim_mode_lbl"))
        self.lbl_stim_freq.setText(self.tr("stim_freq_lbl"))
        self.lbl_stim_curr.setText(self.tr("stim_curr_lbl"))
        self.lbl_stim_trig.setText(self.tr("stim_trig_lbl"))
        self.lbl_stim_ru.setText(self.tr("stim_ramp_up_lbl"))
        self.lbl_stim_h.setText(self.tr("stim_hold_lbl"))
        self.lbl_stim_rd.setText(self.tr("stim_ramp_down_lbl"))
        self.btn_cfg_stim.setText(self.tr("btn_apply_stim"))
        self.btn_arm.setText(self.tr("btn_arm"))
        self.btn_disarm.setText(self.tr("btn_disarm"))
        self.btn_stim_start.setText(self.tr("btn_stim_start"))
        self.btn_stim_stop.setText(self.tr("btn_stim_stop"))

        # Update Stimulation Modes dropdown translated items
        current_mode_idx = self.cb_stim_mode.currentIndex()
        self.cb_stim_mode.blockSignals(True)
        self.cb_stim_mode.clear()
        self.cb_stim_mode.addItems([
            self.tr("stim_mode_square"),
            self.tr("stim_mode_dc"),
            self.tr("stim_mode_sine"),
            self.tr("stim_mode_triangle")
        ])
        if current_mode_idx >= 0:
            self.cb_stim_mode.setCurrentIndex(current_mode_idx)
        self.cb_stim_mode.blockSignals(False)

        # Update Contact Test and Stimulation live status
        if hasattr(self, "ct_state_key") and self.ct_state_key:
            if self.ct_state_key == "raw_status":
                self.lbl_ct_state.setText(self.ct_state_raw_text)
            else:
                self.lbl_ct_state.setText(self.tr(self.ct_state_key, **self.ct_state_args))

        if hasattr(self, "stim_state_key") and self.stim_state_key:
            self.lbl_stim_state.setText(self.tr(self.stim_state_key))

    # ==============================================================================
    # TAB 1: Connection & Control
    # ==============================================================================
    def setup_tab_control(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)

        # Left Panel - Connection Controls & Device Specs
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setSpacing(15)

        # Box 1: Connection manager
        self.box_conn = QtWidgets.QGroupBox(self.tr("conn_title"))
        conn_grid = QtWidgets.QGridLayout(self.box_conn)
        conn_grid.setContentsMargins(12, 20, 12, 12)

        # Connection Method
        self.lbl_conn_method = QtWidgets.QLabel(self.tr("conn_method_lbl"))
        conn_grid.addWidget(self.lbl_conn_method, 0, 0)
        self.cb_conn_method = QtWidgets.QComboBox()
        self.cb_conn_method.currentIndexChanged.connect(self.on_connection_method_changed)
        conn_grid.addWidget(self.cb_conn_method, 0, 1, 1, 2)

        # Serial Ports Widgets
        self.lbl_port_label = QtWidgets.QLabel(self.tr("port_lbl"))
        conn_grid.addWidget(self.lbl_port_label, 1, 0)
        self.cb_ports = QtWidgets.QComboBox()
        conn_grid.addWidget(self.cb_ports, 1, 1)

        self.btn_refresh = QtWidgets.QPushButton(self.tr("btn_refresh"))
        self.btn_refresh.clicked.connect(self.refresh_ports)
        conn_grid.addWidget(self.btn_refresh, 1, 2)

        # BLE Target Widgets
        self.lbl_ble_target = QtWidgets.QLabel(self.tr("ble_target_lbl"))
        conn_grid.addWidget(self.lbl_ble_target, 2, 0)
        self.cb_ble_names = QtWidgets.QComboBox()
        self.cb_ble_names.setEditable(True)
        self.cb_ble_names.addItems(["BrainSync", "BrainSync", "BrainSync-EEG"])
        conn_grid.addWidget(self.cb_ble_names, 2, 1, 1, 2)

        self.btn_connect = QtWidgets.QPushButton(self.tr("btn_connect"))
        self.btn_connect.setStyleSheet("""
            QPushButton {
                background-color: #145a32;
                border-color: #1e8449;
                color: #d4efdf;
                border: 1px solid #1e8449;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:disabled {
                background-color: #1d2b21;
                border-color: #243529;
                color: #5d7565;
            }
        """)
        self.btn_connect.clicked.connect(self.connect_device)
        conn_grid.addWidget(self.btn_connect, 3, 0, 1, 3)

        self.btn_disconnect = QtWidgets.QPushButton(self.tr("btn_disconnect"))
        self.btn_disconnect.setEnabled(False)
        self.btn_disconnect.setStyleSheet("""
            QPushButton {
                background-color: #78281f;
                border-color: #943126;
                color: #fadbd8;
                border: 1px solid #943126;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:disabled {
                background-color: #2b1f1d;
                border-color: #352624;
                color: #755f5d;
            }
        """)
        self.btn_disconnect.clicked.connect(self.disconnect_device)
        conn_grid.addWidget(self.btn_disconnect, 4, 0, 1, 3)

        left_layout.addWidget(self.box_conn)

        # Box 2: Device Stats Cards
        self.box_stats = QtWidgets.QGroupBox(self.tr("dev_status_title"))
        stats_layout = QtWidgets.QVBoxLayout(self.box_stats)
        stats_layout.setContentsMargins(12, 20, 12, 12)
        stats_layout.setSpacing(10)

        self.card_fw = StatCard(self.tr("fw_card"), "--", "#00ffff")
        self.card_bat = StatCard(self.tr("bat_card"), "--", "#00ffcc")
        self.card_charge = StatCard(self.tr("charge_card"), "--", "#ffcc00")

        stats_layout.addWidget(self.card_fw)
        stats_layout.addWidget(self.card_bat)
        stats_layout.addWidget(self.card_charge)

        self.btn_refresh_status = QtWidgets.QPushButton(self.tr("btn_refresh_status"))
        self.btn_refresh_status.setEnabled(False)
        self.btn_refresh_status.clicked.connect(self.refresh_device_status)
        stats_layout.addWidget(self.btn_refresh_status)

        left_layout.addWidget(self.box_stats)

        # Box 6: LSL bridge status
        self.box_lsl = QtWidgets.QGroupBox(self.tr("lsl_bridge_title"))
        lsl_v = QtWidgets.QVBoxLayout(self.box_lsl)
        lsl_v.setContentsMargins(12, 20, 12, 12)
        self.chk_lsl = QtWidgets.QCheckBox(self.tr("lsl_chk_lbl"))
        self.chk_lsl.setChecked(self.lsl_enabled)
        self.chk_lsl.stateChanged.connect(self.toggle_lsl)
        lsl_v.addWidget(self.chk_lsl)

        self.lbl_lsl_status = QtWidgets.QLabel("")
        self.update_lsl_indicator()
        lsl_v.addWidget(self.lbl_lsl_status)
        left_layout.addWidget(self.box_lsl)

        # Box 7: TriggerBox Synchronization Settings
        self.box_trb = QtWidgets.QGroupBox(self.tr("trb_group_title"))
        trb_v = QtWidgets.QVBoxLayout(self.box_trb)
        trb_v.setContentsMargins(12, 20, 12, 12)

        self.btn_trb_config = QtWidgets.QPushButton(self.tr("btn_trb_config"))
        self.btn_trb_config.clicked.connect(self.show_triggerbox_dialog)
        trb_v.addWidget(self.btn_trb_config)

        self.lbl_trb_status = QtWidgets.QLabel("")
        trb_v.addWidget(self.lbl_trb_status)

        left_layout.addWidget(self.box_trb)

        # Run state indicator update
        self.update_trb_indicator()

        left_layout.addStretch()

        # Right Panel - EDF Data Recorder Manager
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setSpacing(15)

        self.box_edf = QtWidgets.QGroupBox(self.tr("edf_title"))
        edf_v = QtWidgets.QVBoxLayout(self.box_edf)
        edf_v.setContentsMargins(12, 24, 12, 12)

        # File selector
        file_h = QtWidgets.QHBoxLayout()
        self.lbl_filename_label = QtWidgets.QLabel(self.tr("filename_lbl"))
        file_h.addWidget(self.lbl_filename_label)
        self.txt_edf_filename = QtWidgets.QLineEdit()
        self.txt_edf_filename.setText("recording_complete.bdf")
        file_h.addWidget(self.txt_edf_filename)
        edf_v.addLayout(file_h)

        # Record button row
        rec_h = QtWidgets.QHBoxLayout()
        self.btn_edf_start = QtWidgets.QPushButton(self.tr("btn_edf_start"))
        self.btn_edf_start.setStyleSheet("background-color: #21618c; border-color: #2e86c1;")
        self.btn_edf_start.clicked.connect(self.start_recording)
        rec_h.addWidget(self.btn_edf_start)

        self.btn_edf_stop = QtWidgets.QPushButton(self.tr("btn_edf_stop"))
        self.btn_edf_stop.setEnabled(False)
        self.btn_edf_stop.setStyleSheet("background-color: #7b7d7d; border-color: #95a5a6;")
        self.btn_edf_stop.clicked.connect(self.stop_recording)
        rec_h.addWidget(self.btn_edf_stop)

        edf_v.addLayout(rec_h)

        # Annotation Box
        self.box_ann = QtWidgets.QGroupBox(self.tr("ann_grp"))
        self.box_ann.setStyleSheet("QGroupBox { border: 1px solid #222233; color: #bbbbcc; margin-top: 10px; font-size: 11px; }")
        ann_h = QtWidgets.QHBoxLayout(self.box_ann)
        self.txt_annotation = QtWidgets.QLineEdit()
        self.txt_annotation.setPlaceholderText(self.tr("ann_placeholder"))
        ann_h.addWidget(self.txt_annotation)

        self.btn_send_ann = QtWidgets.QPushButton(self.tr("btn_ann_send"))
        self.btn_send_ann.clicked.connect(self.send_annotation)
        ann_h.addWidget(self.btn_send_ann)
        edf_v.addWidget(self.box_ann)

        self.lbl_recording_info = QtWidgets.QLabel(self.tr("edf_inactive"))
        self.lbl_recording_info.setStyleSheet("color: #888899; font-style: italic;")
        edf_v.addWidget(self.lbl_recording_info)

        right_layout.addWidget(self.box_edf)

        # Rapid Event Marker Box
        self.box_markers = QtWidgets.QGroupBox(self.tr("marker_title"))
        m_lay = QtWidgets.QVBoxLayout(self.box_markers)
        m_lay.setContentsMargins(12, 20, 12, 12)

        # Hotkey marker buttons
        btn_grid = QtWidgets.QGridLayout()
        self.marker_buttons = []
        for i in range(5):
            btn = QtWidgets.QPushButton(self.tr("btn_marker", tag=f"F{i+1} - {self.event_tags[i]}"))
            btn.clicked.connect(lambda checked=False, idx=i: self.post_rapid_marker(idx))
            btn_grid.addWidget(btn, i // 2, i % 2)
            self.marker_buttons.append(btn)
        m_lay.addLayout(btn_grid)

        self.lbl_history_label = QtWidgets.QLabel(self.tr("history_lbl"))
        m_lay.addWidget(self.lbl_history_label)
        self.list_marker_history = QtWidgets.QListWidget()
        self.list_marker_history.setFixedHeight(80)
        m_lay.addWidget(self.list_marker_history)

        right_layout.addWidget(self.box_markers)

        # Logging Panel
        self.box_log = QtWidgets.QGroupBox(self.tr("log_title"))
        log_v = QtWidgets.QVBoxLayout(self.box_log)
        log_v.setContentsMargins(12, 20, 12, 12)

        self.txt_log = QtWidgets.QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFocusPolicy(QtCore.Qt.NoFocus)
        self.txt_log.setStyleSheet("background-color: #050508; border: 1px solid #202030; color: #a6ffb2; font-family: monospace; font-size: 11px;")
        log_v.addWidget(self.txt_log)

        right_layout.addWidget(self.box_log)

        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 2)

        self.tabs.addTab(tab, self.tr("conn_title"))
        self.refresh_ports()

    def toggle_lsl(self, state):
        self.lsl_enabled = (state == QtCore.Qt.Checked)
        if self.lsl_enabled:
            self.init_lsl_outlets()
        else:
            self.lsl_eeg_outlet = None
            self.lsl_marker_outlet = None
            self.write_log("LSL Bridge deactivated.")
        self.update_lsl_indicator()

    def update_lsl_indicator(self):
        if not _LSL_AVAILABLE:
            self.lbl_lsl_status.setText(self.tr("lsl_err_lbl"))
            self.lbl_lsl_status.setStyleSheet("color: #ff3333; font-weight: bold;")
            self.chk_lsl.setEnabled(False)
        elif not self.lsl_enabled:
            self.lbl_lsl_status.setText(self.tr("lsl_status_off"))
            self.lbl_lsl_status.setStyleSheet("color: #888899;")
        else:
            self.lbl_lsl_status.setText(self.tr("lsl_status_ok"))
            self.lbl_lsl_status.setStyleSheet("color: #00ffcc; font-weight: bold;")

    # ==============================================================================
    # Rapid Event Marker Implementation (hotkeys 1-4)
    # ==============================================================================
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        key = event.key()
        # Intercept hotkeys F1-F5
        if key == QtCore.Qt.Key_F1:
            self.post_rapid_marker(0)
        elif key == QtCore.Qt.Key_F2:
            self.post_rapid_marker(1)
        elif key == QtCore.Qt.Key_F3:
            self.post_rapid_marker(2)
        elif key == QtCore.Qt.Key_F4:
            self.post_rapid_marker(3)
        elif key == QtCore.Qt.Key_F5:
            self.post_rapid_marker(4)
        else:
            super().keyPressEvent(event)

    def post_rapid_marker(self, index):
        tag = self.event_tags[index]
        self.inject_marker(tag)

    # ==============================================================================
    # TAB 2: EEG & CES Stream Plots
    # ==============================================================================
    def setup_tab_eeg(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)

        # Left Panel - Plots
        plot_layout = QtWidgets.QVBoxLayout()

        # EEG Plots (Single stacked canvas)
        self.eeg_group = QtWidgets.QGroupBox(self.tr("eeg_plot_title"))
        eeg_v = QtWidgets.QVBoxLayout(self.eeg_group)
        eeg_v.setContentsMargins(5, 15, 5, 5)

        # Channel Filter Checkboxes
        ch_layout = QtWidgets.QHBoxLayout()
        ch_layout.setContentsMargins(6, 0, 6, 4)
        self.chk_channels = []
        for i in range(8):
            chk = QtWidgets.QCheckBox(self.eeg_ch_names[i])
            chk.setChecked(self.channel_config.active_mask & (1 << i) != 0)
            chk.stateChanged.connect(self.update_channel_visibility)
            ch_layout.addWidget(chk)
            self.chk_channels.append(chk)
        eeg_v.addLayout(ch_layout)

        self.eeg_plot_widget = pg.PlotWidget()
        self.eeg_plot_widget.setBackground('#08080c')
        self.eeg_plot_widget.showGrid(x=True, y=True, alpha=0.1)
        self.eeg_plot_widget.setMouseEnabled(x=False, y=False)
        eeg_v.addWidget(self.eeg_plot_widget)

        # Initialize Y-ticks mapping for stacked channels
        ay = self.eeg_plot_widget.getAxis('left')
        ticks = []
        for i in range(8):
            baseline = (7 - i) * self.eeg_offset_gap
            ticks.append((baseline, self.eeg_ch_names[i]))
        ay.setTicks([ticks])
        self.eeg_plot_widget.setYRange(-self.eeg_offset_gap, self.eeg_offset_gap * 8)

        self.eeg_curves = []
        colors = ['#FF5555', '#55FF55', '#5555FF', '#FFFF55', '#FF55FF', '#55FFFF', '#FF9900', '#00FFCC']

        for i in range(8):
            curve = self.eeg_plot_widget.plot(pen=pg.mkPen(color=colors[i], width=1.5))
            self.eeg_curves.append(curve)

        plot_layout.addWidget(self.eeg_group, 5)

        # CES Plot Widget
        self.ces_group = QtWidgets.QGroupBox(self.tr("ces_plot_title"))
        ces_v = QtWidgets.QVBoxLayout(self.ces_group)
        ces_v.setContentsMargins(5, 15, 5, 5)

        self.ces_plot_widget = pg.PlotWidget()
        self.ces_plot_widget.setBackground('#08080c')
        self.ces_plot_widget.showGrid(x=True, y=True, alpha=0.1)
        self.ces_plot_widget.setMouseEnabled(x=False, y=False)
        self.ces_plot_widget.setYRange(0, 4096) # 12-bit ADC
        self.ces_curves = [
            self.ces_plot_widget.plot(pen=pg.mkPen(color='#ffaa00', width=1.5), name="CES1"),
            self.ces_plot_widget.plot(pen=pg.mkPen(color='#00ffcc', width=1.5), name="CES2")
        ]
        ces_v.addWidget(self.ces_plot_widget)

        plot_layout.addWidget(self.ces_group, 2)

        # Right Panel - Control and settings
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setSpacing(15)

        self.box_stream_ctrl = QtWidgets.QGroupBox(self.tr("stream_switches_title"))
        v_stream = QtWidgets.QVBoxLayout(self.box_stream_ctrl)
        v_stream.setContentsMargins(12, 20, 12, 12)
        v_stream.setSpacing(10)

        self.btn_eeg_start = QtWidgets.QPushButton(self.tr("btn_eeg_start"))
        self.btn_eeg_start.clicked.connect(lambda: self.toggle_sensor_transfer("eeg"))
        v_stream.addWidget(self.btn_eeg_start)

        self.btn_ces_start = QtWidgets.QPushButton(self.tr("btn_ces_start"))
        self.btn_ces_start.clicked.connect(lambda: self.toggle_sensor_transfer("ces"))
        v_stream.addWidget(self.btn_ces_start)

        right_panel.addWidget(self.box_stream_ctrl)

        # EEG Config Card
        self.box_eeg_cfg = QtWidgets.QGroupBox(self.tr("eeg_config_title"))
        v_eeg_cfg = QtWidgets.QVBoxLayout(self.box_eeg_cfg)
        v_eeg_cfg.setContentsMargins(12, 20, 12, 12)
        v_eeg_cfg.setSpacing(8)

        v_eeg_cfg.addWidget(QtWidgets.QLabel("Sample Rate:"))
        self.cb_eeg_sr = QtWidgets.QComboBox()
        self.cb_eeg_sr.addItems(["250 Hz", "500 Hz", "1000 Hz", "2000 Hz", "4000 Hz"])
        v_eeg_cfg.addWidget(self.cb_eeg_sr)

        v_eeg_cfg.addWidget(QtWidgets.QLabel("Gain:"))
        self.cb_eeg_gain = QtWidgets.QComboBox()
        self.cb_eeg_gain.addItems(["Gain 1", "Gain 2", "Gain 4", "Gain 6", "Gain 8", "Gain 12", "Gain 24"])
        self.cb_eeg_gain.setCurrentIndex(6) # Gain 24 default
        v_eeg_cfg.addWidget(self.cb_eeg_gain)

        self.btn_apply_eeg = QtWidgets.QPushButton(self.tr("apply_settings"))
        self.btn_apply_eeg.clicked.connect(self.apply_eeg_config)
        v_eeg_cfg.addWidget(self.btn_apply_eeg)

        montage_buttons = QtWidgets.QHBoxLayout()
        self.btn_edit_montage = QtWidgets.QPushButton(self.tr("btn_edit_montage"))
        self.btn_edit_montage.clicked.connect(self.edit_montage_config)
        montage_buttons.addWidget(self.btn_edit_montage)
        self.btn_load_channel_config = QtWidgets.QPushButton(self.tr("btn_load_montage"))
        self.btn_load_channel_config.clicked.connect(self.reload_channel_config)
        montage_buttons.addWidget(self.btn_load_channel_config)
        self.btn_save_channel_config = QtWidgets.QPushButton(self.tr("btn_save_montage"))
        self.btn_save_channel_config.clicked.connect(self.persist_channel_config)
        montage_buttons.addWidget(self.btn_save_channel_config)
        v_eeg_cfg.addLayout(montage_buttons)

        # Stacked Plot Base Offset settings
        self.lbl_eeg_offset_lbl = QtWidgets.QLabel(self.tr("plot_offset_lbl"))
        v_eeg_cfg.addWidget(self.lbl_eeg_offset_lbl)
        self.spin_eeg_offset = QtWidgets.QSpinBox()
        self.spin_eeg_offset.setRange(50, 5000)
        self.spin_eeg_offset.setValue(300)
        self.spin_eeg_offset.setSingleStep(50)
        self.spin_eeg_offset.valueChanged.connect(self.update_eeg_plot_ticks)
        v_eeg_cfg.addWidget(self.spin_eeg_offset)

        right_panel.addWidget(self.box_eeg_cfg)

        # Software filters card
        self.box_filter = QtWidgets.QGroupBox(self.tr("sw_filters_title"))
        v_filter = QtWidgets.QVBoxLayout(self.box_filter)
        v_filter.setContentsMargins(12, 20, 12, 12)

        self.chk_bp = QtWidgets.QCheckBox("Bandpass (1-45 Hz)")
        self.chk_bp.setChecked(True)
        self.chk_bp.stateChanged.connect(self.filter_setting_changed)
        v_filter.addWidget(self.chk_bp)

        self.chk_n50 = QtWidgets.QCheckBox("Notch Filter (50 Hz)")
        self.chk_n50.setChecked(True)
        self.chk_n50.stateChanged.connect(self.filter_setting_changed)
        v_filter.addWidget(self.chk_n50)

        self.chk_n60 = QtWidgets.QCheckBox("Notch Filter (60 Hz)")
        self.chk_n60.setChecked(True)
        self.chk_n60.stateChanged.connect(self.filter_setting_changed)
        v_filter.addWidget(self.chk_n60)

        right_panel.addWidget(self.box_filter)

        # Performance Stat Box
        self.box_perf = QtWidgets.QGroupBox(self.tr("sig_quality_title"))
        v_perf = QtWidgets.QVBoxLayout(self.box_perf)
        v_perf.setContentsMargins(12, 20, 12, 12)

        self.lbl_eeg_pkts = QtWidgets.QLabel("EEG " + self.tr("total_pkts", count=0))
        self.lbl_eeg_loss = QtWidgets.QLabel("EEG " + self.tr("loss_rate", rate=0.00))
        self.lbl_ces_pkts = QtWidgets.QLabel("CES " + self.tr("total_pkts", count=0))
        self.lbl_ces_loss = QtWidgets.QLabel("CES " + self.tr("loss_rate", rate=0.00))

        v_perf.addWidget(self.lbl_eeg_pkts)
        v_perf.addWidget(self.lbl_eeg_loss)
        v_perf.addWidget(self.lbl_ces_pkts)
        v_perf.addWidget(self.lbl_ces_loss)

        right_panel.addWidget(self.box_perf)
        right_panel.addStretch()

        layout.addLayout(plot_layout, 5)
        layout.addLayout(right_panel, 1)

        self.tabs.addTab(tab, self.tr("tab_eeg"))

    def filter_setting_changed(self):
        self.eeg_filter.enabled_bp = self.chk_bp.isChecked()
        self.eeg_filter.enabled_notch50 = self.chk_n50.isChecked()
        self.eeg_filter.enabled_notch60 = self.chk_n60.isChecked()

    # ==============================================================================
    # TAB 3: Motion & Mag Plots
    # ==============================================================================
    def setup_tab_motion(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)

        # Plots area
        plot_layout = QtWidgets.QVBoxLayout()

        # 1. IMU Accel
        self.box_acc = QtWidgets.QGroupBox(self.tr("imu_acc_title"))
        v_acc = QtWidgets.QVBoxLayout(self.box_acc)
        v_acc.setContentsMargins(5, 15, 5, 5)

        self.plot_acc = pg.PlotWidget()
        self.plot_acc.setBackground('#08080c')
        self.plot_acc.showGrid(x=True, y=True, alpha=0.1)
        self.plot_acc.setMouseEnabled(x=False, y=False)
        self.plot_acc.addLegend()
        self.acc_curves = [
            self.plot_acc.plot(pen=pg.mkPen(color='#ff3333', width=1.5), name="Acc X"),
            self.plot_acc.plot(pen=pg.mkPen(color='#33ff33', width=1.5), name="Acc Y"),
            self.plot_acc.plot(pen=pg.mkPen(color='#3333ff', width=1.5), name="Acc Z")
        ]
        v_acc.addWidget(self.plot_acc)
        plot_layout.addWidget(self.box_acc)

        # 2. IMU Gyro
        self.box_gyro = QtWidgets.QGroupBox(self.tr("imu_gyro_title"))
        v_gyro = QtWidgets.QVBoxLayout(self.box_gyro)
        v_gyro.setContentsMargins(5, 15, 5, 5)

        self.plot_gyro = pg.PlotWidget()
        self.plot_gyro.setBackground('#08080c')
        self.plot_gyro.showGrid(x=True, y=True, alpha=0.1)
        self.plot_gyro.setMouseEnabled(x=False, y=False)
        self.plot_gyro.addLegend()
        self.gyro_curves = [
            self.plot_gyro.plot(pen=pg.mkPen(color='#ff3333', width=1.5), name="Gyro X"),
            self.plot_gyro.plot(pen=pg.mkPen(color='#33ff33', width=1.5), name="Gyro Y"),
            self.plot_gyro.plot(pen=pg.mkPen(color='#3333ff', width=1.5), name="Gyro Z")
        ]
        v_gyro.addWidget(self.plot_gyro)
        plot_layout.addWidget(self.box_gyro)

        # 3. Magnetometer
        self.box_mag = QtWidgets.QGroupBox(self.tr("mag_title"))
        v_mag = QtWidgets.QVBoxLayout(self.box_mag)
        v_mag.setContentsMargins(5, 15, 5, 5)

        self.plot_mag = pg.PlotWidget()
        self.plot_mag.setBackground('#08080c')
        self.plot_mag.showGrid(x=True, y=True, alpha=0.1)
        self.plot_mag.setMouseEnabled(x=False, y=False)
        self.plot_mag.addLegend()
        self.mag_curves = [
            self.plot_mag.plot(pen=pg.mkPen(color='#ffaa00', width=1.5), name="Mag X"),
            self.plot_mag.plot(pen=pg.mkPen(color='#00ffcc', width=1.5), name="Mag Y"),
            self.plot_mag.plot(pen=pg.mkPen(color='#ff00ff', width=1.5), name="Mag Z")
        ]
        v_mag.addWidget(self.plot_mag)
        plot_layout.addWidget(self.box_mag)

        # Right control panel
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setSpacing(15)

        self.box_motion_ctrl = QtWidgets.QGroupBox(self.tr("motion_switches_title"))
        v_motion = QtWidgets.QVBoxLayout(self.box_motion_ctrl)
        v_motion.setContentsMargins(12, 20, 12, 12)
        v_motion.setSpacing(10)

        self.btn_imu_start = QtWidgets.QPushButton(self.tr("btn_imu_start"))
        self.btn_imu_start.clicked.connect(lambda: self.toggle_sensor_transfer("imu"))
        v_motion.addWidget(self.btn_imu_start)

        self.btn_mag_start = QtWidgets.QPushButton(self.tr("btn_mag_start"))
        self.btn_mag_start.clicked.connect(lambda: self.toggle_sensor_transfer("mag"))
        v_motion.addWidget(self.btn_mag_start)

        right_panel.addWidget(self.box_motion_ctrl)
        right_panel.addStretch()

        layout.addLayout(plot_layout, 5)
        layout.addLayout(right_panel, 1)

        self.tabs.addTab(tab, self.tr("tab_motion"))

    # ==============================================================================
    # TAB 4: Stimulation & Contact Test Controls
    # ==============================================================================
    def setup_tab_stimulation(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)

        # Left Panel: Electrode Contact Test
        left_layout = QtWidgets.QVBoxLayout()

        self.box_ct = QtWidgets.QGroupBox(self.tr("ct_title"))
        ct_grid = QtWidgets.QGridLayout(self.box_ct)
        ct_grid.setContentsMargins(15, 24, 15, 15)
        ct_grid.setSpacing(12)

        self.lbl_ct_freq = QtWidgets.QLabel(self.tr("ct_freq_lbl"))
        ct_grid.addWidget(self.lbl_ct_freq, 0, 0)
        self.cb_ct_freq = QtWidgets.QComboBox()
        self.cb_ct_freq.addItems(["10 Hz", "20 Hz", "30 Hz", "100 Hz"])
        ct_grid.addWidget(self.cb_ct_freq, 0, 1)

        self.lbl_ct_curr = QtWidgets.QLabel(self.tr("ct_curr_lbl"))
        ct_grid.addWidget(self.lbl_ct_curr, 0, 2)
        self.spin_ct_current = QtWidgets.QSpinBox()
        self.spin_ct_current.setRange(10, 500)
        self.spin_ct_current.setValue(100)
        ct_grid.addWidget(self.spin_ct_current, 0, 3)

        self.lbl_ct_mask = QtWidgets.QLabel(self.tr("ct_mask_lbl"))
        ct_grid.addWidget(self.lbl_ct_mask, 1, 0)
        self.cb_ct_mask = QtWidgets.QComboBox()
        self.cb_ct_mask.addItems([
            self.tr("ct_mask_ch0"),
            self.tr("ct_mask_ch1"),
            self.tr("ct_mask_both")
        ])
        self.cb_ct_mask.setCurrentIndex(2) # Both channels default
        ct_grid.addWidget(self.cb_ct_mask, 1, 1)

        self.lbl_ct_dur = QtWidgets.QLabel(self.tr("ct_dur_lbl"))
        ct_grid.addWidget(self.lbl_ct_dur, 1, 2)
        self.spin_ct_duration = QtWidgets.QSpinBox()
        self.spin_ct_duration.setRange(0, 60) # 0 means infinite
        self.spin_ct_duration.setValue(3)
        ct_grid.addWidget(self.spin_ct_duration, 1, 3)

        self.btn_ct_start = QtWidgets.QPushButton(self.tr("btn_ct_start"))
        self.btn_ct_start.setStyleSheet("background-color: #21618c; border-color: #2e86c1;")
        self.btn_ct_start.clicked.connect(self.start_contact_test)
        ct_grid.addWidget(self.btn_ct_start, 2, 0, 1, 2)

        self.btn_ct_stop = QtWidgets.QPushButton(self.tr("btn_ct_stop"))
        self.btn_ct_stop.setEnabled(False)
        self.btn_ct_stop.setStyleSheet("background-color: #7b7d7d; border-color: #95a5a6;")
        self.btn_ct_stop.clicked.connect(self.stop_contact_test)
        ct_grid.addWidget(self.btn_ct_stop, 2, 2, 1, 2)

        self.lbl_ct_progress = QtWidgets.QLabel(self.tr("ct_progress_lbl"))
        ct_grid.addWidget(self.lbl_ct_progress, 3, 0)
        self.ct_progress = QtWidgets.QProgressBar()
        self.ct_progress.setValue(0)
        ct_grid.addWidget(self.ct_progress, 3, 1, 1, 3)

        self.card_ces1_res = StatCard(self.tr("ces1_res_title"), "--", "#ff3333")
        self.card_ces2_res = StatCard(self.tr("ces2_res_title"), "--", "#ff3333")
        ct_grid.addWidget(self.card_ces1_res, 4, 0, 1, 2)
        ct_grid.addWidget(self.card_ces2_res, 4, 2, 1, 2)

        self.lbl_ct_state = QtWidgets.QLabel(self.tr("ct_status_idle"))
        self.lbl_ct_state.setStyleSheet("font-style: italic; color: #888899;")
        ct_grid.addWidget(self.lbl_ct_state, 5, 0, 1, 4)

        left_layout.addWidget(self.box_ct)
        left_layout.addStretch()

        # Right Panel: Stimulation parameters & control
        right_layout = QtWidgets.QVBoxLayout()

        self.box_stim = QtWidgets.QGroupBox(self.tr("stim_title"))
        stim_v = QtWidgets.QVBoxLayout(self.box_stim)
        stim_v.setContentsMargins(15, 24, 15, 15)
        stim_v.setSpacing(10)

        # Channel Select
        ch_layout = QtWidgets.QHBoxLayout()
        self.lbl_stim_chan = QtWidgets.QLabel(self.tr("stim_chan_lbl"))
        ch_layout.addWidget(self.lbl_stim_chan)
        self.cb_stim_chan = QtWidgets.QComboBox()
        self.cb_stim_chan.addItems([self.tr("stim_chan_0"), self.tr("stim_chan_1")])
        ch_layout.addWidget(self.cb_stim_chan)
        stim_v.addLayout(ch_layout)

        # Params Grid
        grid_params = QtWidgets.QGridLayout()
        grid_params.setSpacing(8)

        self.lbl_stim_mode = QtWidgets.QLabel(self.tr("stim_mode_lbl"))
        grid_params.addWidget(self.lbl_stim_mode, 0, 0)
        self.cb_stim_mode = QtWidgets.QComboBox()
        self.cb_stim_mode.addItems([
            self.tr("stim_mode_square"),
            self.tr("stim_mode_dc"),
            self.tr("stim_mode_sine"),
            self.tr("stim_mode_triangle")
        ])
        grid_params.addWidget(self.cb_stim_mode, 0, 1)

        self.lbl_stim_freq = QtWidgets.QLabel(self.tr("stim_freq_lbl"))
        grid_params.addWidget(self.lbl_stim_freq, 0, 2)
        self.spin_stim_freq = QtWidgets.QDoubleSpinBox()
        self.spin_stim_freq.setRange(0.1, 250.0)
        self.spin_stim_freq.setValue(2.0)
        self.spin_stim_freq.setSingleStep(0.5)
        grid_params.addWidget(self.spin_stim_freq, 0, 3)

        self.lbl_stim_curr = QtWidgets.QLabel(self.tr("stim_curr_lbl"))
        grid_params.addWidget(self.lbl_stim_curr, 1, 0)
        self.spin_stim_current = QtWidgets.QSpinBox()
        self.spin_stim_current.setRange(50, 4000)
        self.spin_stim_current.setValue(1000)
        self.spin_stim_current.setSingleStep(100)
        grid_params.addWidget(self.spin_stim_current, 1, 1)

        self.lbl_stim_trig = QtWidgets.QLabel(self.tr("stim_trig_lbl"))
        grid_params.addWidget(self.lbl_stim_trig, 1, 2)
        self.spin_stim_label = QtWidgets.QSpinBox()
        self.spin_stim_label.setRange(1, 255)
        self.spin_stim_label.setValue(1)
        grid_params.addWidget(self.spin_stim_label, 1, 3)

        stim_v.addLayout(grid_params)

        # Durations Box
        grid_dur = QtWidgets.QGridLayout()
        grid_dur.setSpacing(8)

        self.lbl_stim_ru = QtWidgets.QLabel(self.tr("stim_ramp_up_lbl"))
        grid_dur.addWidget(self.lbl_stim_ru, 0, 0)
        self.spin_ramp_up = QtWidgets.QSpinBox()
        self.spin_ramp_up.setRange(0, 10000)
        self.spin_ramp_up.setValue(1000)
        grid_dur.addWidget(self.spin_ramp_up, 0, 1)

        self.lbl_stim_h = QtWidgets.QLabel(self.tr("stim_hold_lbl"))
        grid_dur.addWidget(self.lbl_stim_h, 0, 2)
        self.spin_hold = QtWidgets.QSpinBox()
        self.spin_hold.setRange(0, 600000)
        self.spin_hold.setValue(2000)
        grid_dur.addWidget(self.spin_hold, 0, 3)

        self.lbl_stim_rd = QtWidgets.QLabel(self.tr("stim_ramp_down_lbl"))
        grid_dur.addWidget(self.lbl_stim_rd, 1, 0)
        self.spin_ramp_down = QtWidgets.QSpinBox()
        self.spin_ramp_down.setRange(0, 10000)
        self.spin_ramp_down.setValue(1000)
        grid_dur.addWidget(self.spin_ramp_down, 1, 1)

        stim_v.addLayout(grid_dur)

        # Configure button
        self.btn_cfg_stim = QtWidgets.QPushButton(self.tr("btn_apply_stim"))
        self.btn_cfg_stim.clicked.connect(self.apply_stim_params)
        stim_v.addWidget(self.btn_cfg_stim)

        # Stimulation states
        state_layout = QtWidgets.QHBoxLayout()
        self.btn_arm = QtWidgets.QPushButton(self.tr("btn_arm"))
        self.btn_arm.setStyleSheet("background-color: #d35400; border-color: #e67e22; color: #fdf2e9;")
        self.btn_arm.clicked.connect(self.arm_stimulation)
        state_layout.addWidget(self.btn_arm)

        self.btn_disarm = QtWidgets.QPushButton(self.tr("btn_disarm"))
        self.btn_disarm.setEnabled(False)
        self.btn_disarm.clicked.connect(self.disarm_stimulation)
        state_layout.addWidget(self.btn_disarm)

        stim_v.addLayout(state_layout)

        # Direct Action Control (Start / Stop)
        act_layout = QtWidgets.QHBoxLayout()
        self.btn_stim_start = QtWidgets.QPushButton(self.tr("btn_stim_start"))
        self.btn_stim_start.setEnabled(False)
        self.btn_stim_start.setStyleSheet("background-color: #27ae60; border-color: #2ecc71; color: #eafaf1;")
        self.btn_stim_start.clicked.connect(self.start_stimulation_output)
        act_layout.addWidget(self.btn_stim_start)

        self.btn_stim_stop = QtWidgets.QPushButton(self.tr("btn_stim_stop"))
        self.btn_stim_stop.setEnabled(False)
        self.btn_stim_stop.setStyleSheet("background-color: #c0392b; border-color: #e74c3c; color: #fdedec;")
        self.btn_stim_stop.clicked.connect(self.stop_stimulation_output)
        act_layout.addWidget(self.btn_stim_stop)

        stim_v.addLayout(act_layout)

        self.lbl_stim_state = QtWidgets.QLabel(self.tr("stim_state_disarmed"))
        self.lbl_stim_state.setStyleSheet("font-weight: bold; color: #e74c3c;")
        stim_v.addWidget(self.lbl_stim_state)

        right_layout.addWidget(self.box_stim)
        right_layout.addStretch()

        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 1)

        self.tabs.addTab(tab, self.tr("tab_stim"))

    # ==============================================================================
    # Utility Log helper
    # ==============================================================================
    def write_log(self, text):
        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted_log = f"[{now}] {text}"
        self.txt_log.append(formatted_log)
        scrollbar = self.txt_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        print(formatted_log, flush=True)

    # ==============================================================================
    # Hardware Connection Logic (Serial & BLE Wrapper)
    # ==============================================================================
    def refresh_ports(self):
        self.cb_ports.clear()
        if self.mock_mode:
            self.cb_ports.addItem("MOCK_DEVICE_PORT")
            return

        try:
            try:
                # 优先调用升级后的 list_serial_ports()，它默认会匹配所有 BrainSync 兼容的 VID/PID 组合
                ports = sdk.list_serial_ports()
            except (TypeError, AttributeError):
                # 兼容旧版本只支持 2 个必填位置参数的 SDK，合并新旧端口
                try:
                    ports = sdk.list_serial_ports(0x5243, 0x0008) + sdk.list_serial_ports(0xCAFE, 0x4012)
                except Exception:
                    ports = sdk.list_serial_ports(0xCAFE, 0x4012)

            if not ports:
                import serial.tools.list_ports
                ports = [p.device for p in serial.tools.list_ports.comports()]

            import sys
            for p in ports:
                if sys.platform == 'darwin' and 'usb' not in p.lower():
                    continue
                self.cb_ports.addItem(p)

            if self.cb_ports.count() == 0:
                self.cb_ports.addItem("No serial ports found")
        except Exception as e:
            self.write_log(f"Error listing ports: {e}")

    def connect_device(self):
        is_ble = self.cb_conn_method.currentIndex() == 1
        self.connected_method = "ble" if is_ble else "serial"

        target = ""
        if is_ble:
            target = self.cb_ble_names.currentText()
            if not target:
                self.write_log("No BLE target specified!")
                return
        else:
            target = self.cb_ports.currentText()
            if not target or target == "No serial ports found":
                self.write_log("No port selected!")
                return

        self.write_log(f"Connecting via {self.connected_method.upper()}: {target}...")
        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(False)
        self.set_connection_controls_enabled(False)

        if self.mock_mode:
            self.handle = 9999
            self.lbl_connection_status.setText(self.tr("status_connected_mock"))
            self.lbl_connection_status.setStyleSheet("color: #00ffcc; font-weight: bold;")
            self.btn_disconnect.setEnabled(True)
            self.btn_connect.setEnabled(False)
            self.set_device_controls_enabled(True)
            self.card_fw.set_value("BrainSync (Mock V1.0)")
            self.write_log("Mock connection successfully established!")

            # Start simulated hardware task
            self.mock_task = asyncio.create_task(self.run_mock_stream())
            return

        asyncio.create_task(self.async_connect(is_ble, target))

    async def async_connect(self, is_ble, target):
        try:
            if is_ble:
                if ":" in target or "-" in target or len(target) >= 12:
                    self.write_log(f"Routing BLE connection by MAC address: {target}")
                    handle = await sdk.open_brainsync_ble_by_id(target)
                else:
                    self.write_log(f"Routing BLE connection by Name: {target}")
                    handle = await sdk.open_brainsync_ble(target)
            else:
                handle = await sdk.open_brainsync_serial()

            self.handle = handle
            self.connected_method = "ble" if is_ble else "serial"

            # Stop any residual streams to prevent Ingress queue lag
            self.write_log("Stopping any potential residual streams on open...")
            try:
                await sdk.set_eeg_transfer(self.handle, False)
                await sdk.set_adc_transfer(self.handle, False)
                await sdk.set_imu_transfer(self.handle, False)
                await sdk.set_mag_transfer(self.handle, False)
                await sdk.disarm_stimulation(self.handle)
                await asyncio.sleep(0.5)
            except Exception as reset_err:
                self.write_log(f"Warning: Failed to reset residual streams: {reset_err}")

            # Set connection UI states on success
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.set_connection_controls_enabled(False)
            self.set_device_controls_enabled(True)

            if is_ble:
                self.lbl_connection_status.setText(self.tr("status_connected_ble", target=target))
            else:
                self.lbl_connection_status.setText(self.tr("status_connected", port=target))
            self.lbl_connection_status.setStyleSheet("color: #00ffcc; font-weight: bold;")
            self.write_log(f"Connected. Handle ID: {handle}")

            # Try to get FW version
            try:
                fw = await sdk.get_firmware_version(self.handle)
                if isinstance(fw, dict):
                    dev_type = fw.get("device_type", "BrainSync")
                    sw = fw.get("sw_version", "N/A")
                    hw = fw.get("hw_version", "N/A")
                    fw_str = f"{dev_type} (SW: {sw}, HW: {hw})"
                else:
                    fw_str = str(fw)
                self.card_fw.set_value(fw_str)
            except Exception as ex:
                self.write_log(f"Warning: Failed to get firmware version: {ex}")
                self.card_fw.set_value("Unknown")

            # Try to get initial battery status
            try:
                bat = await sdk.get_battery_status(self.handle)
                self.on_battery_received([bat])
                self.write_log("Initial battery status fetched successfully.")
            except Exception as ex:
                self.write_log(f"Warning: Failed to get initial battery status: {ex}")

            # Try to subscribe to battery status
            try:
                await sdk.subscribe_adc_battery_status(self.handle, 1, self.on_battery_received)
                self.running_streams["battery"] = True
                self.write_log("Subscribed to battery status notifications.")
            except Exception as ex:
                self.write_log(f"Warning: Failed to subscribe to battery status: {ex}")

            self.write_log("Device initialized successfully.")

        except Exception as e:
            self.write_log(f"Connection failed: {e}")
            self.handle = None
            self.btn_connect.setEnabled(True)
            self.btn_disconnect.setEnabled(False)
            self.set_connection_controls_enabled(True)
            self.set_device_controls_enabled(False)
            self.lbl_connection_status.setText(self.tr("status_disconnected"))
            self.lbl_connection_status.setStyleSheet("color: #ff5555; font-weight: bold;")

    def on_connection_state_changed(self, device_id, state):
        self.write_log(f"[BLE Event] Device ID={device_id} connection state changed to: {state}")
        if state == "Disconnected":
            self.sig_device_disconnected.emit(device_id)

    def on_device_disconnected_signal(self, device_id):
        if self.handle is not None:
            self.write_log(f"Handling unexpected device disconnection (ID: {device_id})...")
            self.handle_unexpected_disconnection()

    def handle_unexpected_disconnection(self):
        self.handle = None
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.set_connection_controls_enabled(True)
        self.set_device_controls_enabled(False)
        self.lbl_connection_status.setText(self.tr("status_disconnected"))
        self.lbl_connection_status.setStyleSheet("color: #ff5555; font-weight: bold;")
        self.running_streams.clear()

        # Reset active stream buttons text
        self.btn_eeg_start.setText(self.tr("btn_eeg_start"))
        self.btn_ces_start.setText(self.tr("btn_ces_start"))
        self.btn_imu_start.setText(self.tr("btn_imu_start"))
        self.btn_mag_start.setText(self.tr("btn_mag_start"))

    def disconnect_device(self):
        self.write_log("Disconnecting device...")

        if self.mock_mode:
            if self.mock_task:
                self.mock_task.cancel()
                self.mock_task = None
            self.handle = None
            self.btn_connect.setEnabled(True)
            self.btn_disconnect.setEnabled(False)
            self.set_connection_controls_enabled(True)
            self.set_device_controls_enabled(False)
            self.lbl_connection_status.setText(self.tr("status_disconnected"))
            self.lbl_connection_status.setStyleSheet("color: #ff5555; font-weight: bold;")
            self.write_log("Disconnected.")
            return

        asyncio.create_task(self.async_disconnect())

    async def async_disconnect(self):
        if self.handle is None:
            return
        try:
            for stream in list(self.running_streams.keys()):
                await self.disable_stream_by_name(stream)

            self.handle = None
            self.btn_connect.setEnabled(True)
            self.btn_disconnect.setEnabled(False)
            self.set_connection_controls_enabled(True)
            self.set_device_controls_enabled(False)
            self.lbl_connection_status.setText(self.tr("status_disconnected"))
            self.lbl_connection_status.setStyleSheet("color: #ff5555; font-weight: bold;")
            self.write_log("Disconnected successfully.")
        except Exception as e:
            self.write_log(f"Error during disconnect: {e}")

    async def disable_stream_by_name(self, name):
        if self.handle is None:
            return
        try:
            if name == "battery":
                await sdk.unsubscribe_adc_battery_status(self.handle)
            elif name == "eeg":
                await sdk.set_eeg_transfer(self.handle, False)
                await sdk.unsubscribe_eeg_data(self.handle)
                self.btn_eeg_start.setText(self.tr("btn_eeg_start"))
            elif name == "ces":
                await sdk.set_adc_transfer(self.handle, False)
                await sdk.unsubscribe_ces_stream(self.handle)
                self.btn_ces_start.setText(self.tr("btn_ces_start"))
            elif name == "imu":
                await sdk.set_imu_transfer(self.handle, False)
                await sdk.unsubscribe_imu_data(self.handle)
                self.btn_imu_start.setText(self.tr("btn_imu_start"))
            elif name == "mag":
                await sdk.set_mag_transfer(self.handle, False)
                await sdk.unsubscribe_mag_data(self.handle)
                self.btn_mag_start.setText(self.tr("btn_mag_start"))

            self.running_streams.pop(name, None)
        except Exception as e:
            self.write_log(f"Unsubscribe {name} failed: {e}")

    # ==============================================================================
    # Configuration / Settings Triggers
    # ==============================================================================
    def toggle_sensor_transfer(self, name):
        if self.handle is None:
            self.write_log("Device not connected!")
            return

        is_active = name in self.running_streams
        if is_active:
            self.write_log(f"Stopping {name.upper()} stream...")
            if self.mock_mode:
                self.running_streams.pop(name, None)
                if name == "eeg": self.btn_eeg_start.setText(self.tr("btn_eeg_start"))
                elif name == "ces": self.btn_ces_start.setText(self.tr("btn_ces_start"))
                elif name == "imu": self.btn_imu_start.setText(self.tr("btn_imu_start"))
                elif name == "mag": self.btn_mag_start.setText(self.tr("btn_mag_start"))
                return
            asyncio.create_task(self.disable_stream_by_name(name))
        else:
            self.write_log(f"Starting {name.upper()} stream...")
            if self.mock_mode:
                self.running_streams[name] = True
                if name == "eeg": self.btn_eeg_start.setText(self.tr("btn_eeg_stop"))
                elif name == "ces": self.btn_ces_start.setText(self.tr("btn_ces_stop"))
                elif name == "imu": self.btn_imu_start.setText(self.tr("btn_imu_stop"))
                elif name == "mag": self.btn_mag_start.setText(self.tr("btn_mag_stop"))
                return
            asyncio.create_task(self.enable_stream_by_name(name))

    async def enable_stream_by_name(self, name):
        try:
            if name == "eeg":
                # Make sure the device is not left in TestSignal/InternalShort mode.
                try:
                    await sdk.set_eeg_signal_type(self.handle, sdk.EegSignalType.Normal)
                    await sdk.set_eeg_signal_types(self.handle, [sdk.EegSignalType.Normal] * 8)
                    await sdk.disable_all_eeg_leadoff_channels(self.handle)
                    await sdk.set_eeg_leadoff_channel_mask(self.handle, 0)
                except Exception as exc:
                    self.write_log(f"Normal EEG signal type setup failed: {exc}")
                await sdk.subscribe_eeg_data(self.handle, 250, self.on_eeg_received)
                await sdk.set_eeg_transfer(self.handle, True)
                self.btn_eeg_start.setText(self.tr("btn_eeg_stop"))
            elif name == "ces":
                await sdk.set_adc_mode(self.handle, 2)
                await sdk.subscribe_ces_stream(self.handle, 1, self.on_ces_received)
                await sdk.set_adc_transfer(self.handle, True)
                self.btn_ces_start.setText(self.tr("btn_ces_stop"))
            elif name == "imu":
                await sdk.subscribe_imu_data(self.handle, 25, self.on_imu_received)
                await sdk.set_imu_transfer(self.handle, True)
                self.btn_imu_start.setText(self.tr("btn_imu_stop"))
            elif name == "mag":
                await sdk.subscribe_mag_data(self.handle, 25, self.on_mag_received)
                await sdk.set_mag_transfer(self.handle, True)
                self.btn_mag_start.setText(self.tr("btn_mag_stop"))

            self.running_streams[name] = True
            self.write_log(f"Stream {name.upper()} successfully started.")
        except Exception as e:
            self.write_log(f"Failed to start {name.upper()} stream: {e}")

    def apply_eeg_config(self):
        if self.handle is None:
            self.write_log("Device not connected!")
            return
        sr_idx = self.cb_eeg_sr.currentIndex()
        gain_idx = self.cb_eeg_gain.currentIndex()

        sr_enums = [
            sdk.EegSampleRate.Hz250,
            sdk.EegSampleRate.Hz500,
            sdk.EegSampleRate.Hz1000,
            sdk.EegSampleRate.Hz2000,
            sdk.EegSampleRate.Hz4000
        ]
        sr_vals = [250, 500, 1000, 2000, 4000]
        gain_enums = [
            sdk.EegGain.Gain1,
            sdk.EegGain.Gain2,
            sdk.EegGain.Gain4,
            sdk.EegGain.Gain6,
            sdk.EegGain.Gain8,
            sdk.EegGain.Gain12,
            sdk.EegGain.Gain24
        ]

        self.write_log(f"Applying EEG Settings: SR={sr_vals[sr_idx]}Hz, Gain={self.cb_eeg_gain.currentText()}")

        if self.mock_mode:
            self.eeg_fs = sr_vals[sr_idx]
            self.eeg_filter = SoftwareFilter(fs=self.eeg_fs)
            self.write_log("Mock configuration applied.")
            return

        async def do_config():
            try:
                was_eeg_active = "eeg" in self.running_streams
                if was_eeg_active:
                    await sdk.set_eeg_transfer(self.handle, False)

                await sdk.set_eeg_sample_rate(self.handle, sr_enums[sr_idx])
                await sdk.set_eeg_gain(self.handle, gain_enums[gain_idx])
                # Force every hardware channel back to real EEG input. Some SDK
                # examples leave per-channel test/short modes behind.
                await sdk.set_eeg_signal_type(self.handle, sdk.EegSignalType.Normal)
                try:
                    await sdk.set_eeg_signal_types(self.handle, [sdk.EegSignalType.Normal] * 8)
                except Exception as exc:
                    self.write_log(f"Per-channel signal type skipped: {exc}")
                try:
                    await sdk.set_eeg_gains(self.handle, [gain_enums[gain_idx]] * 8)
                except Exception as exc:
                    self.write_log(f"Per-channel gain skipped: {exc}")
                # Critical for dry caps: lead-off test current pins inactive inputs
                # to the ADC negative rail. Disable it for normal EEG streaming.
                try:
                    await sdk.disable_all_eeg_leadoff_channels(self.handle)
                    await sdk.set_eeg_leadoff_channel_mask(self.handle, 0)
                except Exception as exc:
                    self.write_log(f"Leadoff disable skipped: {exc}")

                self.eeg_fs = sr_vals[sr_idx]
                self.eeg_filter = SoftwareFilter(fs=self.eeg_fs)

                if was_eeg_active:
                    await sdk.set_eeg_transfer(self.handle, True)
                self.write_log("EEG Configuration successfully applied.")
            except Exception as e:
                self.write_log(f"Apply EEG config failed: {e}")
                QtWidgets.QMessageBox.warning(
                    self,
                    "Apply EEG Settings Failed",
                    f"Failed to apply EEG settings:\n\n{e}"
                )

        asyncio.create_task(do_config())

    # ==============================================================================
    # EDF File Recording Controls
    # ==============================================================================
    def start_recording(self):
        if self.handle is None:
            self.write_log("Device not connected!")
            return
        filename = self.txt_edf_filename.text().strip()
        if not filename:
            filename = "recording.edf"

        self.write_log(f"Starting EDF/BDF+ Recording: {filename}...")
        self.edf_recorder = EDFRecorder(self.handle, self.mock_mode)
        channel_config = self.current_channel_config()
        self.persist_channel_config(quiet=True)

        async def do_start():
            try:
                await self.edf_recorder.start(filename, channel_config)
                self.lbl_recording_info.setText(self.tr("edf_active", file=filename))
                self.lbl_recording_info.setStyleSheet("color: #00ffcc; font-weight: bold;")
                self.btn_edf_start.setEnabled(False)
                self.btn_edf_stop.setEnabled(True)
                self.write_log("EDF/BDF+ Recording successfully started.")
                self.recording_start_time = time.time()

                # Active background polling loop for physical markers if TriggerBox is connected
                if self.trb_device and self.trb_polling_task is None:
                    self.trb_polling_task = asyncio.create_task(self.trb_sync_polling_loop())
            except Exception as e:
                self.write_log(f"Start recording failed: {e}")
        asyncio.create_task(do_start())

    def stop_recording(self):
        self.write_log("Stopping EDF/BDF+ Recording...")
        if not self.edf_recorder:
            return

        async def do_stop():
            try:
                # Safely terminate background physical polling loop
                if self.trb_polling_task is not None:
                    self.trb_polling_task.cancel()
                    self.trb_polling_task = None

                await self.edf_recorder.stop()
                self.lbl_recording_info.setText(self.tr("edf_inactive"))
                self.lbl_recording_info.setStyleSheet("color: #888899; font-style: italic;")
                self.btn_edf_start.setEnabled(True)
                self.btn_edf_stop.setEnabled(False)
                self.write_log("EDF/BDF+ Recording stopped. File finalized.")
            except Exception as e:
                self.write_log(f"Stop recording failed: {e}")
        asyncio.create_task(do_stop())

    def send_annotation(self):
        txt = self.txt_annotation.text().strip()
        if not txt:
            return
        self.write_log(f"Sending annotation: '{txt}'")

        # Log to Marker history
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.list_marker_history.addItem(f"[{ts}] Annotation: {txt}")
        if self.list_marker_history.count() > 10:
            self.list_marker_history.takeItem(0)
        self.list_marker_history.scrollToBottom()

        self.inject_marker(txt)
        self.txt_annotation.clear()

    def refresh_device_status(self):
        if self.handle is None:
            return

        async def do_refresh():
            self.btn_refresh_status.setEnabled(False)
            try:
                if self.mock_mode:
                    self.card_fw.set_value("BrainSync (Mock V1.0)")
                    self.card_bat.set_value("3700 mV / 25.0 °C")
                    self.lbl_battery.setText("Battery: 85 %")
                    self.write_log("Mock device status refreshed successfully.")
                    return

                # 1. Try to get FW version
                try:
                    fw = await sdk.get_firmware_version(self.handle)
                    if isinstance(fw, dict):
                        dev_type = fw.get("device_type", "BrainSync")
                        sw = fw.get("sw_version", "N/A")
                        hw = fw.get("hw_version", "N/A")
                        fw_str = f"{dev_type} (SW: {sw}, HW: {hw})"
                    else:
                        fw_str = str(fw)
                    self.card_fw.set_value(fw_str)
                except Exception as ex:
                    self.write_log(f"Warning: Failed to get firmware version: {ex}")

                # 2. Try to get battery status
                try:
                    bat = await sdk.get_battery_status(self.handle)
                    self.on_battery_received([bat])
                    self.write_log("Device status refreshed successfully.")
                except Exception as ex:
                    self.write_log(f"Warning: Failed to get battery status: {ex}")
            except Exception as e:
                self.write_log(f"Failed to refresh status: {e}")
            finally:
                self.btn_refresh_status.setEnabled(True)

        asyncio.create_task(do_refresh())

    # ==============================================================================
    # EEG, CES, IMU, Mag Callback data handlers
    # ==============================================================================
    def on_battery_received(self, packets):
        if not packets: return
        p = packets[-1]
        self.lbl_battery.setText(f"Battery: {p.bat_percent} %")
        self.card_bat.set_value(f"{p.bat_voltage_mv} mV / {p.ntc_temp_c / 10.0} °C")

        # Align with Protocol V2 and support translations via i18n
        charge_key = f"charge_state_{p.charge_state}"
        charge_str = self.tr(charge_key)
        if charge_str == charge_key:
            charge_str = self.tr("charge_state_unknown", state=p.charge_state)
        self.card_charge.set_value(charge_str)

    def on_eeg_received(self, packets):
        self.total_eeg_pkts += len(packets)

        # Calculate EEG packet loss using seq_num (u24/u32)
        for p in packets:
            if hasattr(p, "seq_num"):
                if self.last_eeg_seq is not None:
                    diff = (p.seq_num - self.last_eeg_seq) & 0xFFFFFF
                    if diff > 1:
                        self.lost_eeg_pkts += (diff - 1)
                self.last_eeg_seq = p.seq_num

        # Get current gain for converting to microvolts
        try:
            gain_idx = self.cb_eeg_gain.currentIndex()
            gain_enums = [
                sdk.EegGain.Gain1, sdk.EegGain.Gain2, sdk.EegGain.Gain4,
                sdk.EegGain.Gain6, sdk.EegGain.Gain8, sdk.EegGain.Gain12, sdk.EegGain.Gain24
            ]
            current_gain = gain_enums[gain_idx]
        except Exception:
            current_gain = sdk.EegGain.Gain24

        # BrainSync EEG is displayed and exported exclusively in microvolts (µV).
        # Never fall back to raw adc_values: raw ADC would look flat for normal EEG
        # and only become visible on movement artifacts.
        converted_samples = []
        for p in packets:
            if not hasattr(p, "to_microvolts"):
                print("EEG packet has no to_microvolts(); skipping packet", flush=True)
                self.eeg_conversion_errors += 1
                continue
            try:
                converted_samples.append(p.to_microvolts(current_gain))
            except Exception as e:
                print(f"EEG to_microvolts conversion error: {e}", flush=True)
                self.eeg_conversion_errors += 1
                continue

        # Forward samples to LSL EEG Outlet
        if self.lsl_eeg_outlet is not None:
            try:
                for mv in converted_samples:
                    self.lsl_eeg_outlet.push_sample(mv)
            except Exception as e:
                self.write_log(f"LSL EEG Push failed: {e}")

        for mv in converted_samples:
            for ch in range(8):
                val = mv[ch]
                self.eeg_buffers[ch].push(val)

    def on_ces_received(self, packets):
        self.total_ces_pkts += len(packets)

        # Calculate CES packet loss using seq (u16)
        for p in packets:
            if hasattr(p, "seq"):
                if self.last_ces_seq is not None:
                    diff = (p.seq - self.last_ces_seq) & 0xFFFF
                    if diff > 1:
                        self.lost_ces_pkts += (diff - 1)
                self.last_ces_seq = p.seq

        for p in packets:
            n_samples = len(p.samples)
            ces1_data = []
            ces2_data = []
            for i in range(0, n_samples, 2):
                if i + 1 < n_samples:
                    ces1_data.append(p.samples[i])
                    ces2_data.append(p.samples[i+1])
            if ces1_data:
                self.ces_buffers[0].push_many(ces1_data)
                self.ces_buffers[1].push_many(ces2_data)

    def on_imu_received(self, packets):
        for p in packets:
            self.imu_acc_buffers[0].push(p.accel_x)
            self.imu_acc_buffers[1].push(p.accel_y)
            self.imu_acc_buffers[2].push(p.accel_z)
            self.imu_gyro_buffers[0].push(p.gyro_x)
            self.imu_gyro_buffers[1].push(p.gyro_y)
            self.imu_gyro_buffers[2].push(p.gyro_z)

    def on_mag_received(self, packets):
        for p in packets:
            self.mag_buffers[0].push(p.mag_x)
            self.mag_buffers[1].push(p.mag_y)
            self.mag_buffers[2].push(p.mag_z)

    # ==============================================================================
    # Stimulation Controls
    # ==============================================================================
    def apply_stim_params(self):
        if self.handle is None:
            self.write_log("Device not connected!")
            return

        chan = self.cb_stim_chan.currentIndex()
        mode = self.cb_stim_mode.currentIndex()
        freq = self.spin_stim_freq.value()
        current = self.spin_stim_current.value()
        ramp_up = self.spin_ramp_up.value()
        hold = self.spin_hold.value()
        ramp_down = self.spin_ramp_down.value()
        label_id = self.spin_stim_label.value()

        freq_hz_10x = int(freq * 10)

        self.write_log(f"Applying Stimulation Params on CH{chan}: Mode={self.cb_stim_mode.currentText()}, Freq={freq}Hz, Current={current}uA")

        if self.mock_mode:
            self.write_log("Stimulation configuration mock-applied successfully.")
            self.btn_arm.setEnabled(True)
            return

        async def do_apply():
            try:
                await sdk.set_stimulation_params(
                    self.handle,
                    chan,
                    mode,
                    freq_hz_10x,
                    current,
                    ramp_up,
                    hold,
                    ramp_down,
                    label_id
                )
                self.write_log("Stimulation parameters successfully configured.")
                self.btn_arm.setEnabled(True)
            except Exception as e:
                self.write_log(f"Stim configuration failed: {e}")
        asyncio.create_task(do_apply())

    def arm_stimulation(self, checked=False):
        if self.handle is None: return
        self.write_log("Arming stimulation circuitry...")

        if self.mock_mode:
            self.update_stim_state("stim_state_armed")
            self.lbl_stim_state.setStyleSheet("font-weight: bold; color: #ff9900;")
            self.btn_arm.setEnabled(False)
            self.btn_disarm.setEnabled(True)
            self.btn_stim_start.setEnabled(True)
            self.write_log("Stimulation armed.")
            return

        async def do_arm():
            try:
                await sdk.set_stimulation_arm(self.handle, True)
                self.update_stim_state("stim_state_armed")
                self.lbl_stim_state.setStyleSheet("font-weight: bold; color: #ff9900;")
                self.btn_arm.setEnabled(False)
                self.btn_disarm.setEnabled(True)
                self.btn_stim_start.setEnabled(True)
                self.write_log("Stimulation armed successfully.")
            except Exception as e:
                self.write_log(f"Arm failed: {e}")
        asyncio.create_task(do_arm())

    def disarm_stimulation(self, checked=False):
        if self.handle is None: return
        self.write_log("Disarming stimulation circuitry...")

        if self.mock_mode:
            self.update_stim_state("stim_state_disarmed")
            self.lbl_stim_state.setStyleSheet("font-weight: bold; color: #e74c3c;")
            self.btn_arm.setEnabled(True)
            self.btn_disarm.setEnabled(False)
            self.btn_stim_start.setEnabled(False)
            self.btn_stim_stop.setEnabled(False)
            self.write_log("Stimulation disarmed.")
            return

        async def do_disarm():
            try:
                await sdk.set_stimulation_arm(self.handle, False)
                self.update_stim_state("stim_state_disarmed")
                self.lbl_stim_state.setStyleSheet("font-weight: bold; color: #e74c3c;")
                self.btn_arm.setEnabled(True)
                self.btn_disarm.setEnabled(False)
                self.btn_stim_start.setEnabled(False)
                self.btn_stim_stop.setEnabled(False)
                self.write_log("Stimulation disarmed successfully.")
            except Exception as e:
                self.write_log(f"Disarm failed: {e}")
        asyncio.create_task(do_disarm())

    def start_stimulation_output(self, checked=False):
        if self.handle is None: return
        self.write_log("Starting active stimulation output...")

        if self.mock_mode:
            self.update_stim_state("stim_state_active")
            self.lbl_stim_state.setStyleSheet("font-weight: bold; color: #00ffcc;")
            self.btn_stim_start.setEnabled(False)
            self.btn_stim_stop.setEnabled(True)
            self.btn_disarm.setEnabled(False)
            self.write_log("Active stimulation output running.")
            return

        async def do_start():
            try:
                await sdk.stimulation_go(self.handle, 0x03, 0x01)
                self.update_stim_state("stim_state_active")
                self.lbl_stim_state.setStyleSheet("font-weight: bold; color: #00ffcc;")
                self.btn_stim_start.setEnabled(False)
                self.btn_stim_stop.setEnabled(True)
                self.btn_disarm.setEnabled(False)
                self.write_log("Stimulation output successfully started.")
            except Exception as e:
                self.write_log(f"Start stim output failed: {e}")
        asyncio.create_task(do_start())

    def stop_stimulation_output(self, checked=False):
        if self.handle is None: return
        self.write_log("Stopping stimulation output...")

        if self.mock_mode:
            self.update_stim_state("stim_state_armed")
            self.lbl_stim_state.setStyleSheet("font-weight: bold; color: #ff9900;")
            self.btn_stim_start.setEnabled(True)
            self.btn_stim_stop.setEnabled(False)
            self.btn_disarm.setEnabled(True)
            self.write_log("Stimulation output stopped.")
            return

        async def do_stop():
            try:
                await sdk.stimulation_go(self.handle, 0x03, 0x00)
                self.update_stim_state("stim_state_armed")
                self.lbl_stim_state.setStyleSheet("font-weight: bold; color: #ff9900;")
                self.btn_stim_start.setEnabled(True)
                self.btn_stim_stop.setEnabled(False)
                self.btn_disarm.setEnabled(True)
                self.write_log("Stimulation output successfully stopped.")
            except Exception as e:
                self.write_log(f"Stop stim output failed: {e}")
        asyncio.create_task(do_stop())

    # ==============================================================================
    # Contact Test (Electrode Impedance Test)
    # ==============================================================================
    def start_contact_test(self):
        if self.handle is None: return

        freq_sel = [100, 200, 300, 1000]
        freq_val = freq_sel[self.cb_ct_freq.currentIndex()]
        current = self.spin_ct_current.value()

        mask_sel = [0x01, 0x02, 0x03]
        chan_mask = mask_sel[self.cb_ct_mask.currentIndex()]
        duration = self.spin_ct_duration.value() * 1000

        self.write_log(f"Starting Electrode Test: Mask={self.cb_ct_mask.currentText()}, Current={current}uA")

        self.btn_ct_start.setEnabled(False)
        self.btn_ct_stop.setEnabled(True)
        self.update_ct_state("ct_status_running", pct=0)
        self.ct_progress.setValue(0)

        if self.mock_mode:
            asyncio.create_task(self.run_mock_contact_test(self.spin_ct_duration.value()))
            return

        async def do_test():
            try:
                await sdk.subscribe_adc_contact_test_status(self.handle, 1, self.on_ct_status)
                await sdk.subscribe_adc_contact_test_progress(self.handle, 1, self.on_ct_progress)
                await sdk.subscribe_adc_contact_test_measurement(self.handle, 1, self.on_ct_measurement)

                await sdk.configure_contact_test(
                    self.handle,
                    1,
                    0,
                    chan_mask,
                    freq_val,
                    current,
                    duration,
                    1
                )
                self.write_log("Electrode test command sent.")
            except Exception as e:
                self.write_log(f"Start contact test failed: {e}")
                self.btn_ct_start.setEnabled(True)
                self.btn_ct_stop.setEnabled(False)
        asyncio.create_task(do_test())

    def stop_contact_test(self):
        if self.handle is None: return
        self.write_log("Requesting diagnostics abort...")

        if self.mock_mode:
            self.update_ct_state("ct_status_aborted")
            self.btn_ct_start.setEnabled(True)
            self.btn_ct_stop.setEnabled(False)
            return

        async def do_stop():
            try:
                await sdk.configure_contact_test(self.handle, 0, 0, 0, 0, 0, 0, 0)

                await sdk.unsubscribe_adc_contact_test_status(self.handle)
                await sdk.unsubscribe_adc_contact_test_progress(self.handle)
                await sdk.unsubscribe_adc_contact_test_measurement(self.handle)

                self.btn_ct_start.setEnabled(True)
                self.btn_ct_stop.setEnabled(False)
                self.update_ct_state("ct_status_aborted")
                self.write_log("Electrode test aborted.")
            except Exception as e:
                self.write_log(f"Abort contact test failed: {e}")
        asyncio.create_task(do_stop())

    def on_ct_status(self, packets):
        if not packets: return
        p = packets[-1]
        self.update_ct_state("raw_status", raw_text=f"Status: Diagnostic state = {p.state}, Channel mask bits = {p.chan_bits}")

    def on_ct_progress(self, packets):
        if not packets: return
        p = packets[-1]
        self.ct_progress.setValue(p.progress_percent)
        self.update_ct_state("ct_status_running", pct=p.progress_percent)

    def on_ct_measurement(self, packets):
        if not packets: return
        p = packets[-1]

        r1 = p.ces1_resistance_ohm
        r2 = p.ces2_resistance_ohm

        self.card_ces1_res.set_value(f"{r1:,} Ω")
        self.card_ces2_res.set_value(f"{r2:,} Ω")

        def get_color_for_res(r):
            if r > 100000: return "#ff3333"
            elif r > 50000: return "#ffcc00"
            else: return "#00ffcc"

        self.card_ces1_res.lbl_val.setStyleSheet(f"color: {get_color_for_res(r1)}; font-size: 20px; font-weight: bold;")
        self.card_ces2_res.lbl_val.setStyleSheet(f"color: {get_color_for_res(r2)}; font-size: 20px; font-weight: bold;")

        self.write_log(f"Final Contact Resistances: CES1={r1} Ω, CES2={r2} Ω")

        self.btn_ct_start.setEnabled(True)
        self.btn_ct_stop.setEnabled(False)
        self.update_ct_state("ct_status_done")

        asyncio.create_task(self.cleanup_contact_test_subs())

    async def cleanup_contact_test_subs(self):
        try:
            await sdk.unsubscribe_adc_contact_test_status(self.handle)
            await sdk.unsubscribe_adc_contact_test_progress(self.handle)
            await sdk.unsubscribe_adc_contact_test_measurement(self.handle)
        except Exception:
            pass

    # ==============================================================================
    # Mock Mode Loops
    # ==============================================================================
    async def run_mock_stream(self):
        """Simulates device streaming data packets for offline testing"""
        t = 0.0
        while True:
            await asyncio.sleep(0.04)
            t += 0.04

            # 1. Mock EEG (250Hz, 10 samples per 0.04s)
            if "eeg" in self.running_streams:
                eeg_packets = []
                for _ in range(10):
                    ch_vals = []
                    for ch in range(8):
                        alpha = 40.0 * math.sin(2 * math.pi * 10.0 * t + ch)
                        noise50 = 15.0 * math.sin(2 * math.pi * 50.0 * t)
                        rand = np.random.normal(0, 4)
                        ch_vals.append(alpha + noise50 + rand)

                    eeg_packets.append(MockEegPacket(ch_vals))

                self.on_eeg_received(eeg_packets)

            # 2. Mock CES High speed ADC
            if "ces" in self.running_streams:
                ces_packets = []
                samples = []
                for _ in range(25):
                    v1 = int(2048 + 500 * math.sin(2 * math.pi * 10.0 * t))
                    v2 = int(2048 + 300 * math.cos(2 * math.pi * 25.0 * t))
                    samples.extend([v1, v2])

                class MockCesPacket:
                    def __init__(self, s):
                        self.samples = s
                ces_packets.append(MockCesPacket(samples))
                self.on_ces_received(ces_packets)

            # 3. Mock IMU (25Hz, 1 sample per 0.04s)
            if "imu" in self.running_streams:
                class MockImuPacket:
                    def __init__(self, t):
                        self.accel_x = 0.05 * math.sin(t)
                        self.accel_y = 0.02 * math.cos(t)
                        self.accel_z = 0.98 + 0.01 * math.sin(t * 2)
                        self.gyro_x = 2.0 * math.sin(t * 1.5)
                        self.gyro_y = 0.5 * math.cos(t)
                        self.gyro_z = 0.1 * math.sin(t * 3)
                self.on_imu_received([MockImuPacket(t)])

            # 4. Mock Magnetometer
            if "mag" in self.running_streams:
                class MockMagPacket:
                    def __init__(self, t):
                        self.mag_x = 0.35 * math.sin(t / 2)
                        self.mag_y = 0.12 * math.cos(t / 3)
                        self.mag_z = -0.45 + 0.05 * math.sin(t * 0.8)
                self.on_mag_received([MockMagPacket(t)])

            # 5. Mock Battery stats update
            if t % 5.0 < 0.04:
                class MockBatteryPacket:
                    def __init__(self):
                        self.bat_percent = 88
                        self.bat_voltage_mv = 3985
                        self.ntc_temp_c = 285
                        self.charge_state = 4  # Charging (V2 protocol)
                self.on_battery_received([MockBatteryPacket()])

    async def run_mock_contact_test(self, duration_sec):
        steps = 10
        sleep_interval = duration_sec / steps
        for i in range(1, steps + 1):
            await asyncio.sleep(sleep_interval)
            pct = int((i / steps) * 100)
            self.ct_progress.setValue(pct)
            self.update_ct_state("ct_status_running", pct=pct)

        class MockMeasurementPacket:
            def __init__(self):
                self.ces1_resistance_ohm = 4250
                self.ces2_resistance_ohm = 145200
        self.on_ct_measurement([MockMeasurementPacket()])

    # ==============================================================================
    # TriggerBox Synchronization & Marker Injection
    # ==============================================================================
    def update_trb_indicator(self):
        if not _TRB_AVAILABLE:
            self.lbl_trb_status.setText(self.tr("trb_status_uninstalled"))
            self.lbl_trb_status.setStyleSheet("color: #7f8c8d; font-style: italic;")
            self.btn_trb_config.setEnabled(False)
        else:
            status_str = "Connected" if self.trb_device is not None else "Disconnected"
            color_str = "#2ecc71" if self.trb_device is not None else "#7f8c8d"
            self.lbl_trb_status.setText(self.tr("trb_status_lbl", status=status_str))
            self.lbl_trb_status.setStyleSheet(f"color: {color_str}; font-weight: bold;")
            self.btn_trb_config.setEnabled(True)

    def show_triggerbox_dialog(self):
        dialog = TriggerBoxDialog(self, self.mock_mode)
        dialog.exec()
        self.update_trb_indicator()

    async def trb_sync_polling_loop(self):
        self.write_log("[TRB] TriggerBox physical synchronization marking loop activated.")
        prev_status = None
        start_time = time.time()

        while self.trb_device and self.edf_recorder and self.edf_recorder.recording_active:
            try:
                if self.mock_mode:
                    await asyncio.sleep(0.1)
                    continue

                # Poll physical TriggerBox status
                status = await self.trb_device.trigger().get_status()
                rel_time = time.time() - start_time

                if prev_status is not None:
                    # Rising edge checks
                    if status.pd_active and not prev_status.pd_active:
                        tag = self.trb_aliases.get("PD", "PD_Onset")
                        self.inject_marker(tag, rel_time)
                    if status.aud_active and not prev_status.aud_active:
                        tag = self.trb_aliases.get("AUD", "AUD_Onset")
                        self.inject_marker(tag, rel_time)
                    if status.mic_active and not prev_status.mic_active:
                        tag = self.trb_aliases.get("MIC", "MIC_Onset")
                        self.inject_marker(tag, rel_time)
                    if status.btn_active and not prev_status.btn_active:
                        tag = self.trb_aliases.get("BTN", "BTN_Press")
                        self.inject_marker(tag, rel_time)

                prev_status = status
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.write_log(f"[TRB] Synchronization loop warning: {e}")
            await asyncio.sleep(0.05)

        self.write_log("[TRB] TriggerBox synchronization loop stopped.")

    def inject_marker(self, tag, rel_time=None):
        if rel_time is None:
            rel_time = time.time() - (self.recording_start_time or time.time())

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.list_marker_history.addItem(f"[{ts}] Annotation: {tag}")
        if self.list_marker_history.count() > 10:
            self.list_marker_history.takeItem(0)
        self.list_marker_history.scrollToBottom()

        if self.lsl_marker_outlet is not None:
            try:
                self.lsl_marker_outlet.push_sample([tag])
            except Exception as e:
                self.write_log(f"LSL Marker push failed: {e}")

        if self.edf_recorder and self.edf_recorder.recording_active:
            async def add_ann():
                try:
                    await self.edf_recorder.add_annotation(rel_time, 0.0, tag)
                    self.write_log(f"Annotation recorded to EDF/BDF+: '{tag}' at {rel_time:.3f}s")
                except Exception as ex:
                    self.write_log(f"Failed to record annotation: {ex}")
            asyncio.create_task(add_ann())
        else:
            self.write_log(f"Event marked: '{tag}'")


    def update_eeg_plot_ticks(self, val=None):
        if val is not None:
            self.eeg_offset_gap = val
        else:
            self.eeg_offset_gap = self.spin_eeg_offset.value()

        # 1. Update Y-axis range
        self.eeg_plot_widget.setYRange(-self.eeg_offset_gap, self.eeg_offset_gap * 8)

        # 2. Update Ticks label based on checked channels
        ay = self.eeg_plot_widget.getAxis('left')
        ticks = []
        for i in range(8):
            if not hasattr(self, "chk_channels") or not self.chk_channels or self.chk_channels[i].isChecked():
                baseline = (7 - i) * self.eeg_offset_gap
                ticks.append((baseline, self.eeg_ch_names[i]))
        ay.setTicks([ticks])

    def update_channel_visibility(self):
        self.channel_config = with_active_mask(self.channel_config, self.current_active_mask())
        # Refresh left axis ticks and line curves visibility
        ticks = []
        for i in range(8):
            visible = self.chk_channels[i].isChecked()
            self.eeg_curves[i].setVisible(visible)
            if visible:
                baseline = (7 - i) * self.eeg_offset_gap
                ticks.append((baseline, self.eeg_ch_names[i]))
        ay = self.eeg_plot_widget.getAxis('left')
        ay.setTicks([ticks])
        self.persist_channel_config()


    # ==============================================================================
    # Drawing & Plot Updates (Called on timer)
    # ==============================================================================
    def update_plots(self):
        # 1. Update EEG channels (apply software filtering if needed)
        if "eeg" in self.running_streams:
            offset_gap = self.eeg_offset_gap
            for ch in range(8):
                raw = self.eeg_buffers[ch].get()
                if len(raw) > 0:
                    filtered = self.eeg_filter.process(raw)
                    baseline = (7 - ch) * offset_gap
                    self.eeg_curves[ch].setData(filtered + baseline)
                    
                    if ch == 0:
                        if not hasattr(self, "_eeg_debug_counter"):
                            self._eeg_debug_counter = 0
                        self._eeg_debug_counter += 1
                        if self._eeg_debug_counter % 50 == 0:
                            print(f"[DEBUG EEG GUI] total_pkts={self.total_eeg_pkts}, lost={self.lost_eeg_pkts}, ch0_val_range={np.min(raw):.1f} to {np.max(raw):.1f}")

            self.lbl_eeg_pkts.setText("EEG " + self.tr("total_pkts", count=self.total_eeg_pkts))
            self.lbl_eeg_loss.setText("EEG " + self.tr("loss_rate", rate=(self.lost_eeg_pkts / max(1, self.total_eeg_pkts) * 100.0)))

        # 2. Update CES Waveforms
        if "ces" in self.running_streams:
            for ch in range(2):
                data = self.ces_buffers[ch].get()
                if len(data) > 0:
                    self.ces_curves[ch].setData(data)
            self.lbl_ces_pkts.setText("CES " + self.tr("total_pkts", count=self.total_ces_pkts))
            self.lbl_ces_loss.setText("CES " + self.tr("loss_rate", rate=(self.lost_ces_pkts / max(1, self.total_ces_pkts) * 100.0)))

        # 3. Update IMU plots
        if "imu" in self.running_streams:
            for idx in range(3):
                acc = self.imu_acc_buffers[idx].get()
                gyro = self.imu_gyro_buffers[idx].get()
                if len(acc) > 0:
                    self.acc_curves[idx].setData(acc)
                if len(gyro) > 0:
                    self.gyro_curves[idx].setData(gyro)

        # 4. Update Mag plots
        if "mag" in self.running_streams:
            for idx in range(3):
                mag = self.mag_buffers[idx].get()
                if len(mag) > 0:
                    self.mag_curves[idx].setData(mag)
