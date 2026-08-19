# -*- coding: utf-8 -*-
"""
Montage editor for BrainSync EEG/tES/Trigger Hub physical wiring.
"""

from PySide6 import QtCore, QtWidgets

import brainsync_sdk as sdk


STANDARD_ELECTRODES = [
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
    "A1", "A2", "M1", "M2"
]


class MontageDialog(QtWidgets.QDialog):
    def __init__(self, parent, channel_config, trigger_hub):
        super().__init__(parent)
        self.channel_config = channel_config
        self.trigger_hub = trigger_hub
        self.setWindowTitle("BrainSync Montage Configuration")
        self.setMinimumWidth(760)
        self.setMinimumHeight(560)
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #08080c;
            }
            QTabWidget::pane {
                border: 1px solid #23253b;
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
            QTableWidget {
                background-color: #0b0c13;
                border: 1px solid #23253b;
                gridline-color: #1a1a26;
                color: #ffffff;
                border-radius: 6px;
            }
            QTableWidget::item {
                color: #ffffff;
                background-color: #0b0c13;
                padding: 6px;
            }
            QTableWidget::item:selected {
                background-color: #1c1d2e;
                color: #00ffcc;
            }
            QHeaderView::section {
                background-color: #121424;
                color: #8f9bb3;
                padding: 6px;
                border: 1px solid #23253b;
                font-weight: bold;
            }
            QHeaderView {
                background-color: #121424;
            }
            QTableCornerButton::section {
                background-color: #121424;
                border: 1px solid #23253b;
            }
            QLabel {
                color: #e2e8f0;
                font-weight: bold;
                font-size: 12px;
            }
            QLineEdit {
                background-color: #121321;
                border: 1px solid #2a2d48;
                border-radius: 5px;
                color: #ffffff;
                padding: 6px;
            }
            QLineEdit:focus {
                border-color: #00ffcc;
            }
            QTableWidget QLineEdit {
                background-color: #121321;
                border: 1px solid #00ffcc;
                border-radius: 0px;
                padding: 2px;
                margin: 0px;
                color: #ffffff;
            }
            QComboBox {
                background-color: #121321;
                border: 1px solid #2a2d48;
                border-radius: 5px;
                color: #ffffff;
                padding: 2px 24px 2px 6px;
                min-height: 24px;
            }
            QComboBox QLineEdit {
                background: transparent;
                border: none;
                padding: 0px;
                margin: 0px;
                color: #ffffff;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 18px;
                border-left: 1px solid #2a2d48;
                border-top-right-radius: 5px;
                border-bottom-right-radius: 5px;
            }
            QComboBox::down-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #8f9bb3;
                width: 0px;
                height: 0px;
            }
            QComboBox::down-arrow:hover {
                border-top-color: #00ffcc;
            }
            QComboBox QAbstractItemView {
                background-color: #121321;
                color: #ffffff;
                selection-background-color: #272846;
                selection-color: #00ffcc;
                border: 1px solid #2a2d48;
            }
            QPushButton {
                background-color: #1a1b2e;
                border: 1px solid #363757;
                border-radius: 5px;
                color: #e2e8f0;
                padding: 8px 20px;
                font-weight: bold;
                font-size: 12px;
                min-width: 90px;
            }
            QPushButton:hover {
                background-color: #272846;
                border-color: #00ffcc;
            }
            QPushButton:pressed {
                background-color: #0f101f;
            }
            QScrollBar:vertical {
                background: #0b0c13;
                width: 10px;
                border: none;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #23253b;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
            QScrollBar:horizontal {
                background: #0b0c13;
                height: 10px;
                border: none;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #23253b;
                min-width: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                border: none;
                background: none;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self.setup_eeg_tab()
        self.setup_tes_tab()
        self.setup_trigger_tab()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def setup_eeg_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.eeg_table = QtWidgets.QTableWidget(8, 3)
        self.eeg_table.setHorizontalHeaderLabels(["Active", "Index", "Label"])
        self.eeg_table.verticalHeader().setDefaultSectionSize(32)
        self.eeg_table.setShowGrid(True)
        
        self.eeg_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.eeg_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.eeg_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)

        labels = list(self.channel_config.labels)
        active_mask = self.channel_config.active_mask
        for row in range(8):
            active = QtWidgets.QTableWidgetItem()
            active.setCheckState(
                QtCore.Qt.CheckState.Checked
                if active_mask & (1 << row)
                else QtCore.Qt.CheckState.Unchecked
            )
            active.setTextAlignment(QtCore.Qt.AlignCenter)
            self.eeg_table.setItem(row, 0, active)
            
            index = QtWidgets.QTableWidgetItem(str(row))
            index.setFlags(index.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            index.setTextAlignment(QtCore.Qt.AlignCenter)
            self.eeg_table.setItem(row, 1, index)
            
            combo = QtWidgets.QComboBox()
            combo.setEditable(False)
            combo.addItems(STANDARD_ELECTRODES)
            curr_label = labels[row] if row < len(labels) else ""
            if curr_label and curr_label not in STANDARD_ELECTRODES:
                combo.addItem(curr_label)
            combo.setCurrentText(curr_label)
            self.eeg_table.setCellWidget(row, 2, combo)
            
        layout.addWidget(self.eeg_table)

        ref_gnd_layout = QtWidgets.QHBoxLayout()
        ref_gnd_layout.setSpacing(16)
        
        ref_label = QtWidgets.QLabel("REF:")
        self.ref_edit = QtWidgets.QComboBox()
        self.ref_edit.setEditable(False)
        self.ref_edit.addItems(STANDARD_ELECTRODES)
        self.ref_edit.setFixedWidth(120)
        ref_curr = self.channel_config.ref_label or ""
        if ref_curr and ref_curr not in STANDARD_ELECTRODES:
            self.ref_edit.addItem(ref_curr)
        self.ref_edit.setCurrentText(ref_curr)
        
        gnd_label = QtWidgets.QLabel("GND:")
        self.gnd_edit = QtWidgets.QComboBox()
        self.gnd_edit.setEditable(False)
        self.gnd_edit.addItems(STANDARD_ELECTRODES)
        self.gnd_edit.setFixedWidth(120)
        gnd_curr = self.channel_config.gnd_label or ""
        if gnd_curr and gnd_curr not in STANDARD_ELECTRODES:
            self.gnd_edit.addItem(gnd_curr)
        self.gnd_edit.setCurrentText(gnd_curr)
        
        ref_gnd_layout.addWidget(ref_label)
        ref_gnd_layout.addWidget(self.ref_edit)
        ref_gnd_layout.addWidget(gnd_label)
        ref_gnd_layout.addWidget(self.gnd_edit)
        ref_gnd_layout.addStretch()
        
        layout.addLayout(ref_gnd_layout)
        self.tabs.addTab(tab, "EEG")

    def setup_tes_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        self.tes_table = QtWidgets.QTableWidget(4, 4)
        self.tes_table.setHorizontalHeaderLabels(["Slot", "Stim Channel", "Polarity", "Label"])
        self.tes_table.verticalHeader().setDefaultSectionSize(32)
        self.tes_table.setShowGrid(True)
        
        self.tes_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.tes_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tes_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.tes_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)

        assignments = {item.slot: item for item in self.channel_config.stim}
        defaults = [
            (0, 0, "anode", ""),
            (1, 0, "cathode", ""),
            (2, 1, "anode", ""),
            (3, 1, "cathode", ""),
        ]
        for row, (slot, channel, polarity, label) in enumerate(defaults):
            item = assignments.get(slot)
            lbl_val = item.label if item else label
            
            slot_item = QtWidgets.QTableWidgetItem(str(slot))
            slot_item.setFlags(slot_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            slot_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tes_table.setItem(row, 0, slot_item)
            
            chan_item = QtWidgets.QTableWidgetItem(str(channel))
            chan_item.setFlags(chan_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            chan_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tes_table.setItem(row, 1, chan_item)
            
            pol_item = QtWidgets.QTableWidgetItem(polarity)
            pol_item.setFlags(pol_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            pol_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tes_table.setItem(row, 2, pol_item)
            
            combo = QtWidgets.QComboBox()
            combo.setEditable(False)
            combo.addItem("")
            combo.addItems(STANDARD_ELECTRODES)
            if lbl_val and lbl_val not in STANDARD_ELECTRODES:
                combo.addItem(lbl_val)
            combo.setCurrentText(lbl_val if lbl_val else "")
            self.tes_table.setCellWidget(row, 3, combo)
            
        layout.addWidget(self.tes_table)
        self.tabs.addTab(tab, "tES")

    def setup_trigger_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        self.trigger_table = QtWidgets.QTableWidget(4, 4)
        self.trigger_table.setHorizontalHeaderLabels(["Enabled", "Input", "Label", "Threshold mV"])
        self.trigger_table.verticalHeader().setDefaultSectionSize(32)
        self.trigger_table.setShowGrid(True)
        
        self.trigger_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.trigger_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.trigger_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.trigger_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)

        inputs = {item.get("code"): item for item in self.trigger_hub.get("inputs", [])}
        for row, code in enumerate(["PD", "AUD", "MIC", "BTN"]):
            item = inputs.get(code, {})
            enabled = QtWidgets.QTableWidgetItem()
            enabled.setCheckState(
                QtCore.Qt.CheckState.Checked
                if item.get("enabled", code == "BTN")
                else QtCore.Qt.CheckState.Unchecked
            )
            enabled.setTextAlignment(QtCore.Qt.AlignCenter)
            self.trigger_table.setItem(row, 0, enabled)
            
            code_item = QtWidgets.QTableWidgetItem(code)
            code_item.setFlags(code_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            code_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.trigger_table.setItem(row, 1, code_item)
            
            self.trigger_table.setItem(row, 2, QtWidgets.QTableWidgetItem(item.get("label", code)))
            
            threshold = "" if item.get("threshold_mv") is None else str(item.get("threshold_mv"))
            thresh_item = QtWidgets.QTableWidgetItem(threshold)
            thresh_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.trigger_table.setItem(row, 3, thresh_item)
        layout.addWidget(self.trigger_table)
        self.tabs.addTab(tab, "Trigger Hub")

    def build_channel_config(self):
        labels = []
        eeg = []
        active_mask = 0
        for row in range(8):
            combo = self.eeg_table.cellWidget(row, 2)
            label = combo.currentText().strip() if combo else ""
            labels.append(label)
            eeg.append(sdk.ElectrodeAssignment(label, row))
            if self.eeg_table.item(row, 0).checkState() == QtCore.Qt.CheckState.Checked:
                active_mask |= 1 << row

        # Duplicate checking
        labels_seen = set()
        for idx, label in enumerate(labels):
            if not label:
                raise ValueError(f"EEG channel {idx} label cannot be empty.")
            if label in labels_seen:
                raise ValueError(f"Duplicate electrode label '{label}' detected in EEG channels.")
            labels_seen.add(label)

        ref_val = self.ref_edit.currentText().strip()
        gnd_val = self.gnd_edit.currentText().strip()

        if ref_val and ref_val in labels_seen:
            raise ValueError(f"REF label '{ref_val}' conflicts with EEG channel labels.")
        if gnd_val and gnd_val in labels_seen:
            raise ValueError(f"GND label '{gnd_val}' conflicts with EEG channel labels.")
        if ref_val and gnd_val and ref_val == gnd_val:
            raise ValueError("REF and GND labels cannot be the same.")

        stim = []
        for row in range(4):
            slot = int(self.tes_table.item(row, 0).text())
            channel = int(self.tes_table.item(row, 1).text())
            polarity = self.tes_table.item(row, 2).text()

            combo = self.tes_table.cellWidget(row, 3)
            label = combo.currentText().strip() if combo else ""
            if not label:
                # This experiment does not use tES; empty rows are intentionally skipped.
                continue
            stim.append(sdk.StimElectrodeAssignment(label, slot, channel, polarity))

        return sdk.ChannelConfig(
            labels=labels,
            active_mask=active_mask,
            ref_label=ref_val or None,
            gnd_label=gnd_val or None,
            eeg=eeg,
            stim=stim,
        )

    def build_trigger_hub(self):
        inputs = []
        for row in range(4):
            code = self.trigger_table.item(row, 1).text()
            threshold_text = self.trigger_table.item(row, 3).text().strip()
            inputs.append({
                "code": code,
                "label": self.trigger_table.item(row, 2).text().strip() or code,
                "enabled": self.trigger_table.item(row, 0).checkState() == QtCore.Qt.CheckState.Checked,
                "threshold_mv": float(threshold_text) if threshold_text else None,
            })
        return {
            "inputs": inputs,
            "output": {"connector": "BrainSync Trigger Hub 3.5 mm"},
        }

    def accept(self):
        try:
            self.channel_config = self.build_channel_config()
            self.trigger_hub = self.build_trigger_hub()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Montage", str(exc))
            return
        super().accept()
