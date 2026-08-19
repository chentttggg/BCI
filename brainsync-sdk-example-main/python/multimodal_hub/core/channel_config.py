# -*- coding: utf-8 -*-
"""
Physical BrainSync channel configuration helpers for the GUI.
"""

from pathlib import Path

import brainsync_sdk as sdk
import json


DEFAULT_EEG_LABELS = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]
# Guess-number P300 does not use tES.
DEFAULT_STIM_ASSIGNMENTS = []
DEFAULT_TRIGGER_HUB_CONFIG = {
    "inputs": [
        {"code": "PD", "label": "PD_Onset", "enabled": True, "threshold_mv": 1650.0},
        {"code": "AUD", "label": "AUD_Onset", "enabled": True, "threshold_mv": 1500.0},
        {"code": "MIC", "label": "MIC_Onset", "enabled": False, "threshold_mv": 1200.0},
        {"code": "BTN", "label": "BTN_Press", "enabled": True, "threshold_mv": None},
    ],
    "output": {"connector": "BrainSync Trigger Hub 3.5 mm"},
}


def default_channel_config(active_mask: int = 0xFF):
    return sdk.ChannelConfig(
        labels=DEFAULT_EEG_LABELS,
        active_mask=active_mask,
        ref_label="A1",
        gnd_label="A2",
        eeg=[
            sdk.ElectrodeAssignment(label, channel)
            for channel, label in enumerate(DEFAULT_EEG_LABELS)
        ],
        stim=[
            sdk.StimElectrodeAssignment(label, slot, channel, polarity)
            for label, slot, channel, polarity in DEFAULT_STIM_ASSIGNMENTS
        ],
    )


def default_gui_config(active_mask: int = 0xFF):
    return {
        "channel_config": default_channel_config(active_mask),
        "trigger_hub": json.loads(json.dumps(DEFAULT_TRIGGER_HUB_CONFIG)),
    }


def load_gui_config(path: Path):
    if path.exists():
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if "channel_config" in payload:
            channel_payload = payload["channel_config"]
            temp_path = path.with_suffix(".channel.tmp.json")
            temp_path.write_text(json.dumps(channel_payload), encoding="utf-8")
            try:
                channel_config = sdk.ChannelConfig.from_json(str(temp_path))
            finally:
                temp_path.unlink(missing_ok=True)
            return {
                "channel_config": channel_config,
                "trigger_hub": payload.get("trigger_hub", json.loads(json.dumps(DEFAULT_TRIGGER_HUB_CONFIG))),
            }
        return {
            "channel_config": sdk.ChannelConfig.from_json(str(path)),
            "trigger_hub": json.loads(json.dumps(DEFAULT_TRIGGER_HUB_CONFIG)),
        }
    config = default_gui_config()
    save_gui_config(config["channel_config"], config["trigger_hub"], path)
    return config


def save_gui_config(config, trigger_hub, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".channel.tmp.json")
    config.to_json(str(temp_path))
    try:
        channel_payload = json.loads(temp_path.read_text(encoding="utf-8"))
    finally:
        temp_path.unlink(missing_ok=True)
    payload = {
        "schema": "brainsync-gui-channel-config-v1",
        "channel_config": channel_payload,
        "trigger_hub": trigger_hub or json.loads(json.dumps(DEFAULT_TRIGGER_HUB_CONFIG)),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_channel_config(path: Path):
    return load_gui_config(path)["channel_config"]


def save_channel_config(config, path: Path) -> None:
    save_gui_config(config, json.loads(json.dumps(DEFAULT_TRIGGER_HUB_CONFIG)), path)


def active_mask_from_booleans(active_flags) -> int:
    mask = 0
    for index, active in enumerate(active_flags):
        if active:
            mask |= 1 << index
    return mask


def with_active_mask(config, active_mask: int):
    return sdk.ChannelConfig(
        labels=list(config.labels),
        active_mask=active_mask,
        ref_label=config.ref_label,
        gnd_label=config.gnd_label,
        eeg=list(config.eeg),
        stim=list(config.stim),
    )


def trigger_hub_dict_to_runtime(trigger_hub):
    aliases = {}
    thresholds = {}
    enabled = {}
    for item in trigger_hub.get("inputs", []):
        code = item.get("code")
        if not code:
            continue
        aliases[code] = item.get("label", code)
        if item.get("threshold_mv") is not None:
            thresholds[code] = float(item.get("threshold_mv"))
        enabled[code] = bool(item.get("enabled", False))
    return aliases, thresholds, enabled


def trigger_hub_runtime_to_dict(aliases, thresholds, enabled):
    defaults = {item["code"]: item for item in DEFAULT_TRIGGER_HUB_CONFIG["inputs"]}
    inputs = []
    for code in ["PD", "AUD", "MIC", "BTN"]:
        default = defaults.get(code, {})
        inputs.append({
            "code": code,
            "label": aliases.get(code, default.get("label", code)),
            "enabled": bool(enabled.get(code, default.get("enabled", False))),
            "threshold_mv": thresholds.get(code, default.get("threshold_mv")),
        })
    return {
        "inputs": inputs,
        "output": {"connector": "BrainSync Trigger Hub 3.5 mm"},
    }
