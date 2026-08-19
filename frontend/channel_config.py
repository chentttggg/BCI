"""Load our 8-channel montage and build BrainSync SDK ChannelConfig objects.

Project montage (single source of truth):
    Fz, Cz, P3, Pz, P4, PO7, PO8, Oz
    REF = A1, GND = A1 (耳部接地/REF 共用电极)

The BrainSync SDK default_8ch montage (C6, C4, FC4, FC6, F2, Fz, Cz, C2) is
NOT used by this project.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_montage(path: str | Path = "config/channel_config.json") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    channels = [str(item["label"]) for item in obj["eeg_channels"]]
    indices = [int(item["index"]) for item in obj["eeg_channels"]]
    if sorted(indices) != list(range(8)):
        raise ValueError(f"channel indices must be exactly 0..7, got {indices}")
    if len(set(channels)) != 8:
        raise ValueError(f"duplicate channel labels: {channels}")
    return {
        "channels": channels,
        "indices": indices,
        "ref_label": str(obj.get("ref_label", "A1")),
        "gnd_label": str(obj.get("gnd_label", "A1")),
    }


def build_sdk_channel_config(path: str | Path = "config/channel_config.json") -> Any:
    """Build a ``brainsync_sdk.ChannelConfig`` with our montage.

    Returns None when the optional SDK is unavailable (mock-only installs).
    """
    montage = read_montage(path)
    try:
        from brainsync_sdk import ChannelConfig, ElectrodeAssignment
    except Exception:
        return None
    assignments = [ElectrodeAssignment(label, idx)
                   for label, idx in zip(montage["channels"], montage["indices"])]
    return ChannelConfig(
        labels=montage["channels"],
        active_mask=0xFF,
        ref_label=montage["ref_label"],
        gnd_label=montage["gnd_label"],
        eeg=assignments,
    )


def gui_schema(path: str | Path = "config/channel_config.json") -> dict[str, Any]:
    """Return the BrainSync GUI/Multimodal-Hub compatible channel-config schema."""
    montage = read_montage(path)
    return {
        "schema": "brainsync-gui-channel-config-v1",
        "channel_config": {
            "active_mask": 255,
            "labels": montage["channels"],
            "eeg": [{"label": label, "channel": idx}
                    for label, idx in zip(montage["channels"], montage["indices"])],
            "stim": [],
            "ref_label": montage["ref_label"],
            "gnd_label": montage["gnd_label"],
        },
        "trigger_hub": {
            "inputs": [
                {"channel": 0, "name": "PD", "enabled": True, "threshold": 500},
                {"channel": 1, "name": "AUDIO", "enabled": False, "threshold": 500},
                {"channel": 2, "name": "KEY", "enabled": True, "threshold": 500},
            ]
        },
    }
