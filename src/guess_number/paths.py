"""Path helpers so config files are found in dev and installed layouts."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_CONFIG_DIR = PACKAGE_DIR / "config"


def config_path(name: str) -> Path:
    """Resolve a packaged config file, allowing an explicit override directory.

    Precedence:
      1. GUESS_NUMBER_CONFIG_DIR/<name>
      2. ./<name> in the current working directory
      3. packaged src/guess_number/config/<name>
    """
    override = os.environ.get("GUESS_NUMBER_CONFIG_DIR")
    if override:
        candidate = Path(override) / name
        if candidate.exists():
            return candidate
    cwd_candidate = Path.cwd() / name
    if cwd_candidate.exists():
        return cwd_candidate
    packaged = PACKAGE_CONFIG_DIR / name
    if packaged.exists():
        return packaged
    return cwd_candidate


def default_channel_config_path() -> str:
    return str(config_path("channel_config.json"))


def default_preprocess_config_path() -> str:
    return str(config_path("preprocessing.json"))


def default_train_config_path() -> str:
    return str(config_path("train.json"))
