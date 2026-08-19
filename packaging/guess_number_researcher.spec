# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

project_root = Path(SPECPATH).parent
src_dir = project_root / "src"
config_dir = src_dir / "guess_number" / "config"
entry_script = src_dir / "guess_number" / "gui" / "researcher.py"

hiddenimports = collect_submodules("guess_number")
# MNE/PySide6/torch hooks handle their own imports. Keep runtime lean-ish.
excludes = [
    "tkinter",
    "PyQt5",
    "PyQt6",
    "IPython",
    "jedi",
    "matplotlib.tests",
    "scipy.tests",
    "pandas.tests",
]

a = Analysis(
    [str(entry_script)],
    pathex=[str(src_dir)],
    binaries=[],
    datas=[(str(config_dir), "guess_number/config")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="GuessNumberResearcher",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
