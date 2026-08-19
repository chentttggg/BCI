# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

project_root = Path(SPECPATH).parent
src_dir = project_root / "src"
config_dir = src_dir / "guess_number" / "config"
entry_script = src_dir / "guess_number" / "gui" / "researcher.py"

# The exe is intentionally a thin GUI:
#   * it bundles PySide6 + the lightweight frontend (numpy, pyedflib, pylsl,
#     brainsync-sdk) so the experiment itself runs from the exe;
#   * heavy backend modules (MNE, torch, scipy, pandas, scikit-learn,
#     matplotlib) run in a local Python interpreter via
#     ``python -m guess_number.backend.main ...`` and are excluded here.
# Do NOT use collect_submodules("guess_number"): that would drag the backend
# (torch/mne/scipy/sklearn/matplotlib) back into the frozen app.
hiddenimports = []

excludes = [
    "tkinter",
    "PyQt5",
    "PyQt6",
    "IPython",
    "jedi",
    # Heavy scientific stack lives in the external Python environment.
    "torch",
    "mne",
    "sklearn",
    "scipy",
    "pandas",
    "matplotlib",
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
