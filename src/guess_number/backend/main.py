"""Backend entry point.

The researcher exe delegates heavy backend work to a local Python interpreter:

    python -m guess_number.backend.main ingest  --data-dir data/raw
    python -m guess_number.backend.main train   --data-dir data/raw --cv --production
    python -m guess_number.backend.main predict --edf data/raw/sub-..._eeg.edf \
        --model-dir data/derived/models/guess_number
    python -m guess_number.backend.main report  --data-dir data/raw

Each subcommand owns its argument parser, so there is only one CLI definition
per command (ingest.py, train.py, predict.py, qc_report.py).
"""
from __future__ import annotations

import sys
from importlib import import_module

_COMMANDS = ("ingest", "train", "predict", "report")


def _subcommand_module(command: str) -> str:
    return {
        "ingest": "guess_number.backend.ingest",
        "train": "guess_number.backend.train",
        "predict": "guess_number.backend.predict",
        "report": "guess_number.backend.qc_report",
    }[command]


def _print_help() -> None:
    print("""usage: guess-number-backend {ingest,train,predict,report} ...

commands:
  ingest   raw data inventory / SHA-256 / channel and marker QC
  report   generate per-session QC HTML/PNG/JSON
  train    prepare data and train the ShallowConvNet ensemble
  predict  predict the thought number for one EDF session

Run `guess-number-backend <command> --help` for command options.""")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        raise SystemExit(0 if argv else 1)
    command, *sub_args = argv
    if command not in _COMMANDS:
        _print_help()
        raise SystemExit(2)
    module = import_module(_subcommand_module(command))
    code = module.main(sub_args)
    raise SystemExit(0 if code is None else int(code))


if __name__ == "__main__":
    main()
