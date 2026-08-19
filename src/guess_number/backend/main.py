"""Backend entry point.

Usage:
    python -m backend.main ingest  --data-dir data/raw
    python -m backend.main train   --data-dir data/raw --cv --production
    python -m backend.main predict --edf data/raw/sub-..._eeg.edf --model-dir data/derived/models/guess_number
"""
from __future__ import annotations

import argparse
import sys

from guess_number.paths import (default_channel_config_path, default_preprocess_config_path, default_train_config_path)


def main() -> None:
    p = argparse.ArgumentParser(prog="guess-number-backend",
                                description="Guess-number P300 backend processing")
    sub = p.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="raw data inventory and integrity check")
    ingest.add_argument("--data-dir", default="data/raw")
    ingest.add_argument("--manifest", default="data/manifest.jsonl")
    ingest.add_argument("--channel-config", default=default_channel_config_path())
    ingest.add_argument("--preprocess-config", default=default_preprocess_config_path())

    train = sub.add_parser("train", help="train ShallowConvNet ensemble")
    train.add_argument("--data-dir", default="data/raw")
    train.add_argument("--output-dir", default="data/derived/models/guess_number")
    train.add_argument("--preprocess-config", default=default_preprocess_config_path())
    train.add_argument("--train-config", default=default_train_config_path())
    train.add_argument("--channel-config", default=default_channel_config_path())
    train.add_argument("--cv", action="store_true")
    train.add_argument("--no-cv", action="store_false", dest="cv")
    train.add_argument("--production", action="store_true")
    train.add_argument("--no-production", action="store_false", dest="production")
    train.add_argument("--log-file", default=None)

    predict = sub.add_parser("predict", help="predict the thought number")
    predict.add_argument("--edf", required=True)
    predict.add_argument("--events", default=None)
    predict.add_argument("--session-json", default=None)
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--channel-config", default=default_channel_config_path())
    predict.add_argument("--output-json", default=None)
    predict.add_argument("--log-file", default=None)

    report = sub.add_parser("report", help="generate per-session QC report")
    report.add_argument("--data-dir", default=None)
    report.add_argument("--edf", default=None)
    report.add_argument("--output-dir", default="data/derived/reports")
    report.add_argument("--channel-config", default=default_channel_config_path())
    report.add_argument("--preprocess-config", default=default_preprocess_config_path())

    args = p.parse_args()
    if args.command == "ingest":
        from .ingest import main as ingest_main
        sys.exit(ingest_main(["--data-dir", args.data_dir, "--manifest", args.manifest,
                              "--channel-config", args.channel_config,
                              "--preprocess-config", args.preprocess_config]))
    if args.command == "train":
        from .train import main as train_main
        sys.exit(train_main(["--data-dir", args.data_dir, "--output-dir", args.output_dir,
                             "--preprocess-config", args.preprocess_config,
                             "--train-config", args.train_config,
                             "--channel-config", args.channel_config,
                             "--cv" if args.cv else "--no-cv",
                             "--production" if args.production else "--no-production",
                             *(["--log-file", args.log_file] if args.log_file else [])]))
    if args.command == "predict":
        from .predict import main as predict_main
        sys.exit(predict_main(["--edf", args.edf,
                               *(["--events", args.events] if args.events else []),
                               *(["--session-json", args.session_json] if args.session_json else []),
                               "--model-dir", args.model_dir,
                               "--channel-config", args.channel_config,
                               *(["--output-json", args.output_json] if args.output_json else []),
                               *(["--log-file", args.log_file] if args.log_file else [])]))
    if args.command == "report":
        from .qc_report import main as report_main
        argv = ["--output-dir", args.output_dir,
                "--channel-config", args.channel_config,
                "--preprocess-config", args.preprocess_config]
        if args.edf:
            argv += ["--edf", args.edf]
        if args.data_dir:
            argv += ["--data-dir", args.data_dir]
        sys.exit(report_main(argv))


if __name__ == "__main__":
    main()
