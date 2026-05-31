from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import analyze_source


def app() -> None:
    parser = argparse.ArgumentParser(prog="warzone_analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a saved video or live stream.")
    analyze_parser.add_argument("source", help="Video file path or stream URL.")
    analyze_parser.add_argument("--output-dir", "-o", type=Path, default=Path("data/output/run"))
    analyze_parser.add_argument("--config", "-c", type=Path, default=None)
    analyze_parser.add_argument("--start-at", type=float, default=0.0)
    analyze_parser.add_argument("--duration", type=float, default=None)

    args = parser.parse_args()
    analyzer_config = load_config(args.config)
    summary = analyze_source(
        source=args.source,
        output_dir=args.output_dir,
        config=analyzer_config,
        start_at_seconds=args.start_at,
        duration_seconds=args.duration,
    )
    print(summary.to_json(indent=2))
