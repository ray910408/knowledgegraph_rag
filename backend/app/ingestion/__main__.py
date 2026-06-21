from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import IngestionError, build_ingestion_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.app.ingestion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build offline ingestion artifacts")
    build.add_argument("--input", required=True, type=Path, help="Raw data directory")
    build.add_argument("--processed", required=True, type=Path, help="Processed output directory")
    build.add_argument(
        "--target",
        choices=("json", "bm25", "qdrant", "neo4j", "all"),
        default="all",
    )
    build.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Write local mock artifacts when Docker services are not available",
    )

    args = parser.parse_args()
    try:
        manifest = build_ingestion_artifacts(
            input_dir=args.input,
            processed_dir=args.processed,
            target=args.target,
            allow_fallback=args.allow_fallback,
        )
    except IngestionError as exc:
        parser.exit(2, f"error: {exc}\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
