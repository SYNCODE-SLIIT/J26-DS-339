"""Command-line entry point for running and comparing melody pipelines."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import paths
from .audio_utils import discover_audio_files
from .compare import compare_pipelines
from .config import PipelineConfig
from .errors import ToolUnavailable
from .manifest import build_manifest
from .pipelines import PIPELINES, PipelineRunner


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List the available pipeline combinations")
    subparsers.add_parser("doctor", help="Check paths and optional tool availability")
    manifest = subparsers.add_parser("manifest", help="Build an ML index of completed outputs")
    manifest.add_argument("--output-dir", type=Path, default=paths.OUTPUT_DIR)
    manifest.add_argument("--pipelines", nargs="+", choices=sorted(PIPELINES))

    for command in ("run", "compare"):
        child = subparsers.add_parser(command, help=f"{command.title()} extraction pipelines")
        child.add_argument("--pipelines", nargs="+", choices=sorted(PIPELINES), required=True)
        child.add_argument("--input-dir", type=Path, default=paths.INPUT_AUDIO_DIR)
        child.add_argument("--output-dir", type=Path, default=paths.OUTPUT_DIR)
        child.add_argument("--limit", type=int)
        child.add_argument("--pattern", help="Optional recursive glob, e.g. '14/*.mp3'")
        child.add_argument("--overwrite", action="store_true")
        child.add_argument("--separation-model", default=PipelineConfig().separation_model)
        child.add_argument(
            "--fail-on-unavailable",
            action="store_true",
            help="Stop instead of recording an optional tool as unavailable",
        )
    return parser


def _config(args: argparse.Namespace) -> PipelineConfig:
    return replace(
        PipelineConfig().with_paths(input_dir=args.input_dir, output_dir=args.output_dir),
        overwrite=args.overwrite,
        separation_model=args.separation_model,
    )


def _doctor() -> int:
    checks = {
        "input_dir": str(paths.INPUT_AUDIO_DIR),
        "input_exists": paths.INPUT_AUDIO_DIR.exists(),
        "output_dir": str(paths.OUTPUT_DIR),
        "mlx_audio_separator": importlib.util.find_spec("mlx_audio_separator") is not None,
        "basic_pitch_in_root": importlib.util.find_spec("basic_pitch") is not None,
        "basic_pitch_python": str(paths.BASIC_PITCH_PYTHON) if paths.BASIC_PITCH_PYTHON else None,
        "essentia_in_root": importlib.util.find_spec("essentia") is not None,
        "essentia_python": str(paths.ESSENTIA_PYTHON) if paths.ESSENTIA_PYTHON else None,
        "sheetsage": importlib.util.find_spec("sheetsage") is not None,
        "game_root": str(paths.GAME_ROOT) if paths.GAME_ROOT else None,
        "game_model": str(paths.GAME_MODEL) if paths.GAME_MODEL else None,
        "rmvpe_python": str(paths.RMVPE_PYTHON) if paths.RMVPE_PYTHON else None,
        "rmvpe_model": str(paths.RMVPE_MODEL) if paths.RMVPE_MODEL else None,
        "rosvot_root": str(paths.ROSVOT_ROOT) if paths.ROSVOT_ROOT else None,
        "rosvot_python": str(paths.ROSVOT_PYTHON) if paths.ROSVOT_PYTHON else None,
    }
    if paths.INPUT_AUDIO_DIR.exists():
        checks["audio_file_count"] = len(discover_audio_files(paths.INPUT_AUDIO_DIR))
    print(json.dumps(checks, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _base_parser().parse_args(argv)
    if args.command == "list":
        for info in PIPELINES.values():
            optional = f" [needs: {', '.join(info.optional_tools)}]" if info.optional_tools else ""
            print(f"{info.name:28} {info.description}{optional}")
        return 0
    if args.command == "doctor":
        return _doctor()
    if args.command == "manifest":
        frame = build_manifest(args.output_dir.expanduser().resolve(), args.pipelines)
        print(f"Indexed {len(frame)} result(s) in {args.output_dir / 'manifest.parquet'}")
        return 0

    config = _config(args)
    if args.command == "compare":
        frame = compare_pipelines(
            config,
            args.pipelines,
            limit=args.limit,
            pattern=args.pattern,
            skip_unavailable=not args.fail_on_unavailable,
        )
        print(frame.to_string(index=False))
        print(f"\nSaved comparison to {config.output_dir / 'comparison.csv'}")
        return 1 if not frame.empty and frame["status"].eq("failed").any() else 0

    runner = PipelineRunner(config)
    files = discover_audio_files(config.input_dir, pattern=args.pattern, limit=args.limit)
    if not files:
        print(f"No audio files found under {config.input_dir}", file=sys.stderr)
        return 1
    failed = False
    unavailable: dict[str, str] = {}
    for audio_path in files:
        for pipeline in args.pipelines:
            if pipeline in unavailable:
                print(f"unavailable {pipeline:28} {unavailable[pipeline]}")
                continue
            try:
                result, status = runner.run(pipeline, audio_path)
                output = runner.output_dir_for(pipeline, audio_path)
                print(f"{status:9} {pipeline:28} {audio_path.name} -> {output} ({len(result.notes)} notes)")
            except ToolUnavailable as exc:
                if args.fail_on_unavailable:
                    raise
                unavailable[pipeline] = str(exc)
                print(f"unavailable {pipeline:28} {exc}")
            except Exception as exc:
                failed = True
                print(f"failed    {pipeline:28} {audio_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
