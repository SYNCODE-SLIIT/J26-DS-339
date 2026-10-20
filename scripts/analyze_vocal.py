"""Run from the repository root: uv run python scripts/analyze_vocal.py AUDIO --output JSON."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from melody_understanding import analyze_vocal

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trim", action="store_true", help="Trim quiet ends and preserve source timestamps")
    args = parser.parse_args()
    if args.output.resolve() == args.audio.resolve():
        parser.error("Output must not overwrite the input recording")
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    result, _ = analyze_vocal(args.audio, trim=args.trim)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {len(result['notes'])} predicted notes to {args.output}")

if __name__ == "__main__":
    main()
