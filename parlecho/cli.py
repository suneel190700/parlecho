"""Command-line entry point: parlecho dub <input> --to <lang>"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="parlecho",
                                     description="Speech translation with voice-cloned dubbing")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dub", help="Dub an audio/video file into a target language")
    p.add_argument("input", type=Path)
    p.add_argument("--to", required=True, dest="target", help="target language code, e.g. en")
    p.add_argument("--from", dest="source", default=None, help="source language (default: auto)")
    p.add_argument("--out", type=Path, default=None, help="output directory")

    args = parser.parse_args()
    if args.command == "dub":
        from parlecho.pipeline import dub
        dub(args.input, args.target, source_lang=args.source, output_dir=args.out)


if __name__ == "__main__":
    main()
