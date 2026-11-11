"""Command-line interface for cartoonify."""

import argparse
import sys
from pathlib import Path

from cartoonify.pipeline import cartoonify, read_image, save_image

STYLES = ("classic", "sketch", "comic", "hdr")
STAGE_NAMES = ("original", "grayscale", "edges", "flattened", "cartoon")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cartoonify",
        description="Turn a photo into a cartoon-style image with OpenCV.",
    )
    parser.add_argument("input", help="path to the input image")
    parser.add_argument("-o", "--output", help="path for the cartoon image (default: <input>_<style>.png)")
    parser.add_argument(
        "--style",
        choices=STYLES,
        default="classic",
        help="cartoon style (default: classic)",
    )
    parser.add_argument(
        "--stages-dir",
        type=Path,
        help="optional directory to save intermediate pipeline stages",
    )
    parser.add_argument("--block-size", type=int, default=9, help="edge detection block size (odd)")
    parser.add_argument("--c", type=int, default=9, help="edge detection constant subtracted from the mean")
    parser.add_argument("--diameter", type=int, default=9, help="bilateral filter diameter")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = (
        Path(args.output)
        if args.output
        else Path(args.input).with_name(f"{Path(args.input).stem}_{args.style}.png")
    )

    try:
        image = read_image(args.input)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        stages = cartoonify(
            image,
            style=args.style,
            block_size=args.block_size,
            c=args.c,
            diameter=args.diameter,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(stages.cartoon, output)
    print(f"Cartoon saved to: {output}")

    if args.stages_dir:
        args.stages_dir.mkdir(parents=True, exist_ok=True)
        for name in STAGE_NAMES:
            stage = getattr(stages, name)
            if stage is not None:
                save_image(stage, args.stages_dir / f"{name}.png")
        print(f"Pipeline stages saved to: {args.stages_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
