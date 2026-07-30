#!/usr/bin/env python3
"""Step 1: build frozen, label-independent physiographic macro-regions."""

from __future__ import annotations

import argparse

from utils.progress import configure_progress, console
from utils.regions import build_from_xml


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build aligned continuous terrain-derived macro-regions. Landslide labels, "
            "DWSS products, and predictions are never read by this step."
        )
    )
    parser.add_argument("xml", help="Project XML configuration.")
    parser.add_argument(
        "--force",
        "--force-regions",
        dest="force",
        action="store_true",
        help="Replace products only after an intentional regionalization change.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示实时进度条（默认开启，可用 --no-progress 关闭）。",
    )
    # Retained as a harmless compatibility alias for commands from the prior revision.
    parser.add_argument("--regions-only", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configure_progress(args.progress)
    console(f"步骤 1/3：构建或核验连续宏区域；配置={args.xml}")
    result = build_from_xml(args.xml, overwrite=args.force)
    for count, path in sorted(result["built"].items()):
        console(f"已构建 K={count}：{path}")
    for count, path in sorted(result["skipped"].items()):
        console(f"已核验 K={count}：{path}")
    console("步骤 1/3 完成。")


if __name__ == "__main__":
    main()
