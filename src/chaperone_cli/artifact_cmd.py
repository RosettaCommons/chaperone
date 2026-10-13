"""CLI for bounded artifact and scheduler-log reads."""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

from chaperone.artifacts import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    pattern_for_view,
    read_excerpt,
)
from chaperone_cli.args import nonnegative_integer, positive_integer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipelines artifact",
        description="Read a bounded excerpt from a line-oriented artifact.",
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--cursor", type=nonnegative_integer, default=0)
    parser.add_argument("--max-lines", type=positive_integer, default=DEFAULT_MAX_LINES)
    parser.add_argument("--max-bytes", type=positive_integer, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--tail", action="store_true")
    parser.add_argument("--view", choices=("raw", "errors", "events"), default="raw")
    parser.add_argument("--pattern", help="case-insensitive byte-oriented regex")
    parser.add_argument(
        "--start-line",
        type=positive_integer,
        help="begin at this 1-indexed line, so 'rg -n' output composes directly",
    )
    parser.add_argument(
        "--section",
        help="read one markdown section by heading text instead of the document",
    )
    args = parser.parse_args(argv)
    if args.tail and args.cursor:
        parser.error("--tail cannot be combined with a nonzero --cursor")

    path = args.path.expanduser().resolve()
    try:
        selected = pattern_for_view(args.view, args.pattern)
        excerpt = read_excerpt(
            path,
            cursor=args.cursor,
            max_lines=args.max_lines,
            max_bytes=args.max_bytes,
            tail=args.tail,
            pattern=selected,
            start_line=args.start_line,
            section=args.section,
        )
    except (OSError, ValueError, re.error) as error:
        parser.error(str(error))

    print(
        f"[artifact path={path} cursor={excerpt.cursor} "
        f"source_lines={excerpt.source_lines} content_bytes={excerpt.content_bytes}]"
    )
    print(excerpt.text, end="" if excerpt.text.endswith("\n") else "\n")
    if excerpt.next_cursor is not None:
        command = [
            "pipelines",
            "artifact",
            str(path),
            "--cursor",
            str(excerpt.next_cursor),
            "--max-lines",
            str(args.max_lines),
            "--max-bytes",
            str(args.max_bytes),
        ]
        if args.view != "raw":
            command.extend(("--view", args.view))
        if args.pattern:
            command.extend(("--pattern", args.pattern))
        print("[truncated; continue with: " + shlex.join(command) + "]")
    elif excerpt.truncated:
        print("[tail excerpt; earlier matching content was omitted]")
    else:
        print("[end of artifact]")
    return 0
