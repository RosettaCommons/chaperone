"""Bounded, cursor-based reads for logs and other line-oriented artifacts."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Pattern


DEFAULT_MAX_LINES = 200
DEFAULT_MAX_BYTES = 20 * 1024
ERROR_PATTERN = re.compile(
    rb"(?:error|exception|traceback|failed|failure|out[_ -]?of[_ -]?memory|"
    rb"\boom\b|timed?\s*out|cancelled)",
    re.IGNORECASE,
)
EVENT_PATTERN = re.compile(
    rb"(?:submitted|pending|running|completed|finished|succeeded|failed|"
    rb"cancelled|timed?\s*out|out[_ -]?of[_ -]?memory|score report|"
    rb"pass[_ -]?rate|summary)",
    re.IGNORECASE,
)
HEADING_PATTERN = re.compile(rb"^(#{1,6})\s+(.*)$")
_SCAN_CHUNK = 64 * 1024


@dataclass(frozen=True, slots=True)
class Excerpt:
    """A bounded slice and the byte cursor needed to continue reading."""

    text: str
    cursor: int
    next_cursor: int | None
    source_lines: int
    content_bytes: int
    truncated: bool


def pattern_for_view(
    view: str = "raw", pattern: str | bytes | Pattern[bytes] | None = None
) -> Pattern[bytes] | None:
    """Compile a custom filter or select a built-in artifact view."""
    if pattern is not None:
        if isinstance(pattern, re.Pattern):
            return pattern
        expression = pattern.encode() if isinstance(pattern, str) else pattern
        return re.compile(expression, re.IGNORECASE)
    if view == "raw":
        return None
    if view == "errors":
        return ERROR_PATTERN
    if view == "events":
        return EVENT_PATTERN
    raise ValueError("view must be one of: raw, errors, events")


def _read_forward(
    handle: BinaryIO,
    *,
    cursor: int,
    max_lines: int,
    max_bytes: int,
    pattern: Pattern[bytes] | None,
) -> Excerpt:
    handle.seek(cursor)
    parts: list[bytes] = []
    emitted_lines = 0
    source_lines = 0
    used = 0
    next_cursor: int | None = None

    while emitted_lines < max_lines and used < max_bytes:
        line_start = handle.tell()
        # Bound the transient allocation as well as the returned excerpt. A
        # pathological log record without a newline is read over multiple
        # cursor-safe fragments instead of being loaded in full.
        read_limit = max_bytes if pattern is not None else max_bytes - used
        raw = handle.readline(read_limit)
        if not raw:
            break
        source_lines += 1
        if pattern is not None and pattern.search(raw) is None:
            continue
        remaining = max_bytes - used
        consumed = min(len(raw), remaining)
        if consumed:
            parts.append(raw[:consumed])
        used += consumed
        emitted_lines += 1
        if consumed < len(raw):
            next_cursor = line_start + consumed
            break

    if next_cursor is None:
        position = handle.tell()
        if handle.read(1):
            next_cursor = position

    content = b"".join(parts)
    return Excerpt(
        text=content.decode("utf-8", errors="replace"),
        cursor=cursor,
        next_cursor=next_cursor,
        source_lines=source_lines,
        content_bytes=len(content),
        truncated=next_cursor is not None,
    )


def _read_tail(
    handle: BinaryIO,
    *,
    max_lines: int,
    max_bytes: int,
    pattern: Pattern[bytes] | None,
) -> Excerpt:
    # Scan without retaining the complete artifact. Keep completed matching
    # records under both public output limits as the window is built. The
    # in-progress record is itself suffix-capped, so a missing newline cannot
    # cause an unbounded allocation.
    matches: deque[tuple[int, bytes]] = deque()
    match_count = 0
    source_lines = 0
    retained_bytes = 0
    clipped_content = False
    current = b""
    current_offset = 0
    current_clipped = False
    current_matches = pattern is None

    def retain_current() -> None:
        nonlocal match_count, retained_bytes, clipped_content
        if not current_matches:
            return
        match_count += 1
        clipped_content = clipped_content or current_clipped
        matches.append((current_offset, current))
        retained_bytes += len(current)

        while len(matches) > max_lines:
            _, removed = matches.popleft()
            retained_bytes -= len(removed)

        if retained_bytes > max_bytes:
            excess = retained_bytes - max_bytes
            first_offset, first = matches[0]
            if excess >= len(first):
                matches.popleft()
                retained_bytes -= len(first)
            else:
                matches[0] = (first_offset + excess, first[excess:])
                retained_bytes -= excess
            clipped_content = True

    while True:
        offset = handle.tell()
        # ``readline()`` without a size can allocate an entire malformed log
        # record. Treat records larger than the excerpt budget as fragments.
        raw = handle.readline(max_bytes)
        if not raw:
            break
        combined = current + raw
        if len(combined) > max_bytes:
            current_clipped = True
            combined = combined[-max_bytes:]
        current = combined
        current_offset = offset + len(raw) - len(current)
        if pattern is not None and pattern.search(current) is not None:
            current_matches = True

        if raw.endswith(b"\n"):
            source_lines += 1
            retain_current()
            current = b""
            current_clipped = False
            current_matches = pattern is None

    if current:
        source_lines += 1
        retain_current()

    content = b"".join(raw for _, raw in matches)
    start = matches[0][0] if matches else handle.tell()
    return Excerpt(
        text=content.decode("utf-8", errors="replace"),
        cursor=start,
        next_cursor=None,
        source_lines=source_lines,
        content_bytes=len(content),
        truncated=match_count > len(matches) or clipped_content,
    )


def _offset_of_line(handle: BinaryIO, line: int) -> int:
    """Return the byte offset where the 1-indexed ``line`` begins."""
    if line == 1:
        return 0
    handle.seek(0)
    remaining = line - 1
    offset = 0
    while True:
        chunk = handle.read(_SCAN_CHUNK)
        if not chunk:
            raise ValueError(f"artifact has fewer than {line} lines")
        found = chunk.count(b"\n")
        if found < remaining:
            remaining -= found
            offset += len(chunk)
            continue
        position = -1
        for _ in range(remaining):
            position = chunk.index(b"\n", position + 1)
        return offset + position + 1


def _find_section(handle: BinaryIO, name: str) -> tuple[int, int | None]:
    """Locate a markdown section by heading text.

    Returns the heading's byte offset and the offset of the next heading at the
    same or higher level, so a named section can be read without scanning the
    surrounding document into memory.
    """
    needle = name.strip().lower().encode()
    handle.seek(0)
    start: int | None = None
    level = 0
    while True:
        offset = handle.tell()
        raw = handle.readline(DEFAULT_MAX_BYTES)
        if not raw:
            break
        heading = HEADING_PATTERN.match(raw)
        if heading is None:
            continue
        if start is None:
            if needle in heading.group(2).strip().lower():
                start, level = offset, len(heading.group(1))
            continue
        if len(heading.group(1)) <= level:
            return start, offset
    if start is None:
        raise ValueError(f"section not found: {name}")
    return start, None


def read_excerpt(
    path: Path | str,
    *,
    cursor: int = 0,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    tail: bool = False,
    pattern: Pattern[bytes] | None = None,
    view: str = "raw",
    start_line: int | None = None,
    section: str | None = None,
) -> Excerpt:
    """Read a bounded excerpt without loading the complete artifact.

    ``next_cursor`` is a byte offset suitable for the next forward call. Tail
    mode scans the source but keeps at most ``max_lines``/``max_bytes`` in
    memory. Built-in ``errors`` and ``events`` views avoid dumping noisy logs.
    ``start_line`` accepts a line number so ``rg -n`` output composes directly,
    and ``section`` reads one markdown section instead of a whole document.
    """
    if cursor < 0:
        raise ValueError("cursor must be zero or greater")
    if max_lines < 1 or max_bytes < 1:
        raise ValueError("max_lines and max_bytes must be positive")
    if tail and cursor:
        raise ValueError("tail and cursor cannot be combined")
    if start_line is not None and start_line < 1:
        raise ValueError("start_line must be one or greater")
    if start_line is not None and (cursor or tail):
        raise ValueError("start_line cannot be combined with cursor or tail")
    if section is not None and (cursor or tail or start_line is not None):
        raise ValueError("section cannot be combined with cursor, tail, or start_line")
    selected_pattern = pattern if pattern is not None else pattern_for_view(view)
    with Path(path).open("rb") as handle:
        if section is not None:
            start, end = _find_section(handle, section)
            if end is not None:
                max_bytes = min(max_bytes, end - start)
            return _read_forward(
                handle,
                cursor=start,
                max_lines=max_lines,
                max_bytes=max_bytes,
                pattern=selected_pattern,
            )
        if start_line is not None:
            cursor = _offset_of_line(handle, start_line)
        if tail:
            return _read_tail(
                handle,
                max_lines=max_lines,
                max_bytes=max_bytes,
                pattern=selected_pattern,
            )
        return _read_forward(
            handle,
            cursor=cursor,
            max_lines=max_lines,
            max_bytes=max_bytes,
            pattern=selected_pattern,
        )
