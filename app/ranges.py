"""HTTP byte-range support so the browser's video player can seek."""
from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from fastapi.responses import Response, StreamingResponse

CHUNK_BYTES = 64 * 1024
_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeNotSatisfiableError(ValueError):
    pass


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Return (start, end) inclusive for one byte range, or None to serve the whole file.

    Malformed or multi-range headers are ignored (whole file); a range that lies
    outside the file raises RangeNotSatisfiableError.
    """
    if not header:
        return None
    match = _RANGE.fullmatch(header.strip())
    if match is None:
        return None
    first, last = match.groups()
    if not first and not last:
        return None
    if not first:  # suffix range: the last N bytes
        length = int(last)
        if length == 0 or size == 0:
            raise RangeNotSatisfiableError("empty suffix range")
        return max(size - length, 0), size - 1
    start = int(first)
    end = int(last) if last else size - 1
    if last and end < start:
        return None
    if start >= size:
        raise RangeNotSatisfiableError("range starts past the end of the file")
    return start, min(end, size - 1)


def iter_file(path: Path, start: int, end: int) -> Iterator[bytes]:
    remaining = end - start + 1
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            data = handle.read(min(CHUNK_BYTES, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


def serve_video(path: Path, range_header: str | None) -> Response:
    size = path.stat().st_size
    try:
        byte_range = parse_range(range_header, size)
    except RangeNotSatisfiableError:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    headers = {"Accept-Ranges": "bytes"}
    if byte_range is None:
        headers["Content-Length"] = str(size)
        return StreamingResponse(
            iter_file(path, 0, size - 1), media_type="video/mp4", headers=headers
        )
    start, end = byte_range
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)
    return StreamingResponse(
        iter_file(path, start, end),
        status_code=206,
        media_type="video/mp4",
        headers=headers,
    )
