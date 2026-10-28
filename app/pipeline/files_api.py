"""Files API helpers: upload a video once, reference it by ID, delete it after."""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import httpx

from app.config import Settings

MAX_UPLOAD_BYTES = 1_073_741_824  # documented 1 GiB cap for Files API uploads
MIN_EXPIRY_S = 3_600  # documented minimum for expires_after.seconds (1 hour)
MAX_EXPIRY_S = 2_592_000  # documented maximum (30 days)
RETRY_STATUSES = {429, 500}


class FilesError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def upload_video(
    path: Path,
    settings: Settings,
    *,
    expires_seconds: int = 86_400,
    max_retries: int = 2,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Upload an mp4 with purpose=user_data and return its file ID.

    Form fields go before the file part, as the documentation recommends.
    """
    path = Path(path)
    if path.suffix.lower() != ".mp4":
        raise ValueError("only .mp4 video is supported")
    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise ValueError(f"file is {size} bytes; the limit is {MAX_UPLOAD_BYTES}")
    if not MIN_EXPIRY_S <= expires_seconds <= MAX_EXPIRY_S:
        raise ValueError(
            f"expires_seconds must be between {MIN_EXPIRY_S} and {MAX_EXPIRY_S}"
        )

    url = f"{settings.api_base}/files"
    headers = {"Authorization": f"Bearer {settings.api_key}"}
    form = {
        "purpose": "user_data",
        "expires_after[anchor]": "created_at",
        "expires_after[seconds]": str(expires_seconds),
    }

    with httpx.Client(timeout=600.0) as client:
        response: httpx.Response | None = None
        for attempt in range(max_retries + 1):
            with path.open("rb") as handle:
                response = client.post(
                    url,
                    headers=headers,
                    data=form,
                    files={"file": (path.name, handle, "video/mp4")},
                )
            if response.status_code in RETRY_STATUSES and attempt < max_retries:
                sleep(2**attempt)
                continue
            break

    assert response is not None
    if not response.is_success:
        raise FilesError(
            f"upload failed with HTTP {response.status_code}: {response.text[:200]}",
            status_code=response.status_code,
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise FilesError("upload response is not valid JSON") from exc
    file_id = body.get("id") if isinstance(body, dict) else None
    if not isinstance(file_id, str) or not file_id.startswith("file-"):
        raise FilesError(f"upload response has no usable file id: {str(body)[:200]}")
    return file_id


def delete_file(
    file_id: str, settings: Settings, *, ignore_missing: bool = True
) -> None:
    """Delete an uploaded file; a 404 counts as already deleted by default."""
    response = httpx.delete(
        f"{settings.api_base}/files/{file_id}",
        headers={"Authorization": f"Bearer {settings.api_key}"},
        timeout=60.0,
    )
    if response.status_code == 404 and ignore_missing:
        return
    if not response.is_success:
        raise FilesError(
            f"delete failed with HTTP {response.status_code}: {response.text[:200]}",
            status_code=response.status_code,
        )
