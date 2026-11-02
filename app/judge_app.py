"""Local app for the blind judging sheet: one object claim per page, no verdicts shown.

Run it with scripts/judge.py. It only answers to localhost names, and every form
carries a per-process token.
"""
from __future__ import annotations

import secrets
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.judging import (
    JudgingDataError,
    Verdict,
    build_item_view,
    load_judgments,
    load_sheet,
    next_unjudged,
    progress,
    record_judgment,
    save_judgments,
)
from app.ranges import serve_video
from app.run_data import is_valid_stem

BASE_DIR = Path(__file__).parent
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
CSP = (
    "default-src 'self'; media-src 'self'; img-src 'self' data:; "
    "frame-ancestors 'none'; form-action 'self'; base-uri 'none'"
)


def create_judge_app(
    work_dir: Path, clips_dir: Path = Path("data/clips")
) -> FastAPI:
    work_dir = Path(work_dir)
    clips_dir = Path(clips_dir)
    app = FastAPI(
        title="Blind judging sheet", docs_url=None, redoc_url=None, openapi_url=None
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=BASE_DIR / "templates")
    csrf_token = secrets.token_urlsafe(32)
    app.state.csrf_token = csrf_token

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        return templates.TemplateResponse(
            request,
            "error.html",
            {"status": exc.status_code, "message": str(exc.detail)},
            status_code=exc.status_code,
        )

    def sheet_or_404():
        try:
            return load_sheet(work_dir)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No judging sheet yet. Run: python scripts/make_judge_sheet.py "
                    f"--work-dir {work_dir}"
                ),
            ) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=500, detail="The judging sheet could not be read."
            ) from exc

    def store_or_500():
        try:
            return load_judgments(work_dir)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=500, detail="The judgments file could not be read."
            ) from exc

    def check_csrf(token: str) -> None:
        if not secrets.compare_digest(token.encode(), csrf_token.encode()):
            raise HTTPException(
                status_code=403,
                detail="Invalid form token. Reload the page and try again.",
            )

    @app.get("/")
    def home(request: Request):
        sheet = sheet_or_404()
        store = store_or_500()
        judged, total = progress(sheet, store)
        following = next_unjudged(sheet, store)
        return templates.TemplateResponse(
            request,
            "judge_home.html",
            {
                "judged": judged,
                "total": total,
                "next_id": following.item_id if following else None,
                "first_id": sheet.items[0].item_id if sheet.items else None,
            },
        )

    @app.get("/judge/{item_id}")
    def judge(request: Request, item_id: str):
        sheet = sheet_or_404()
        store = store_or_500()
        try:
            view = build_item_view(work_dir, sheet, store, item_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Item not found.") from exc
        except (JudgingDataError, OSError, ValueError) as exc:
            raise HTTPException(
                status_code=500, detail="This run's files could not be read."
            ) from exc
        judged, _ = progress(sheet, store)
        return templates.TemplateResponse(
            request,
            "judge.html",
            {"view": view, "judged": judged, "csrf": csrf_token},
        )

    @app.post("/judge/{item_id}")
    def submit(
        item_id: str,
        verdict: Annotated[str, Form()],
        csrf: Annotated[str, Form()] = "",
        note: Annotated[str, Form()] = "",
    ):
        check_csrf(csrf)
        sheet = sheet_or_404()
        store = store_or_500()
        try:
            parsed = Verdict(verdict)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Unknown verdict.") from exc
        try:
            updated = record_judgment(store, sheet, item_id, parsed, note=note)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Item not found.") from exc
        except ValidationError as exc:
            raise HTTPException(
                status_code=400, detail=str(exc.errors()[0]["msg"])
            ) from exc
        save_judgments(work_dir, updated)
        following = next_unjudged(sheet, updated, after=item_id)
        target = f"/judge/{following.item_id}" if following else "/"
        return RedirectResponse(target, status_code=303)

    @app.get("/media/{stem}")
    def media(request: Request, stem: str):
        path = clips_dir / f"{stem}.mp4"
        if not is_valid_stem(stem) or not path.is_file():
            raise HTTPException(status_code=404, detail="Video not found.")
        return serve_video(path, request.headers.get("range"))

    return app
