"""Local review app for the archivist: FastAPI with server-rendered HTML.

Run: python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
It only answers to localhost names, and every form carries a per-process token.
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

from app.pipeline.catalog import build_catalog, catalog_to_csv, catalog_to_json
from app.pipeline.review import (
    Decision,
    load_store,
    record_decision,
    save_store,
    undo_decision,
)
from app.ranges import serve_video
from app.run_data import RunData, is_valid_stem, list_stems, load_run
from app.views import build_views, count_states

BASE_DIR = Path(__file__).parent
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
CSP = (
    "default-src 'self'; media-src 'self'; img-src 'self' data:; "
    "frame-ancestors 'none'; form-action 'self'; base-uri 'none'"
)


def create_app(
    work_dir: Path = Path("data/work"), clips_dir: Path = Path("data/clips")
) -> FastAPI:
    work_dir = Path(work_dir)
    clips_dir = Path(clips_dir)
    app = FastAPI(
        title="Verifiable archive indexer", docs_url=None, redoc_url=None, openapi_url=None
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

    def review_path(stem: str) -> Path:
        return work_dir / f"{stem}.review.json"

    def run_or_404(stem: str) -> RunData:
        try:
            return load_run(stem, work_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Clip not found.") from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=500, detail="This run's files could not be read."
            ) from exc

    def store_or_500(stem: str):
        try:
            return load_store(review_path(stem))
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=500, detail="The review file could not be read."
            ) from exc

    def check_csrf(token: str) -> None:
        if not secrets.compare_digest(token.encode(), csrf_token.encode()):
            raise HTTPException(
                status_code=403,
                detail="Invalid form token. Reload the page and try again.",
            )

    def back_to_clip(stem: str, position: str) -> RedirectResponse:
        anchor = f"#claim-{position}" if position.isascii() and position.isdigit() else ""
        return RedirectResponse(f"/clips/{stem}{anchor}", status_code=303)

    @app.get("/")
    def index(request: Request):
        clips = []
        for stem in list_stems(work_dir):
            try:
                run = load_run(stem, work_dir)
                store = load_store(review_path(stem))
            except (OSError, ValueError):
                clips.append({"stem": stem, "error": True})
                continue
            counts = count_states(build_views(run.claim_set, run.transcript, store))
            clips.append(
                {
                    "stem": stem,
                    "error": False,
                    "total": len(run.claim_set.claims),
                    "counts": counts,
                }
            )
        return templates.TemplateResponse(
            request, "index.html", {"clips": clips, "work_dir": str(work_dir)}
        )

    @app.get("/clips/{stem}")
    def review(request: Request, stem: str):
        run = run_or_404(stem)
        views = build_views(run.claim_set, run.transcript, store_or_500(stem))
        return templates.TemplateResponse(
            request,
            "review.html",
            {
                "stem": stem,
                "views": views,
                "counts": count_states(views),
                "turns": run.transcript.turns,
                "has_video": (clips_dir / f"{stem}.mp4").is_file(),
                "csrf": csrf_token,
            },
        )

    @app.post("/clips/{stem}/decide")
    def decide(
        stem: str,
        claim_id: Annotated[str, Form()],
        decision: Annotated[str, Form()],
        csrf: Annotated[str, Form()] = "",
        value: Annotated[str, Form()] = "",
        note: Annotated[str, Form()] = "",
        position: Annotated[str, Form()] = "",
    ):
        check_csrf(csrf)
        run = run_or_404(stem)
        try:
            parsed = Decision(decision)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Unknown decision.") from exc
        store = store_or_500(stem)
        try:
            updated = record_decision(
                store,
                run.claim_set,
                claim_id,
                parsed,
                value=value or None,
                note=note.strip() or None,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Claim not found.") from exc
        except ValidationError as exc:
            raise HTTPException(
                status_code=400, detail=str(exc.errors()[0]["msg"])
            ) from exc
        save_store(review_path(stem), updated)
        return back_to_clip(stem, position)

    @app.post("/clips/{stem}/undo")
    def undo(
        stem: str,
        claim_id: Annotated[str, Form()],
        csrf: Annotated[str, Form()] = "",
        position: Annotated[str, Form()] = "",
    ):
        check_csrf(csrf)
        run_or_404(stem)
        save_store(review_path(stem), undo_decision(store_or_500(stem), claim_id))
        return back_to_clip(stem, position)

    def catalog_for(stem: str, include_pending: bool):
        run = run_or_404(stem)
        return build_catalog(
            stem, run.claim_set, store_or_500(stem), include_pending=include_pending
        )

    @app.get("/clips/{stem}/export.json")
    def export_json(stem: str, include_pending: bool = False):
        body = catalog_to_json(catalog_for(stem, include_pending))
        return Response(
            body,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{stem}.catalog.json"'},
        )

    @app.get("/clips/{stem}/export.csv")
    def export_csv(stem: str, include_pending: bool = False):
        body = catalog_to_csv(catalog_for(stem, include_pending))
        return Response(
            body,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{stem}.catalog.csv"'},
        )

    @app.get("/media/{stem}")
    def media(request: Request, stem: str):
        path = clips_dir / f"{stem}.mp4"
        if not is_valid_stem(stem) or not path.is_file():
            raise HTTPException(status_code=404, detail="Video not found.")
        return serve_video(path, request.headers.get("range"))

    return app


app = create_app()
