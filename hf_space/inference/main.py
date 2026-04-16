"""FastAPI server for the March Madness 2026 retrospective.

Runtime is almost entirely static — all model predictions and retrospective
statistics are pre-computed by `prepare_data.py` at build time and written as
JSON + markdown into `web/public/data/`.  This server:

  - Serves the single-page app at `/`
  - Exposes a tiny API surface for the client to fetch pre-computed JSON
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

APP_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = APP_ROOT / "web" / "public"
DATA_DIR = WEB_DIR / "data"
DOCS_DIR = WEB_DIR / "docs"

app = FastAPI(title="March Madness 2026", version="1.0.0")


def _load_json(path: Path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not in build")
    with path.open() as f:
        return json.load(f)


@app.get("/api/manifest")
def manifest():
    """List of every model with metadata, plus tournament info."""
    return _load_json(DATA_DIR / "manifest.json")


@app.get("/api/bracket/{model_slug}")
def bracket(model_slug: str):
    path = DATA_DIR / "brackets" / f"{model_slug}.json"
    return _load_json(path)


@app.get("/api/retrospective")
def retrospective():
    return _load_json(DATA_DIR / "retrospective.json")


@app.get("/api/hindsight")
def hindsight():
    return _load_json(DATA_DIR / "hindsight.json")


@app.get("/api/docs/{slug}")
def docs(slug: str):
    path = DOCS_DIR / f"{slug}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{slug} not in build")
    return PlainTextResponse(path.read_text(), media_type="text/markdown")


@app.get("/api/health")
def health():
    return JSONResponse({"ok": True})


# Static site (must be mounted last so /api/* wins)
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
