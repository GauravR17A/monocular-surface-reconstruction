"""Opt-in classification routes with no client-provided filesystem paths."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from uuid import uuid4

from fastapi import Form, HTTPException

from msr.inference.rgb_preview import PROJECT_ROOT

DEMO_IMAGES = {
    "urban": "copenhagen_rgb.jpg", "sparse": "landscapes/sparse/texture.jpg",
    "hilly": "landscapes/hilly/texture.jpg", "forest": "forest_validation/satellite_rgb.jpg",
}


def resolve_source(root: Path, job_id: str | None, demo_id: str | None) -> Path:
    if bool(job_id) == bool(demo_id):
        raise ValueError("Choose exactly one uploaded scene or reference scene")
    if demo_id:
        if demo_id not in DEMO_IMAGES:
            raise ValueError("Unknown reference scene")
        return PROJECT_ROOT / "viewer/public/demo" / DEMO_IMAGES[demo_id]
    if not re.fullmatch(r"[0-9a-f]{32}", job_id or ""):
        raise ValueError("Invalid scene identifier")
    directory = (root / job_id).resolve()
    if directory.parent != root.resolve():
        raise ValueError("Invalid scene directory")
    # A completed height job ensures source and current reconstruction match.
    if not (directory / "metadata.json").is_file():
        raise ValueError("Scene is missing or has not finished reconstruction")
    sources = [p for p in directory.glob("input.*") if p.is_file() and p.resolve().parent == directory]
    if len(sources) != 1:
        raise ValueError("Original RGB source is unavailable; upload the image again")
    return sources[0]


def register_classification_routes(app, runtime, root: Path):
    @app.post("/api/classify")
    async def classify(job_id: str | None = Form(None), demo_id: str | None = Form(None)):
        try:
            source = resolve_source(root, job_id, demo_id)
            if not source.is_file():
                raise ValueError("Scene image is unavailable")
            identity = "classification_" + uuid4().hex
            output = root / identity
            report = await asyncio.to_thread(runtime.classify, source, output)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"Classification preview failed: {error}") from error
        base = f"/results/{identity}"
        return {**report, "labels_url": f"{base}/classes.bin", "raster_url": f"{base}/classes.tif",
                "metadata_url": f"{base}/metadata.json", "source_job_id": job_id, "source_demo_id": demo_id}
