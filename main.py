"""
main.py
FastAPI app — English-to-Spanish image text translator.

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, JSONResponse

from image_translator import translate_image, _get_reader

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — warm up EasyOCR on startup so the first request isn't slow
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up EasyOCR model…")
    _get_reader()
    logger.info("Server ready.")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EN→ES Image Translator",
    description=(
        "Upload an image containing English text. "
        "The API detects all text, translates it to Spanish, "
        "replaces it in-place, and returns the modified image. "
        "Nothing else in the image is altered."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp", "image/tiff"}
MAX_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Utility"])
async def health():
    """Simple liveness check."""
    return {"status": "ok"}


@app.post(
    "/translate",
    tags=["Translation"],
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Translated image returned as PNG.",
        },
        400: {"description": "Invalid input (wrong file type, too large, etc.)"},
        500: {"description": "Internal processing error"},
    },
    summary="Translate English text in an image to Spanish",
)
async def translate(
    image: UploadFile = File(..., description="Image file (PNG, JPEG, WebP, BMP, or TIFF)"),
):
    """
    Upload an image. Every English text region is detected, translated to
    Spanish, and redrawn in the same position with a matching background
    and approximate font size. The translated PNG is returned directly.
    """
    # ── Validate content type ────────────────────────────────────────────────
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{image.content_type}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    # ── Read & size-check ────────────────────────────────────────────────────
    image_bytes = await image.read()
    if len(image_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(image_bytes) / 1e6:.1f} MB). Max allowed: 20 MB.",
        )
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Translate ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        result_bytes = translate_image(image_bytes)
    except Exception as exc:
        logger.exception("Error during image translation")
        raise HTTPException(status_code=500, detail=f"Translation failed: {exc}") from exc

    elapsed = time.perf_counter() - t0
    logger.info(f"Translated '{image.filename}' in {elapsed:.2f}s  ({len(result_bytes)} bytes out)")

    return Response(
        content=result_bytes,
        media_type="image/png",
        headers={
            "X-Processing-Time-Seconds": f"{elapsed:.3f}",
            "Content-Disposition": f'inline; filename="translated_{image.filename}"',
        },
    )


@app.post(
    "/translate/batch",
    tags=["Translation"],
    summary="Translate multiple images in one request",
    responses={
        200: {"description": "List of results (base64-encoded PNG or error per file)"},
    },
)
async def translate_batch(
    images: list[UploadFile] = File(..., description="One or more image files"),
):
    """
    Translate multiple images at once. Returns a JSON array where each element
    has either `image_b64` (base64-encoded translated PNG) or `error`.
    """
    import base64

    results = []
    for img_file in images:
        entry: dict = {"filename": img_file.filename}
        try:
            if img_file.content_type not in ALLOWED_CONTENT_TYPES:
                entry["error"] = f"Unsupported type: {img_file.content_type}"
                results.append(entry)
                continue
            raw = await img_file.read()
            translated = translate_image(raw)
            entry["image_b64"] = base64.b64encode(translated).decode()
            entry["media_type"] = "image/png"
        except Exception as exc:
            logger.exception(f"Error translating '{img_file.filename}'")
            entry["error"] = str(exc)
        results.append(entry)

    return JSONResponse(content={"results": results})