"""
image_translator.py
Core pipeline: OCR → Translate (EN→ES) → Erase original text → Redraw translated text
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load EasyOCR reader (heavy init, only done once)
# ---------------------------------------------------------------------------
_ocr_reader = None


def _get_reader():
    """Return a cached EasyOCR reader. Initialises on first call."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr  # imported here so FastAPI boots fast even if GPU missing
        logger.info("Initialising EasyOCR reader (first call — may take a moment)…")
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
        logger.info("EasyOCR ready.")
    return _ocr_reader


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class TextRegion:
    """A detected text region with its bounding polygon and translated text."""
    # EasyOCR returns a list of 4 [x, y] points (clockwise from top-left)
    bbox: list[list[int]]
    original: str
    translated: str


# ---------------------------------------------------------------------------
# Background estimation helpers
# ---------------------------------------------------------------------------

def _bbox_to_rect(bbox: list[list[int]]) -> tuple[int, int, int, int]:
    """Convert a polygon bbox to an axis-aligned (left, top, right, bottom) rect."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _estimate_bg_color(img_array: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """
    Estimate the background colour of a bounding rect by sampling a 3-pixel
    border around its perimeter.  Falls back to white if the rect is too small.
    """
    left, top, right, bottom = rect
    h, w = img_array.shape[:2]

    # Expand border by 3 px (clamped to image edges)
    bx1 = max(left - 3, 0)
    by1 = max(top - 3, 0)
    bx2 = min(right + 3, w)
    by2 = min(bottom + 3, h)

    if bx2 <= bx1 or by2 <= by1:
        return (255, 255, 255)

    region = img_array[by1:by2, bx1:bx2]

    # Build a mask that selects only border pixels
    mask = np.ones(region.shape[:2], dtype=bool)
    inner_y1 = top - by1 + 2
    inner_y2 = bottom - by1 - 2
    inner_x1 = left - bx1 + 2
    inner_x2 = right - bx1 - 2
    if inner_y2 > inner_y1 and inner_x2 > inner_x1:
        mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

    border_pixels = region[mask]
    if len(border_pixels) == 0:
        border_pixels = region.reshape(-1, region.shape[-1])

    median = np.median(border_pixels, axis=0).astype(int)
    channels = median[:3].tolist()  # strip alpha if RGBA
    return tuple(channels)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

def _fit_font(text: str, box_w: int, box_h: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Return a PIL font sized to fill ~90% of the bounding box height,
    shrinking further if the text is too wide.
    """
    target_h = max(int(box_h * 0.90), 8)

    # Try to load a bundled truetype font; fall back to the default bitmap font
    try:
        font = ImageFont.truetype("arial.ttf", size=target_h)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=target_h)
        except (IOError, OSError):
            return ImageFont.load_default()

    # Shrink until text fits within box_w
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    size = target_h
    while size > 6:
        bbox_text = draw.textbbox((0, 0), text, font=font)
        tw = bbox_text[2] - bbox_text[0]
        if tw <= box_w:
            break
        size -= 1
        try:
            font = font.font_variant(size=size)  # type: ignore[attr-defined]
        except AttributeError:
            # Older Pillow versions don't have font_variant
            try:
                font = ImageFont.truetype(font.path, size=size)  # type: ignore[attr-defined]
            except Exception:
                break

    return font


def _estimate_text_color(img_array: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """
    Estimate the original text colour by looking at pixels *inside* the
    bounding rect that contrast most with the background.
    """
    left, top, right, bottom = rect
    region = img_array[top:bottom, left:right]
    if region.size == 0:
        return (0, 0, 0)

    flat = region.reshape(-1, region.shape[-1])[:, :3]
    bg = np.array(_estimate_bg_color(img_array, rect), dtype=float)

    # Pick the colour that differs most from bg (i.e. the ink colour)
    dists = np.linalg.norm(flat.astype(float) - bg, axis=1)
    text_pixel = flat[np.argmax(dists)]
    return tuple(int(v) for v in text_pixel)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def translate_image(image_bytes: bytes, src_lang: str = "en", tgt_lang: str = "es") -> bytes:
    """
    Full pipeline:
      1. OCR the image to find English text regions.
      2. Translate each region to Spanish.
      3. Erase original text (fill with estimated background colour).
      4. Redraw translated text in the same position.
      5. Return the modified image as PNG bytes.
    """
    # ── 1. Load image ────────────────────────────────────────────────────────
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)

    # ── 2. OCR ───────────────────────────────────────────────────────────────
    reader = _get_reader()
    results = reader.readtext(img_array, detail=1, paragraph=False)
    # results: [ ([bbox_points], text, confidence), … ]

    if not results:
        logger.info("No text detected — returning original image.")
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()

    logger.info(f"Detected {len(results)} text region(s).")

    # ── 3. Batch-translate all detected strings ───────────────────────────────
    translator = GoogleTranslator(source=src_lang, target=tgt_lang)

    regions: list[TextRegion] = []
    for (bbox, text, _conf) in results:
        text = text.strip()
        if not text:
            continue
        try:
            translated = translator.translate(text)
        except Exception as exc:
            logger.warning(f"Translation failed for '{text}': {exc}. Using original.")
            translated = text
        regions.append(TextRegion(bbox=bbox, original=text, translated=translated or text))
        logger.debug(f"  '{text}' → '{translated}'")

    # ── 4. Erase + redraw on a copy of the image ────────────────────────────
    result_img = img.copy()
    draw = ImageDraw.Draw(result_img)
    result_array = np.array(result_img)

    for region in regions:
        rect = _bbox_to_rect(region.bbox)
        left, top, right, bottom = rect
        box_w = max(right - left, 1)
        box_h = max(bottom - top, 1)

        # Estimate colours from the *original* array (before we start overwriting)
        bg_color = _estimate_bg_color(img_array, rect)
        text_color = _estimate_text_color(img_array, rect)

        # Erase: fill bounding rect with background colour (with 2px padding)
        draw.rectangle(
            [left - 2, top - 2, right + 2, bottom + 2],
            fill=bg_color,
        )

        # Choose font size
        font = _fit_font(region.translated, box_w, box_h)

        # Centre the translated text within the box
        dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        tbbox = dummy_draw.textbbox((0, 0), region.translated, font=font)
        tw = tbbox[2] - tbbox[0]
        th = tbbox[3] - tbbox[1]
        tx = left + (box_w - tw) // 2
        ty = top + (box_h - th) // 2

        draw.text((tx, ty), region.translated, fill=text_color, font=font)

    # ── 5. Serialise to PNG bytes ─────────────────────────────────────────────
    out = io.BytesIO()
    result_img.save(out, format="PNG")
    return out.getvalue()