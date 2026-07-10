import base64
import binascii
import hashlib
from io import BytesIO
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

from image_utils import prepare_for_aspect_ratio, resize_if_needed


logger = logging.getLogger(__name__)

IMAGES_URL = "https://nano-gpt.com/api/v1/images"
IMAGE_EDITS_URL = "https://nano-gpt.com/api/v1/images/edits"
SEEDREAM_PRO_EDIT_MODEL_ID = "bytedance/seedream-v5.0-pro/edit"
SEEDREAM_PRO_EDIT_ASPECT_RATIOS = (
    "1:1",
    "16:9",
    "9:16",
    "3:2",
    "2:3",
    "4:3",
    "3:4",
)
# NanoGPT's endpoint runs behind a Vercel function with a 4.5 MB body limit.
# Base64 expands binary data by roughly one third, so leave room for JSON.
MAX_FUNCTION_BODY_BYTES = 4_500_000
MAX_DATA_URL_IMAGE_BYTES = 3_000_000
JPEG_QUALITY = 85
OUTPUT_CACHE = tempfile.TemporaryDirectory(prefix="image-studio-nanogpt-")
OUTPUT_CACHE_PATH = Path(OUTPUT_CACHE.name)
OUTPUT_CACHE_LOCK = threading.Lock()


def _jpeg_compatible_image(image_path: str) -> Image.Image:
    with Image.open(image_path) as source:
        oriented = ImageOps.exif_transpose(source)
        try:
            has_alpha = oriented.mode in {"RGBA", "LA"} or (
                oriented.mode == "P" and "transparency" in oriented.info
            )
            if not has_alpha:
                return oriented.convert("RGB")

            rgba = oriented.convert("RGBA")
            try:
                flattened = Image.new("RGB", rgba.size, "white")
                flattened.paste(rgba, mask=rgba.getchannel("A"))
                return flattened
            finally:
                rgba.close()
        finally:
            if oriented is not source:
                oriented.close()


def _compressed_jpeg_bytes(image_path: str, max_bytes: int) -> bytes:
    image = _jpeg_compatible_image(image_path)
    try:
        for _ in range(8):
            output = BytesIO()
            image.save(
                output,
                format="JPEG",
                quality=JPEG_QUALITY,
                optimize=True,
            )
            image_bytes = output.getvalue()
            if len(image_bytes) <= max_bytes:
                return image_bytes

            scale = min(
                0.9,
                math.sqrt(max_bytes / len(image_bytes)) * 0.95,
            )
            new_size = (
                max(64, int(image.width * scale)),
                max(64, int(image.height * scale)),
            )
            if new_size == image.size:
                break
            resized = image.resize(new_size, Image.LANCZOS)
            image.close()
            image = resized
    finally:
        image.close()

    raise RuntimeError("Could not compress NanoGPT input below the request limit.")


def image_to_data_url(image_path: str) -> str:
    image_path = resize_if_needed(image_path)
    with Image.open(image_path) as image:
        mime_type = Image.MIME.get(image.format or "", "image/png")
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    original_bytes = len(image_bytes)
    if original_bytes > MAX_DATA_URL_IMAGE_BYTES:
        image_bytes = _compressed_jpeg_bytes(
            image_path,
            MAX_DATA_URL_IMAGE_BYTES,
        )
        mime_type = "image/jpeg"
        logger.info(
            "input compressed original_bytes=%d compressed_bytes=%d",
            original_bytes,
            len(image_bytes),
        )

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _api_key() -> str:
    api_key = os.environ.get("NANO_GPT_KEY")
    if not api_key:
        raise RuntimeError("NANO_GPT_KEY is not configured.")
    return api_key


def _error_message(detail: str) -> str:
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        return "unparseable_error_response"
    error = payload.get("error", "unknown_error")
    if isinstance(error, dict):
        error = error.get("message") or error.get("code") or "unknown_error"
    return str(error)[:500]


def _cache_base64_image(image_base64: str) -> str:
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError("NanoGPT returned invalid base64 image data.") from exc

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = image.format
            image.verify()
    except (OSError, ValueError) as exc:
        raise RuntimeError("NanoGPT returned invalid image data.") from exc

    suffix = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
    }.get(image_format or "", ".img")
    digest = hashlib.sha256(image_bytes).hexdigest()
    image_path = OUTPUT_CACHE_PATH / f"{digest}{suffix}"
    with OUTPUT_CACHE_LOCK:
        if not image_path.exists():
            image_path.write_bytes(image_bytes)
    return str(image_path)


def _request_images(
    payload: dict[str, Any],
    api_key: str,
    endpoint: str = IMAGES_URL,
) -> list[str]:
    started_at = time.monotonic()
    input_count = (
        1
        if payload.get("imageDataUrl")
        else len(payload.get("input_references", []))
    )
    logger.info(
        "request endpoint=%s model=%s prompt_chars=%d inputs=%d outputs=%s resolution=%s",
        endpoint,
        payload.get("model"),
        len(payload.get("prompt", "")),
        input_count,
        payload.get("n"),
        payload.get("resolution"),
    )
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            status = getattr(response, "status", 200)
            result: dict[str, Any] = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.error(
            "response endpoint=%s model=%s status=%d duration_ms=%d error=%s",
            endpoint,
            payload.get("model"),
            exc.code,
            int((time.monotonic() - started_at) * 1000),
            _error_message(detail),
        )
        raise RuntimeError(f"NanoGPT request failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        logger.error(
            "response endpoint=%s model=%s status=network_error duration_ms=%d error=%s",
            endpoint,
            payload.get("model"),
            int((time.monotonic() - started_at) * 1000),
            exc.reason,
        )
        raise RuntimeError(f"NanoGPT request failed: {exc.reason}") from exc

    images: list[dict[str, Any]] = result.get("data", [])
    image_urls: list[str] = []
    for image in images:
        if image_url := image.get("url"):
            image_urls.append(image_url)
        elif image_base64 := image.get("b64_json"):
            image_urls.append(_cache_base64_image(image_base64))
    logger.info(
        "response endpoint=%s model=%s status=%s duration_ms=%d images=%d",
        endpoint,
        payload.get("model"),
        status,
        int((time.monotonic() - started_at) * 1000),
        len(image_urls),
    )
    return image_urls


def edit_image(model_id: str, image_path: str, prompt: str) -> str | None:
    api_key = _api_key()
    prepared_path = image_path
    payload: dict[str, Any] = {
        "model": model_id,
        "prompt": prompt,
        "n": 1,
    }
    if model_id == SEEDREAM_PRO_EDIT_MODEL_ID:
        prepared_path, aspect_ratio = prepare_for_aspect_ratio(
            image_path,
            SEEDREAM_PRO_EDIT_ASPECT_RATIOS,
        )
        payload["size"] = aspect_ratio
    payload["imageDataUrl"] = image_to_data_url(prepared_path)
    image_urls = _request_images(payload, api_key, IMAGE_EDITS_URL)
    return image_urls[0] if image_urls else None


def generate_images(
    model_id: str,
    prompt: str,
    resolution: str,
    num_images: int,
) -> list[str]:
    return _request_images(
        {
            "model": model_id,
            "prompt": prompt,
            "resolution": resolution,
            "n": num_images,
        },
        _api_key(),
    )
