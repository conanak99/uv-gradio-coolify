import base64
import json
import logging
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from image_utils import resize_if_needed


logger = logging.getLogger(__name__)

IMAGES_URL = "https://nano-gpt.com/api/v1/images"
IMAGE_EDITS_URL = "https://nano-gpt.com/api/v1/images/edits"
MAX_INPUT_BYTES = 10 * 1024 * 1024


def image_to_data_url(image_path: str) -> str:
    image_path = resize_if_needed(image_path)
    with Image.open(image_path) as image:
        mime_type = Image.MIME.get(image.format or "", "image/png")
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    if len(image_bytes) > MAX_INPUT_BYTES:
        raise RuntimeError("NanoGPT input image must be 10 MB or smaller after resizing.")

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
            image_urls.append(f"data:image/png;base64,{image_base64}")
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
    payload = {
        "model": model_id,
        "prompt": prompt,
        "imageDataUrl": image_to_data_url(image_path),
        "n": 1,
    }
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
