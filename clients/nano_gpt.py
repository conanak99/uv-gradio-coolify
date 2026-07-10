import base64
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from image_utils import resize_if_needed


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


def _request_images(
    payload: dict[str, Any],
    api_key: str,
    endpoint: str = IMAGES_URL,
) -> list[str]:
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
            result: dict[str, Any] = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NanoGPT request failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"NanoGPT request failed: {exc.reason}") from exc

    images: list[dict[str, Any]] = result.get("data", [])
    image_urls: list[str] = []
    for image in images:
        if image_url := image.get("url"):
            image_urls.append(image_url)
        elif image_base64 := image.get("b64_json"):
            image_urls.append(f"data:image/png;base64,{image_base64}")
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
