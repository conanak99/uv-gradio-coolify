from typing import Any

import fal_client

from image_utils import resize_if_needed


GROK_GENERATE_MODEL_ID = "xai/grok-imagine-image"


def upload_image(image_path: str) -> str:
    return fal_client.upload_file(resize_if_needed(image_path))


def edit_image(model_id: str, image_url: str, prompt: str) -> str | None:
    arguments: dict[str, Any] = {
        "prompt": prompt,
        "image_urls": [image_url],
        "sync_mode": False,
        "num_images": 1,
    }
    # Grok Imagine doesn't accept `enable_safety_checker`.
    if not model_id.startswith("xai/"):
        arguments["enable_safety_checker"] = False

    result: dict[str, Any] = fal_client.subscribe(
        model_id,
        arguments=arguments,
    )
    images: list[dict[str, Any]] = result.get("images", [])
    return images[0]["url"] if images else None


def generate_images(prompt: str, aspect_ratio: str, num_images: int) -> list[str]:
    result: dict[str, Any] = fal_client.subscribe(
        GROK_GENERATE_MODEL_ID,
        arguments={
            "prompt": prompt,
            "num_images": num_images,
            "aspect_ratio": aspect_ratio,
            "resolution": "1k",
            "output_format": "jpeg",
            "sync_mode": False,
        },
    )
    return [image["url"] for image in result.get("images", [])]
