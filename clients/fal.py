import logging
import time
from typing import Any

import fal_client

from image_utils import resize_if_needed
from model_catalog import GROK_GENERATE_MODEL_ID


logger = logging.getLogger(__name__)


def upload_image(image_path: str) -> str:
    prepared_path = resize_if_needed(image_path)
    started_at = time.monotonic()
    logger.info("upload request")
    try:
        image_url = fal_client.upload_file(prepared_path)
    except Exception as exc:
        logger.error(
            "upload response status=error duration_ms=%d error_type=%s",
            int((time.monotonic() - started_at) * 1000),
            type(exc).__name__,
        )
        raise
    logger.info(
        "upload response status=success duration_ms=%d",
        int((time.monotonic() - started_at) * 1000),
    )
    return image_url


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

    started_at = time.monotonic()
    logger.info(
        "request operation=edit model=%s prompt_chars=%d inputs=1 outputs=1",
        model_id,
        len(prompt),
    )
    try:
        result: dict[str, Any] = fal_client.subscribe(
            model_id,
            arguments=arguments,
        )
    except Exception as exc:
        logger.error(
            "response operation=edit model=%s status=error duration_ms=%d error_type=%s",
            model_id,
            int((time.monotonic() - started_at) * 1000),
            type(exc).__name__,
        )
        raise
    images: list[dict[str, Any]] = result.get("images", [])
    logger.info(
        "response operation=edit model=%s status=success duration_ms=%d images=%d",
        model_id,
        int((time.monotonic() - started_at) * 1000),
        len(images),
    )
    return images[0]["url"] if images else None


def generate_images(
    model_id: str,
    prompt: str,
    aspect_ratio: str,
    num_images: int,
) -> list[str]:
    started_at = time.monotonic()
    logger.info(
        "request operation=generate model=%s prompt_chars=%d outputs=%d aspect_ratio=%s",
        model_id,
        len(prompt),
        num_images,
        aspect_ratio,
    )
    try:
        result: dict[str, Any] = fal_client.subscribe(
            model_id,
            arguments={
                "prompt": prompt,
                "num_images": num_images,
                "aspect_ratio": aspect_ratio,
                "resolution": "1k",
                "output_format": "jpeg",
                "sync_mode": False,
            },
        )
    except Exception as exc:
        logger.error(
            "response operation=generate model=%s status=error duration_ms=%d error_type=%s",
            model_id,
            int((time.monotonic() - started_at) * 1000),
            type(exc).__name__,
        )
        raise
    images = [image["url"] for image in result.get("images", [])]
    logger.info(
        "response operation=generate model=%s status=success duration_ms=%d images=%d",
        model_id,
        int((time.monotonic() - started_at) * 1000),
        len(images),
    )
    return images
