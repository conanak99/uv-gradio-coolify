import math
import os
import tempfile
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


MAX_DIMENSION = 2048
MAX_REMOTE_IMAGE_BYTES = 25 * 1024 * 1024


def aspect_ratio_value(aspect_ratio: str) -> float:
    width, height = aspect_ratio.split(":", 1)
    return int(width) / int(height)


def closest_aspect_ratio(
    image_path: str,
    supported_aspect_ratios: tuple[str, ...],
) -> str:
    with Image.open(image_path) as image:
        source_ratio = image.width / image.height
    return min(
        supported_aspect_ratios,
        key=lambda ratio: abs(
            math.log(source_ratio / aspect_ratio_value(ratio))
        ),
    )


def crop_to_aspect_ratio(image_path: str, aspect_ratio: str) -> str:
    target_ratio = aspect_ratio_value(aspect_ratio)
    with Image.open(image_path) as image:
        source_ratio = image.width / image.height
        if math.isclose(source_ratio, target_ratio, rel_tol=1e-3):
            return image_path

        if source_ratio > target_ratio:
            cropped_width = round(image.height * target_ratio)
            left = (image.width - cropped_width) // 2
            crop_box = (left, 0, left + cropped_width, image.height)
        else:
            cropped_height = round(image.width / target_ratio)
            top = (image.height - cropped_height) // 2
            crop_box = (0, top, image.width, top + cropped_height)
        cropped_image = image.crop(crop_box)

    ratio_slug = aspect_ratio.replace(":", "x")
    cropped_path = f"{image_path}_cropped_{ratio_slug}.png"
    try:
        cropped_image.save(cropped_path)
    finally:
        cropped_image.close()
    return cropped_path


def prepare_for_aspect_ratio(
    image_path: str,
    supported_aspect_ratios: tuple[str, ...],
) -> tuple[str, str]:
    aspect_ratio = closest_aspect_ratio(image_path, supported_aspect_ratios)
    return crop_to_aspect_ratio(image_path, aspect_ratio), aspect_ratio


def resize_if_needed(image_path: str) -> str:
    with Image.open(image_path) as image:
        width, height = image.size
        if width <= MAX_DIMENSION and height <= MAX_DIMENSION:
            return image_path
        scale = MAX_DIMENSION / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        resized_image = image.resize(new_size, Image.LANCZOS)

    resized_path = image_path + "_resized.png"
    try:
        resized_image.save(resized_path)
    finally:
        resized_image.close()
    return resized_path


def _image_suffix(image_format: str | None) -> str:
    return {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "GIF": ".gif",
    }.get(image_format or "", ".img")


def download_image_url(image_url: str) -> str:
    request = Request(
        image_url,
        headers={"User-Agent": "Image Studio/1.0"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            image_bytes = response.read(MAX_REMOTE_IMAGE_BYTES + 1)
    except HTTPError as exc:
        raise RuntimeError(
            f"Could not download input image URL ({exc.code})."
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not download input image URL: {exc.reason}."
        ) from exc

    if len(image_bytes) > MAX_REMOTE_IMAGE_BYTES:
        raise RuntimeError("Input image URL is larger than the download limit.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = image.format
            image.verify()
    except (OSError, ValueError) as exc:
        raise RuntimeError("Input image URL did not return a valid image.") from exc

    file_descriptor, image_path = tempfile.mkstemp(
        prefix="image-studio-url-",
        suffix=_image_suffix(image_format),
    )
    with os.fdopen(file_descriptor, "wb") as image_file:
        image_file.write(image_bytes)
    return image_path
