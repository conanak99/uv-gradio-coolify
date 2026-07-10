import concurrent.futures
import logging
import queue
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlparse

from clients import fal as fal_client
from clients import nano_gpt as nano_gpt_client
import gradio as gr

from history import (
    GalleryItem,
    add_history_entry,
    get_history,
    history_view,
    unchanged_history_view,
)
from image_utils import download_image_url


logger = logging.getLogger(__name__)

SEEDREAM_PRO_EDIT_MODEL = "Seedream 5.0 Pro Edit (NanoGPT)"
WAN_26_EDIT_MODEL = "WAN 2.6 Image Edit (NanoGPT)"

type ImageReference = str
type ModelName = str
type ProgressCallback = Callable[[list[GalleryItem]], None]
type CompletedJob = tuple[str, list[GalleryItem]]


MODEL_MAP: dict[ModelName, str] = {
    "Qwen Image Edit": "fal-ai/qwen-image-edit-2511",
    "FLUX.2 Klein 9B Edit": "fal-ai/flux-2/klein/9b/edit",
    "Grok Imagine Image Edit": "xai/grok-imagine-image/edit",
    SEEDREAM_PRO_EDIT_MODEL: nano_gpt_client.SEEDREAM_PRO_EDIT_MODEL_ID,
    WAN_26_EDIT_MODEL: nano_gpt_client.WAN_26_EDIT_MODEL_ID,
}
NANO_GPT_MODELS = {SEEDREAM_PRO_EDIT_MODEL, WAN_26_EDIT_MODEL}

GROK_GENERATE_MODEL = "Grok Imagine (fal.ai)"
SEEDREAM_LITE_GENERATE_MODEL = "Seedream 5.0 Lite (NanoGPT)"
SEEDREAM_PRO_GENERATE_MODEL = "Seedream 5.0 Pro (NanoGPT)"
GENERATE_MODEL_MAP: dict[ModelName, str] = {
    GROK_GENERATE_MODEL: fal_client.GROK_GENERATE_MODEL_ID,
    SEEDREAM_LITE_GENERATE_MODEL: "seedream-v5.0-lite",
    SEEDREAM_PRO_GENERATE_MODEL: "bytedance/seedream-v5.0-pro",
}
GENERATE_ASPECT_RATIOS = ["16:9", "4:3", "1:1", "3:4", "9:16"]
SEEDREAM_LITE_RATIO_MAP = {
    "16:9": "16:9",
    "4:3": "3:2",
    "1:1": "1:1",
    "3:4": "2:3",
    "9:16": "9:16",
}
SEEDREAM_LITE_RESOLUTIONS = {
    "1:1": "2048x2048",
    "16:9": "2560x1440",
    "9:16": "1440x2560",
    "3:2": "3072x2048",
    "2:3": "2048x3072",
}

JOB_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)
JOB_HEARTBEAT_SECONDS = 1.0


def normalize_image_url(image_url: str | None) -> str | None:
    if not image_url:
        return None

    normalized_url = image_url.strip()
    if not normalized_url:
        return None

    parsed_url = urlparse(normalized_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise gr.Error("Please provide a valid http(s) image URL.")
    return normalized_url


def select_edit_image_input(
    image_path: str | None,
    image_url: str | None = None,
) -> ImageReference:
    normalized_url = normalize_image_url(image_url)
    if normalized_url:
        return normalized_url
    if image_path:
        return image_path
    raise gr.Error("Please provide an input image or image URL.")


def is_remote_image_reference(image_reference: ImageReference) -> bool:
    return urlparse(image_reference).scheme in {"http", "https"}


def elapsed_button_label(action: str, started_at: float) -> str:
    elapsed_seconds = max(0, int(time.monotonic() - started_at))
    minutes, seconds = divmod(elapsed_seconds, 60)
    if minutes:
        return f"{action}... {minutes}m {seconds:02d}s"
    return f"{action}... {seconds}s"


def run_model(
    model_name: ModelName, image_reference: ImageReference, prompt: str
) -> GalleryItem | None:
    model_id = MODEL_MAP[model_name]
    if model_name in NANO_GPT_MODELS:
        image_url = nano_gpt_client.edit_image(model_id, image_reference, prompt)
    else:
        image_url = fal_client.edit_image(model_id, image_reference, prompt)
    return (image_url, model_name) if image_url else None


def edit_image(
    image_path: str | None,
    prompt: str,
    models: list[ModelName],
    progress_callback: ProgressCallback | None = None,
    image_url: str | None = None,
) -> list[GalleryItem]:
    image_reference = select_edit_image_input(image_path, image_url)
    if not prompt:
        raise gr.Error("Please provide a prompt.")
    if not models:
        raise gr.Error("Please select at least one model.")

    image_references: dict[ModelName, ImageReference] = {}
    if any(model not in NANO_GPT_MODELS for model in models):
        fal_image_url = (
            image_reference
            if is_remote_image_reference(image_reference)
            else fal_client.upload_image(image_reference)
        )
        image_references.update(
            (model, fal_image_url)
            for model in models
            if model not in NANO_GPT_MODELS
        )
    if any(model in NANO_GPT_MODELS for model in models):
        nano_image_path = (
            download_image_url(image_reference)
            if is_remote_image_reference(image_reference)
            else image_reference
        )
        image_references.update(
            (model, nano_image_path) for model in models if model in NANO_GPT_MODELS
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures: dict[concurrent.futures.Future[GalleryItem | None], ModelName] = {
            executor.submit(run_model, name, image_references[name], prompt): name
            for name in models
        }
        results: list[GalleryItem] = []
        errors: list[str] = []
        for future in concurrent.futures.as_completed(futures):
            model_name = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
                logger.warning(
                    "model response operation=edit model=%s status=error error_type=%s",
                    model_name,
                    type(exc).__name__,
                )
                continue
            if result:
                results.append(result)
                if progress_callback:
                    progress_callback(list(results))

    if not results:
        detail = "; ".join(errors) if errors else "no images returned"
        raise gr.Error(f"All model calls failed ({detail}).")
    if errors:
        gr.Warning(f"Some models failed: {'; '.join(errors)}")
    return results


def run_generate_model(
    prompt: str,
    model_name: ModelName,
    aspect_ratio: str,
    num_images: int,
) -> list[GalleryItem]:
    output_ratio = aspect_ratio
    if model_name == GROK_GENERATE_MODEL:
        image_urls = fal_client.generate_images(prompt, aspect_ratio, num_images)
    else:
        if model_name == SEEDREAM_LITE_GENERATE_MODEL:
            output_ratio = SEEDREAM_LITE_RATIO_MAP[aspect_ratio]
            resolution = SEEDREAM_LITE_RESOLUTIONS[output_ratio]
        else:
            resolution = aspect_ratio
        image_urls = nano_gpt_client.generate_images(
            GENERATE_MODEL_MAP[model_name],
            prompt,
            resolution,
            num_images,
        )

    ratio_label = (
        aspect_ratio
        if output_ratio == aspect_ratio
        else f"{aspect_ratio} → {output_ratio}"
    )
    return [(url, f"{model_name} ({ratio_label})") for url in image_urls]


def generate_image(
    prompt: str,
    models: list[ModelName],
    aspect_ratio: str,
    num_images: int,
    progress_callback: ProgressCallback | None = None,
) -> list[GalleryItem]:
    if not prompt:
        raise gr.Error("Please provide a prompt.")
    if not models:
        raise gr.Error("Please select at least one generation model.")
    if any(model not in GENERATE_MODEL_MAP for model in models):
        raise gr.Error("Please select valid generation models.")
    if aspect_ratio not in GENERATE_ASPECT_RATIOS:
        raise gr.Error("Please select a valid aspect ratio.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures: dict[
            concurrent.futures.Future[list[GalleryItem]], ModelName
        ] = {
            executor.submit(
                run_generate_model,
                prompt,
                model_name,
                aspect_ratio,
                num_images,
            ): model_name
            for model_name in models
        }
        results: list[GalleryItem] = []
        errors: list[str] = []
        for future in concurrent.futures.as_completed(futures):
            model_name = futures[future]
            try:
                results.extend(future.result())
                if progress_callback:
                    progress_callback(list(results))
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
                logger.warning(
                    "model response operation=generate model=%s status=error error_type=%s",
                    model_name,
                    type(exc).__name__,
                )

    if not results:
        detail = "; ".join(errors) if errors else "no images returned"
        raise gr.Error(f"All generation model calls failed ({detail}).")
    if errors:
        gr.Warning(f"Some generation models failed: {'; '.join(errors)}")
    return results


def iter_job_progress(
    future: concurrent.futures.Future[Any],
    progress_queue: queue.Queue[list[GalleryItem]],
    heartbeat_seconds: float = JOB_HEARTBEAT_SECONDS,
) -> Iterator[list[GalleryItem] | None]:
    next_heartbeat = time.monotonic() + heartbeat_seconds
    while not future.done() or not progress_queue.empty():
        timeout = max(0.0, min(0.1, next_heartbeat - time.monotonic()))
        try:
            results = progress_queue.get(timeout=timeout)
        except queue.Empty:
            if time.monotonic() >= next_heartbeat:
                yield None
                next_heartbeat = time.monotonic() + heartbeat_seconds
            continue
        yield results
        next_heartbeat = time.monotonic() + heartbeat_seconds


def run_edit_job(
    image_path: str | None,
    image_url: str | None,
    prompt: str,
    models: list[ModelName],
    progress_queue: queue.Queue[list[GalleryItem]],
) -> CompletedJob:
    started_at = time.monotonic()
    image_reference = select_edit_image_input(image_path, image_url)
    logger.info(
        "job request operation=edit models=%d prompt_chars=%d",
        len(models),
        len(prompt),
    )
    try:
        results = edit_image(
            image_path,
            prompt,
            models,
            progress_queue.put,
            image_url=image_url,
        )
    except Exception as exc:
        logger.error(
            "job response operation=edit status=error duration_ms=%d error_type=%s",
            int((time.monotonic() - started_at) * 1000),
            type(exc).__name__,
        )
        raise
    entry_id = add_history_entry(
        operation="Edit",
        prompt=prompt,
        input_image=image_reference,
        outputs=results,
        settings=f'Models: {", ".join(models)}',
    )
    logger.info(
        "job response operation=edit status=success duration_ms=%d images=%d history_id=%s",
        int((time.monotonic() - started_at) * 1000),
        len(results),
        entry_id,
    )
    return entry_id, results


def edit_image_flow(
    image_path: str | None,
    image_url: str | None,
    prompt: str,
    models: list[ModelName],
):
    started_at = time.monotonic()
    progress_queue: queue.Queue[list[GalleryItem]] = queue.Queue()
    future = JOB_EXECUTOR.submit(
        run_edit_job,
        image_path,
        image_url,
        prompt,
        models,
        progress_queue,
    )
    yield (
        gr.update(),
        gr.update(
            interactive=False,
            value=elapsed_button_label("Editing", started_at),
        ),
        *unchanged_history_view(),
    )
    for partial_results in iter_job_progress(future, progress_queue):
        yield (
            partial_results if partial_results is not None else gr.update(),
            gr.update(
                interactive=False,
                value=elapsed_button_label("Editing", started_at),
            ),
            *unchanged_history_view(),
        )
    try:
        entry_id, results = future.result()
    except BaseException:
        yield (
            gr.update(),
            gr.update(interactive=True, value="Edit Image"),
            *unchanged_history_view(),
        )
        raise
    yield (
        results,
        gr.update(interactive=True, value="Edit Image"),
        *history_view(get_history(), entry_id),
    )


def run_generate_job(
    prompt: str,
    models: list[ModelName],
    aspect_ratio: str,
    num_images: int,
    progress_queue: queue.Queue[list[GalleryItem]],
) -> CompletedJob:
    started_at = time.monotonic()
    logger.info(
        "job request operation=generate models=%d prompt_chars=%d outputs_per_model=%d aspect_ratio=%s",
        len(models),
        len(prompt),
        num_images,
        aspect_ratio,
    )
    try:
        results = generate_image(
            prompt,
            models,
            aspect_ratio,
            num_images,
            progress_queue.put,
        )
    except Exception as exc:
        logger.error(
            "job response operation=generate status=error duration_ms=%d error_type=%s",
            int((time.monotonic() - started_at) * 1000),
            type(exc).__name__,
        )
        raise
    entry_id = add_history_entry(
        operation="Generate",
        prompt=prompt,
        input_image=None,
        outputs=results,
        settings=(
            f'Models: {", ".join(models)} · Aspect ratio: {aspect_ratio} · '
            f"Images: {num_images}"
        ),
    )
    logger.info(
        "job response operation=generate status=success duration_ms=%d images=%d history_id=%s",
        int((time.monotonic() - started_at) * 1000),
        len(results),
        entry_id,
    )
    return entry_id, results


def generate_image_flow(
    prompt: str,
    models: list[ModelName],
    aspect_ratio: str,
    num_images: str,
):
    started_at = time.monotonic()
    progress_queue: queue.Queue[list[GalleryItem]] = queue.Queue()
    future = JOB_EXECUTOR.submit(
        run_generate_job,
        prompt,
        models,
        aspect_ratio,
        int(num_images),
        progress_queue,
    )
    yield (
        gr.update(),
        gr.update(
            interactive=False,
            value=elapsed_button_label("Generating", started_at),
        ),
        *unchanged_history_view(),
    )
    for partial_results in iter_job_progress(future, progress_queue):
        yield (
            partial_results if partial_results is not None else gr.update(),
            gr.update(
                interactive=False,
                value=elapsed_button_label("Generating", started_at),
            ),
            *unchanged_history_view(),
        )
    try:
        entry_id, results = future.result()
    except BaseException:
        yield (
            gr.update(),
            gr.update(interactive=True, value="Generate"),
            *unchanged_history_view(),
        )
        raise
    yield (
        results,
        gr.update(interactive=True, value="Generate"),
        *history_view(get_history(), entry_id),
    )
