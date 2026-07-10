import concurrent.futures
import logging
import queue
import time
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

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
from image_utils import (
    download_image_url,
    normalize_http_url,
    prepare_for_aspect_ratio,
)
from model_catalog import (
    EDIT_MODELS,
    GENERATE_ASPECT_RATIOS,
    GENERATE_MODELS,
    GROK_GENERATE_MODEL,
    Operation,
    Provider,
    SEEDREAM_LITE_GENERATE_MODEL,
    SEEDREAM_PRO_EDIT_MODEL,
    SEEDREAM_PRO_GENERATE_MODEL,
    WAN_26_EDIT_MODEL,
    WAN_27_GENERATE_MODEL,
    WAN_27_PRO_GENERATE_MODEL,
)


logger = logging.getLogger(__name__)

type ImageReference = str
type ModelName = str
type ProgressCallback = Callable[[list[GalleryItem]], None]
type CompletedJob = tuple[str, list[GalleryItem]]
type ModelCall = Callable[[], GalleryItem | list[GalleryItem] | None]

# Keep these simple name-to-ID maps for UI choices and backwards compatibility.
MODEL_MAP: dict[ModelName, str] = {
    name: model.api_id for name, model in EDIT_MODELS.items()
}
GENERATE_MODEL_MAP: dict[ModelName, str] = {
    name: model.api_id for name, model in GENERATE_MODELS.items()
}
NANO_GPT_MODELS = {
    name
    for name, model in EDIT_MODELS.items()
    if model.provider is Provider.NANO_GPT
}

JOB_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)
JOB_HEARTBEAT_SECONDS = 1.0


def normalize_image_url(image_url: str | None) -> str | None:
    if not image_url:
        return None

    if not image_url.strip():
        return None

    normalized_url = normalize_http_url(image_url)
    if not normalized_url:
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
    return normalize_http_url(image_reference) is not None


def elapsed_button_label(action: str, started_at: float) -> str:
    elapsed_seconds = max(0, int(time.monotonic() - started_at))
    minutes, seconds = divmod(elapsed_seconds, 60)
    if minutes:
        return f"{action}... {minutes}m {seconds:02d}s"
    return f"{action}... {seconds}s"


def run_model(
    model_name: ModelName, image_reference: ImageReference, prompt: str
) -> GalleryItem | None:
    model = EDIT_MODELS[model_name]
    if model.provider is Provider.FAL:
        image_url = fal_client.edit_image(model.api_id, image_reference, prompt)
    elif model.provider is Provider.NANO_GPT:
        prepared_reference = image_reference
        size = None
        if model.edit_aspect_ratios:
            prepared_reference, size = prepare_for_aspect_ratio(
                image_reference,
                model.edit_aspect_ratios,
            )
        image_url = nano_gpt_client.edit_image(
            model.api_id,
            prepared_reference,
            prompt,
            **({"size": size} if size else {}),
        )
    else:
        raise RuntimeError(f"Unsupported provider: {model.provider.value}")
    return (image_url, model_name) if image_url else None


def _prepare_edit_references(
    image_reference: ImageReference,
    models: list[ModelName],
) -> dict[Provider, ImageReference]:
    providers = {EDIT_MODELS[name].provider for name in models}
    references: dict[Provider, ImageReference] = {}
    if Provider.FAL in providers:
        references[Provider.FAL] = (
            image_reference
            if is_remote_image_reference(image_reference)
            else fal_client.upload_image(image_reference)
        )
    if Provider.NANO_GPT in providers:
        references[Provider.NANO_GPT] = (
            download_image_url(image_reference)
            if is_remote_image_reference(image_reference)
            else image_reference
        )
    return references


def _result_items(
    result: GalleryItem | list[GalleryItem] | None,
) -> list[GalleryItem]:
    if result is None:
        return []
    if isinstance(result, list):
        return result
    return [result]


def run_parallel_model_calls(
    operation: Operation,
    calls: dict[ModelName, ModelCall],
    progress_callback: ProgressCallback | None = None,
) -> list[GalleryItem]:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(model_call): model_name
            for model_name, model_call in calls.items()
        }
        results: list[GalleryItem] = []
        errors: list[str] = []
        for future in concurrent.futures.as_completed(futures):
            model_name = futures[future]
            try:
                new_results = _result_items(future.result())
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
                logger.warning(
                    "model response operation=%s model=%s status=error error_type=%s",
                    operation.log_name,
                    model_name,
                    type(exc).__name__,
                )
                continue
            if new_results:
                results.extend(new_results)
                if progress_callback:
                    progress_callback(list(results))

    model_description = (
        "generation model" if operation is Operation.GENERATE else "model"
    )
    if not results:
        detail = "; ".join(errors) if errors else "no images returned"
        raise gr.Error(f"All {model_description} calls failed ({detail}).")
    if errors:
        gr.Warning(f"Some {model_description}s failed: {'; '.join(errors)}")
    return results


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
    if any(model not in EDIT_MODELS for model in models):
        raise gr.Error("Please select valid edit models.")

    references = _prepare_edit_references(image_reference, models)
    calls = {
        model_name: partial(
            run_model,
            model_name,
            references[EDIT_MODELS[model_name].provider],
            prompt,
        )
        for model_name in models
    }
    return run_parallel_model_calls(Operation.EDIT, calls, progress_callback)


def run_generate_model(
    prompt: str,
    model_name: ModelName,
    aspect_ratio: str,
    num_images: int,
) -> list[GalleryItem]:
    model = GENERATE_MODELS[model_name]
    output_ratio, resolution = model.generation_parameters(aspect_ratio)
    if model.provider is Provider.FAL:
        image_urls = fal_client.generate_images(
            model.api_id,
            prompt,
            output_ratio,
            num_images,
        )
    elif model.provider is Provider.NANO_GPT:
        image_urls = nano_gpt_client.generate_images(
            model.api_id,
            prompt,
            resolution,
            num_images,
        )
    else:
        raise RuntimeError(f"Unsupported provider: {model.provider.value}")

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

    calls = {
        model_name: partial(
            run_generate_model,
            prompt,
            model_name,
            aspect_ratio,
            num_images,
        )
        for model_name in models
    }
    return run_parallel_model_calls(Operation.GENERATE, calls, progress_callback)


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


def run_history_job(
    operation: Operation,
    prompt: str,
    models: list[ModelName],
    input_image: str | None,
    settings: str,
    model_calls: Callable[[], list[GalleryItem]],
) -> CompletedJob:
    started_at = time.monotonic()
    logger.info(
        "job request operation=%s models=%d prompt_chars=%d",
        operation.log_name,
        len(models),
        len(prompt),
    )
    try:
        results = model_calls()
    except Exception as exc:
        logger.error(
            "job response operation=%s status=error duration_ms=%d error_type=%s",
            operation.log_name,
            int((time.monotonic() - started_at) * 1000),
            type(exc).__name__,
        )
        raise
    entry_id = add_history_entry(
        operation=operation.value,
        prompt=prompt,
        input_image=input_image,
        outputs=results,
        settings=settings,
    )
    logger.info(
        "job response operation=%s status=success duration_ms=%d images=%d history_id=%s",
        operation.log_name,
        int((time.monotonic() - started_at) * 1000),
        len(results),
        entry_id,
    )
    return entry_id, results


def run_edit_job(
    image_path: str | None,
    image_url: str | None,
    prompt: str,
    models: list[ModelName],
    progress_queue: queue.Queue[list[GalleryItem]],
) -> CompletedJob:
    image_reference = select_edit_image_input(image_path, image_url)
    return run_history_job(
        Operation.EDIT,
        prompt,
        models,
        image_reference,
        f'Models: {", ".join(models)}',
        partial(
            edit_image,
            image_reference,
            prompt,
            models,
            progress_queue.put,
        ),
    )


def stream_job(
    future: concurrent.futures.Future[CompletedJob],
    progress_queue: queue.Queue[list[GalleryItem]],
    action: str,
    idle_button_label: str,
):
    started_at = time.monotonic()
    yield (
        gr.update(),
        gr.update(
            interactive=False,
            value=elapsed_button_label(action, started_at),
        ),
        *unchanged_history_view(),
    )
    for partial_results in iter_job_progress(future, progress_queue):
        yield (
            partial_results if partial_results is not None else gr.update(),
            gr.update(
                interactive=False,
                value=elapsed_button_label(action, started_at),
            ),
            *unchanged_history_view(),
        )
    try:
        entry_id, results = future.result()
    except BaseException:
        yield (
            gr.update(),
            gr.update(interactive=True, value=idle_button_label),
            *unchanged_history_view(),
        )
        raise
    yield (
        results,
        gr.update(interactive=True, value=idle_button_label),
        *history_view(get_history(), entry_id),
    )


def edit_image_flow(
    image_path: str | None,
    image_url: str | None,
    prompt: str,
    models: list[ModelName],
):
    progress_queue: queue.Queue[list[GalleryItem]] = queue.Queue()
    future = JOB_EXECUTOR.submit(
        run_edit_job,
        image_path,
        image_url,
        prompt,
        models,
        progress_queue,
    )
    yield from stream_job(future, progress_queue, "Editing", "Edit Image")


def run_generate_job(
    prompt: str,
    models: list[ModelName],
    aspect_ratio: str,
    num_images: int,
    progress_queue: queue.Queue[list[GalleryItem]],
) -> CompletedJob:
    return run_history_job(
        Operation.GENERATE,
        prompt,
        models,
        None,
        (
            f'Models: {", ".join(models)} · Aspect ratio: {aspect_ratio} · '
            f"Images: {num_images}"
        ),
        partial(
            generate_image,
            prompt,
            models,
            aspect_ratio,
            num_images,
            progress_queue.put,
        ),
    )


def generate_image_flow(
    prompt: str,
    models: list[ModelName],
    aspect_ratio: str,
    num_images: str,
):
    progress_queue: queue.Queue[list[GalleryItem]] = queue.Queue()
    future = JOB_EXECUTOR.submit(
        run_generate_job,
        prompt,
        models,
        aspect_ratio,
        int(num_images),
        progress_queue,
    )
    yield from stream_job(future, progress_queue, "Generating", "Generate")
