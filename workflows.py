"""Edit/generation orchestration: parallel model fan-out and result streaming.

Model definitions live in models.py; provider-specific HTTP calls live in
clients/. This module routes each selected model to its provider client, runs
the calls in parallel, streams partial results to the UI, and records
completed jobs in the shared history.
"""

import concurrent.futures
import logging
import queue
import time
from collections.abc import Callable, Iterator

from clients import elapsed_ms
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
from image_utils import download_image_url, is_http_url
from models import (
    EDIT_MODELS,
    GENERATE_ASPECT_RATIOS,
    GENERATE_MODELS,
    EditModel,
    GenerateModel,
    Provider,
)


logger = logging.getLogger(__name__)

type ImageReference = str
type ModelName = str
type ProgressCallback = Callable[[list[GalleryItem]], None]
type CompletedJob = tuple[str, list[GalleryItem]]

JOB_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)
JOB_HEARTBEAT_SECONDS = 1.0


# --- Input handling -------------------------------------------------------


def normalize_image_url(image_url: str | None) -> str | None:
    if not image_url:
        return None

    normalized_url = image_url.strip()
    if not normalized_url:
        return None

    if not is_http_url(normalized_url):
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


def prepare_edit_inputs(
    providers: set[Provider],
    image_reference: ImageReference,
) -> dict[Provider, ImageReference]:
    """Convert the input once per provider: fal.ai wants a URL, NanoGPT a path."""
    references: dict[Provider, ImageReference] = {}
    is_remote = is_http_url(image_reference)
    if Provider.FAL in providers:
        references[Provider.FAL] = (
            image_reference if is_remote else fal_client.upload_image(image_reference)
        )
    if Provider.NANO_GPT in providers:
        references[Provider.NANO_GPT] = (
            download_image_url(image_reference) if is_remote else image_reference
        )
    return references


# --- Per-model provider dispatch ------------------------------------------


def run_edit_model(
    model: EditModel,
    image_reference: ImageReference,
    prompt: str,
) -> list[GalleryItem]:
    match model.provider:
        case Provider.FAL:
            image_url = fal_client.edit_image(model.model_id, image_reference, prompt)
        case Provider.NANO_GPT:
            image_url = nano_gpt_client.edit_image(
                model.model_id,
                image_reference,
                prompt,
                crop_aspect_ratios=model.crop_aspect_ratios,
            )
    return [(image_url, model.name)] if image_url else []


def run_generate_model(
    model: GenerateModel,
    prompt: str,
    aspect_ratio: str,
    num_images: int,
) -> list[GalleryItem]:
    output_ratio, resolution = model.resolve_aspect_ratio(aspect_ratio)
    match model.provider:
        case Provider.FAL:
            image_urls = fal_client.generate_images(
                model.model_id, prompt, output_ratio, num_images
            )
        case Provider.NANO_GPT:
            image_urls = nano_gpt_client.generate_images(
                model.model_id, prompt, resolution, num_images
            )

    ratio_label = (
        aspect_ratio
        if output_ratio == aspect_ratio
        else f"{aspect_ratio} → {output_ratio}"
    )
    return [(url, f"{model.name} ({ratio_label})") for url in image_urls]


# --- Parallel fan-out ------------------------------------------------------


def run_models_in_parallel(
    operation: str,
    tasks: dict[ModelName, Callable[[], list[GalleryItem]]],
    progress_callback: ProgressCallback | None,
) -> list[GalleryItem]:
    """Run one task per model, reporting partial results as each finishes."""
    results: list[GalleryItem] = []
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(task): model_name for model_name, task in tasks.items()
        }
        for future in concurrent.futures.as_completed(futures):
            model_name = futures[future]
            try:
                model_results = future.result()
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
                logger.warning(
                    "model response operation=%s model=%s status=error error_type=%s",
                    operation,
                    model_name,
                    type(exc).__name__,
                )
                continue
            if model_results:
                results.extend(model_results)
                if progress_callback:
                    progress_callback(list(results))

    if not results:
        detail = "; ".join(errors) if errors else "no images returned"
        raise gr.Error(f"All model calls failed ({detail}).")
    if errors:
        gr.Warning(f"Some models failed: {'; '.join(errors)}")
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
    if any(model_name not in EDIT_MODELS for model_name in models):
        raise gr.Error("Please select valid edit models.")

    selected_models = [EDIT_MODELS[model_name] for model_name in models]
    provider_inputs = prepare_edit_inputs(
        {model.provider for model in selected_models},
        image_reference,
    )
    tasks: dict[ModelName, Callable[[], list[GalleryItem]]] = {
        model.name: (
            lambda model=model: run_edit_model(
                model, provider_inputs[model.provider], prompt
            )
        )
        for model in selected_models
    }
    return run_models_in_parallel("edit", tasks, progress_callback)


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
    if any(model_name not in GENERATE_MODELS for model_name in models):
        raise gr.Error("Please select valid generation models.")
    if aspect_ratio not in GENERATE_ASPECT_RATIOS:
        raise gr.Error("Please select a valid aspect ratio.")

    tasks: dict[ModelName, Callable[[], list[GalleryItem]]] = {
        model_name: (
            lambda model=GENERATE_MODELS[model_name]: run_generate_model(
                model, prompt, aspect_ratio, num_images
            )
        )
        for model_name in models
    }
    return run_models_in_parallel("generate", tasks, progress_callback)


# --- Background jobs and history ------------------------------------------


def record_job(
    operation: str,
    prompt: str,
    input_image: ImageReference | None,
    settings: str,
    run: Callable[[], list[GalleryItem]],
) -> CompletedJob:
    """Run a job, log its outcome, and store the result in shared history."""
    started_at = time.monotonic()
    try:
        results = run()
    except Exception as exc:
        logger.error(
            "job response operation=%s status=error duration_ms=%d error_type=%s",
            operation.lower(),
            elapsed_ms(started_at),
            type(exc).__name__,
        )
        raise
    entry_id = add_history_entry(
        operation=operation,
        prompt=prompt,
        input_image=input_image,
        outputs=results,
        settings=settings,
    )
    logger.info(
        "job response operation=%s status=success duration_ms=%d images=%d history_id=%s",
        operation.lower(),
        elapsed_ms(started_at),
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
    logger.info(
        "job request operation=edit models=%d prompt_chars=%d",
        len(models),
        len(prompt),
    )
    return record_job(
        operation="Edit",
        prompt=prompt,
        input_image=image_reference,
        settings=f'Models: {", ".join(models)}',
        run=lambda: edit_image(
            image_path,
            prompt,
            models,
            progress_queue.put,
            image_url=image_url,
        ),
    )


def run_generate_job(
    prompt: str,
    models: list[ModelName],
    aspect_ratio: str,
    num_images: int,
    progress_queue: queue.Queue[list[GalleryItem]],
) -> CompletedJob:
    logger.info(
        "job request operation=generate models=%d prompt_chars=%d outputs_per_model=%d aspect_ratio=%s",
        len(models),
        len(prompt),
        num_images,
        aspect_ratio,
    )
    return record_job(
        operation="Generate",
        prompt=prompt,
        input_image=None,
        settings=(
            f'Models: {", ".join(models)} · Aspect ratio: {aspect_ratio} · '
            f"Images: {num_images}"
        ),
        run=lambda: generate_image(
            prompt,
            models,
            aspect_ratio,
            num_images,
            progress_queue.put,
        ),
    )


# --- Gradio streaming flows -------------------------------------------------


def elapsed_button_label(action: str, started_at: float) -> str:
    elapsed_seconds = max(0, int(time.monotonic() - started_at))
    minutes, seconds = divmod(elapsed_seconds, 60)
    if minutes:
        return f"{action}... {minutes}m {seconds:02d}s"
    return f"{action}... {seconds}s"


def iter_job_progress(
    future: concurrent.futures.Future[CompletedJob],
    progress_queue: queue.Queue[list[GalleryItem]],
    heartbeat_seconds: float = JOB_HEARTBEAT_SECONDS,
) -> Iterator[list[GalleryItem] | None]:
    """Yield partial results as they arrive, or None as a keep-alive heartbeat."""
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


def stream_job(
    action: str,
    idle_label: str,
    future: concurrent.futures.Future[CompletedJob],
    progress_queue: queue.Queue[list[GalleryItem]],
):
    """Stream (gallery, button, *history) updates until the job completes."""
    started_at = time.monotonic()

    def busy_button():
        return gr.update(
            interactive=False,
            value=elapsed_button_label(action, started_at),
        )

    yield (gr.update(), busy_button(), *unchanged_history_view())
    for partial_results in iter_job_progress(future, progress_queue):
        yield (
            partial_results if partial_results is not None else gr.update(),
            busy_button(),
            *unchanged_history_view(),
        )
    try:
        entry_id, results = future.result()
    except BaseException:
        yield (
            gr.update(),
            gr.update(interactive=True, value=idle_label),
            *unchanged_history_view(),
        )
        raise
    yield (
        results,
        gr.update(interactive=True, value=idle_label),
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
    yield from stream_job("Editing", "Edit Image", future, progress_queue)


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
    yield from stream_job("Generating", "Generate", future, progress_queue)
