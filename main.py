from collections import OrderedDict
import concurrent.futures
import os
import threading
import time
from typing import Any, TypedDict
import uuid

from clients import fal as fal_client
from clients import nano_gpt as nano_gpt_client
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

type ImageUrl = str
type ImageReference = str
type ModelName = str
type GalleryItem = tuple[ImageUrl, ModelName]
type History = list["HistoryEntry"]


class HistoryEntry(TypedDict):
    id: str
    created_at: str
    operation: str
    prompt: str
    input_image: str | None
    outputs: list[GalleryItem]
    settings: str
    status: str
    error: str | None

MODEL_MAP: dict[ModelName, str] = {
    "Qwen Image Edit": "fal-ai/qwen-image-edit-2511",
    "FLUX.2 Klein 9B Edit": "fal-ai/flux-2/klein/9b/edit",
    "Grok Imagine Image Edit": "xai/grok-imagine-image/edit",
    "Seedream 5.0 Pro Edit (NanoGPT)": "bytedance/seedream-v5.0-pro/edit",
}

NANO_GPT_MODELS = {"Seedream 5.0 Pro Edit (NanoGPT)"}
MAX_HISTORY_ITEMS = 10
MAX_HISTORY_CLIENTS = 100
HISTORY_STORE: OrderedDict[str, History] = OrderedDict()
HISTORY_LOCK = threading.Lock()
JOB_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)

GROK_GENERATE_MODEL = "Grok Imagine (fal.ai)"
SEEDREAM_LITE_GENERATE_MODEL = "Seedream 5.0 Lite (NanoGPT)"
SEEDREAM_PRO_GENERATE_MODEL = "Seedream 5.0 Pro (NanoGPT)"
GENERATE_MODEL_MAP: dict[ModelName, str] = {
    GROK_GENERATE_MODEL: fal_client.GROK_GENERATE_MODEL_ID,
    SEEDREAM_LITE_GENERATE_MODEL: "seedream-v5.0-lite",
    SEEDREAM_PRO_GENERATE_MODEL: "bytedance/seedream-v5.0-pro",
}
GENERATE_ASPECT_RATIOS: dict[ModelName, list[str]] = {
    GROK_GENERATE_MODEL: ["16:9", "4:3", "1:1", "3:4", "9:16"],
    SEEDREAM_LITE_GENERATE_MODEL: ["16:9", "1:1", "9:16", "3:2", "2:3"],
    SEEDREAM_PRO_GENERATE_MODEL: [
        "16:9",
        "4:3",
        "1:1",
        "3:4",
        "9:16",
        "3:2",
        "2:3",
    ],
}
SEEDREAM_LITE_RESOLUTIONS = {
    "1:1": "2048x2048",
    "16:9": "2560x1440",
    "9:16": "1440x2560",
    "3:2": "3072x2048",
    "2:3": "2048x3072",
}


def generation_aspect_ratio_update(
    model_name: ModelName, current_ratio: str
) -> Any:
    choices = GENERATE_ASPECT_RATIOS[model_name]
    value = current_ratio if current_ratio in choices else "1:1"
    return gr.update(choices=choices, value=value)


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
    image_path: str, prompt: str, models: list[ModelName]
) -> list[GalleryItem]:
    if not image_path or not prompt:
        raise gr.Error("Please provide both an image and a prompt.")
    if not models:
        raise gr.Error("Please select at least one model.")

    image_references: dict[ModelName, ImageReference] = {}
    if any(model not in NANO_GPT_MODELS for model in models):
        fal_image_url = fal_client.upload_image(image_path)
        image_references.update(
            (model, fal_image_url)
            for model in models
            if model not in NANO_GPT_MODELS
        )
    if any(model in NANO_GPT_MODELS for model in models):
        image_references.update(
            (model, image_path) for model in models if model in NANO_GPT_MODELS
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
                print(f"[edit_image] {model_name} failed: {exc}")
                continue
            if result:
                results.append(result)

    if not results:
        detail = "; ".join(errors) if errors else "no images returned"
        raise gr.Error(f"All model calls failed ({detail}).")

    if errors:
        gr.Warning(f"Some models failed: {'; '.join(errors)}")

    return results


def ensure_client_id(client_id: str | None) -> str:
    if isinstance(client_id, str) and 0 < len(client_id) <= 128:
        return client_id
    return uuid.uuid4().hex


def get_client_history(client_id: str | None) -> History:
    if not client_id:
        return []
    with HISTORY_LOCK:
        history = HISTORY_STORE.get(client_id, [])
        if client_id in HISTORY_STORE:
            HISTORY_STORE.move_to_end(client_id)
        return [
            {**entry, "outputs": list(entry["outputs"])}
            for entry in history
        ]


def start_history_entry(
    client_id: str,
    *,
    operation: str,
    prompt: str,
    input_image: str | None,
    settings: str,
) -> str:
    entry: HistoryEntry = {
        "id": uuid.uuid4().hex,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "operation": operation,
        "prompt": prompt,
        "input_image": input_image,
        "outputs": [],
        "settings": settings,
        "status": "Running",
        "error": None,
    }
    with HISTORY_LOCK:
        history = HISTORY_STORE.get(client_id, [])
        HISTORY_STORE[client_id] = [entry, *history][:MAX_HISTORY_ITEMS]
        HISTORY_STORE.move_to_end(client_id)
        while len(HISTORY_STORE) > MAX_HISTORY_CLIENTS:
            HISTORY_STORE.popitem(last=False)
    return entry["id"]


def finish_history_entry(
    client_id: str,
    entry_id: str,
    *,
    outputs: list[GalleryItem] | None = None,
    error: str | None = None,
) -> None:
    with HISTORY_LOCK:
        history = HISTORY_STORE.get(client_id)
        if history is None:
            return
        HISTORY_STORE[client_id] = [
            {
                **entry,
                "outputs": list(outputs or []),
                "status": "Failed" if error is not None else "Completed",
                "error": error,
            }
            if entry["id"] == entry_id
            else entry
            for entry in history
        ]
        HISTORY_STORE.move_to_end(client_id)


def history_choices(history: History) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for entry in history:
        prompt = " ".join(entry["prompt"].split())
        if len(prompt) > 60:
            prompt = f"{prompt[:57]}..."
        label = (
            f'{entry["created_at"]} · {entry["status"]} · '
            f'{entry["operation"]} · {prompt}'
        )
        choices.append((label, entry["id"]))
    return choices


def history_entry_view(
    history: History, entry_id: str | None
) -> tuple[str, str, str | None, list[GalleryItem]]:
    if not history:
        return "No history yet.", "", None, []

    entry = next(
        (item for item in history if item["id"] == entry_id),
        history[0],
    )
    details = (
        f'**{entry["operation"]} · {entry["status"]}** · '
        f'{entry["created_at"]}\n\n'
        f'{entry["settings"]}'
    )
    if entry["error"]:
        details += f'\n\nError: {entry["error"]}'
    return (
        details,
        entry["prompt"],
        entry["input_image"],
        entry["outputs"],
    )


def history_view(
    history: History, entry_id: str | None = None
) -> tuple[Any, str, str, str | None, list[GalleryItem]]:
    if not history:
        return gr.update(choices=[], value=None), "No history yet.", "", None, []

    selected_id = entry_id if any(
        item["id"] == entry_id for item in history
    ) else history[0]["id"]
    return (
        gr.update(choices=history_choices(history), value=selected_id),
        *history_entry_view(history, selected_id),
    )


def initialize_client_history(
    client_id: str | None,
) -> tuple[str, Any, str, str, str | None, list[GalleryItem]]:
    client_id = ensure_client_id(client_id)
    return client_id, *history_view(get_client_history(client_id))


def stored_history_entry_view(
    client_id: str | None, entry_id: str | None
) -> tuple[str, str, str | None, list[GalleryItem]]:
    return history_entry_view(get_client_history(client_id), entry_id)


def refresh_history(
    client_id: str | None,
) -> tuple[Any, str, str, str | None, list[GalleryItem]]:
    return history_view(get_client_history(client_id))


def run_edit_job(
    client_id: str,
    entry_id: str,
    image_path: str,
    prompt: str,
    models: list[ModelName],
) -> list[GalleryItem]:
    try:
        results = edit_image(image_path, prompt, models)
    except Exception as exc:
        finish_history_entry(client_id, entry_id, error=str(exc))
        raise
    finish_history_entry(client_id, entry_id, outputs=results)
    return results


def edit_image_flow(
    image_path: str,
    prompt: str,
    models: list[ModelName],
    client_id: str | None,
):
    client_id = ensure_client_id(client_id)
    entry_id = start_history_entry(
        client_id,
        operation="Edit",
        prompt=prompt,
        input_image=image_path,
        settings=f'Models: {", ".join(models)}',
    )
    future = JOB_EXECUTOR.submit(
        run_edit_job,
        client_id,
        entry_id,
        image_path,
        prompt,
        models,
    )
    yield (
        gr.update(),
        gr.update(interactive=False, value="Editing..."),
        client_id,
        *history_view(get_client_history(client_id), entry_id),
    )
    try:
        results = future.result()
    except BaseException:
        yield (
            gr.update(),
            gr.update(interactive=True, value="Edit Image"),
            client_id,
            *history_view(get_client_history(client_id), entry_id),
        )
        raise

    yield (
        results,
        gr.update(interactive=True, value="Edit Image"),
        client_id,
        *history_view(get_client_history(client_id), entry_id),
    )


def generate_image(
    prompt: str,
    model_name: ModelName,
    aspect_ratio: str,
    num_images: int,
) -> list[GalleryItem]:
    if not prompt:
        raise gr.Error("Please provide a prompt.")
    if model_name not in GENERATE_MODEL_MAP:
        raise gr.Error("Please select a valid generation model.")
    if aspect_ratio not in GENERATE_ASPECT_RATIOS[model_name]:
        raise gr.Error(f"{model_name} does not support {aspect_ratio}.")

    if model_name == GROK_GENERATE_MODEL:
        image_urls = fal_client.generate_images(prompt, aspect_ratio, num_images)
    else:
        resolution = (
            SEEDREAM_LITE_RESOLUTIONS[aspect_ratio]
            if model_name == SEEDREAM_LITE_GENERATE_MODEL
            else aspect_ratio
        )
        image_urls = nano_gpt_client.generate_images(
            GENERATE_MODEL_MAP[model_name],
            prompt,
            resolution,
            num_images,
        )
    if not image_urls:
        raise gr.Error("No images returned from the model.")
    return [(url, f"{model_name} ({aspect_ratio})") for url in image_urls]


def run_generate_job(
    client_id: str,
    entry_id: str,
    prompt: str,
    model_name: ModelName,
    aspect_ratio: str,
    num_images: int,
) -> list[GalleryItem]:
    try:
        results = generate_image(
            prompt,
            model_name,
            aspect_ratio,
            num_images,
        )
    except Exception as exc:
        finish_history_entry(client_id, entry_id, error=str(exc))
        raise
    finish_history_entry(client_id, entry_id, outputs=results)
    return results


def generate_image_flow(
    prompt: str,
    model_name: ModelName,
    aspect_ratio: str,
    num_images: str,
    client_id: str | None,
):
    client_id = ensure_client_id(client_id)
    entry_id = start_history_entry(
        client_id,
        operation="Generate",
        prompt=prompt,
        input_image=None,
        settings=(
            f"Model: {model_name} · Aspect ratio: {aspect_ratio} · "
            f"Images: {num_images}"
        ),
    )
    future = JOB_EXECUTOR.submit(
        run_generate_job,
        client_id,
        entry_id,
        prompt,
        model_name,
        aspect_ratio,
        int(num_images),
    )
    yield (
        gr.update(),
        gr.update(interactive=False, value="Generating..."),
        client_id,
        *history_view(get_client_history(client_id), entry_id),
    )
    try:
        results = future.result()
    except BaseException:
        yield (
            gr.update(),
            gr.update(interactive=True, value="Generate"),
            client_id,
            *history_view(get_client_history(client_id), entry_id),
        )
        raise

    yield (
        results,
        gr.update(interactive=True, value="Generate"),
        client_id,
        *history_view(get_client_history(client_id), entry_id),
    )


with gr.Blocks(title="Image Studio") as demo:
    gr.Markdown("# Image Studio")
    client_id_state = gr.BrowserState(
        default_value=None,
        storage_key="image-studio-client-id",
    )

    with gr.Tabs():
        with gr.Tab("Edit"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="filepath", label="Input Image", height="50vh")
                    edit_prompt = gr.Textbox(
                        label="Edit Prompt",
                        placeholder="Describe what you want to change...",
                    )
                    models = gr.CheckboxGroup(
                        choices=list(MODEL_MAP.keys()),
                        value=list(MODEL_MAP.keys()),
                        label="Models",
                    )
                    edit_btn = gr.Button("Edit Image", variant="primary")

                with gr.Column():
                    edit_gallery = gr.Gallery(
                        label="Edited Images", columns=2, object_fit="contain"
                    )

        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column():
                    gen_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image you want to generate...",
                    )
                    gen_model = gr.Radio(
                        choices=list(GENERATE_MODEL_MAP.keys()),
                        value=GROK_GENERATE_MODEL,
                        label="Model",
                    )
                    gen_ratio = gr.Radio(
                        choices=GENERATE_ASPECT_RATIOS[GROK_GENERATE_MODEL],
                        value="3:4",
                        label="Aspect Ratio",
                    )
                    gen_num = gr.Radio(
                        choices=["1", "2", "4"],
                        value="2",
                        label="Number of Images",
                    )
                    gen_btn = gr.Button("Generate", variant="primary")

                with gr.Column():
                    gen_gallery = gr.Gallery(
                        label="Generated Images", columns=2, object_fit="contain"
                    )

        with gr.Tab("History"):
            with gr.Row():
                history_selector = gr.Dropdown(
                    choices=[],
                    label="Latest 10 requests",
                    interactive=True,
                )
                history_refresh = gr.Button("Refresh")

            history_details = gr.Markdown("No history yet.")
            history_prompt = gr.Textbox(
                label="Prompt",
                interactive=False,
            )
            with gr.Row():
                history_input = gr.Image(
                    label="Input Image",
                    interactive=False,
                    height="40vh",
                )
                history_gallery = gr.Gallery(
                    label="Outputs",
                    columns=2,
                    object_fit="contain",
                )

    history_outputs = [
        client_id_state,
        history_selector,
        history_details,
        history_prompt,
        history_input,
        history_gallery,
    ]
    edit_btn.click(
        fn=edit_image_flow,
        inputs=[input_image, edit_prompt, models, client_id_state],
        outputs=[edit_gallery, edit_btn, *history_outputs],
    )
    gen_btn.click(
        fn=generate_image_flow,
        inputs=[gen_prompt, gen_model, gen_ratio, gen_num, client_id_state],
        outputs=[gen_gallery, gen_btn, *history_outputs],
    )
    gen_model.change(
        fn=generation_aspect_ratio_update,
        inputs=[gen_model, gen_ratio],
        outputs=[gen_ratio],
    )
    history_selector.change(
        fn=stored_history_entry_view,
        inputs=[client_id_state, history_selector],
        outputs=[
            history_details,
            history_prompt,
            history_input,
            history_gallery,
        ],
    )
    history_refresh.click(
        fn=refresh_history,
        inputs=[client_id_state],
        outputs=[
            history_selector,
            history_details,
            history_prompt,
            history_input,
            history_gallery,
        ],
    )
    demo.load(
        fn=initialize_client_history,
        inputs=[client_id_state],
        outputs=[
            client_id_state,
            history_selector,
            history_details,
            history_prompt,
            history_input,
            history_gallery,
        ],
    )


if __name__ == "__main__":
    port_env = os.environ.get("PORT")
    port = int(port_env) if port_env else None
    demo.launch(server_name="0.0.0.0", server_port=port)
