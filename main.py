import concurrent.futures
import os
import time
from typing import Any, TypedDict

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

MODEL_MAP: dict[ModelName, str] = {
    "Qwen Image Edit": "fal-ai/qwen-image-edit-2511",
    "FLUX.2 Klein 9B Edit": "fal-ai/flux-2/klein/9b/edit",
    "Grok Imagine Image Edit": "xai/grok-imagine-image/edit",
    "Seedream 5.0 Pro Edit (NanoGPT)": "bytedance/seedream-v5.0-pro/edit",
}

NANO_GPT_MODELS = {"Seedream 5.0 Pro Edit (NanoGPT)"}
MAX_HISTORY_ITEMS = 10

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


def add_history_entry(
    history: History,
    *,
    operation: str,
    prompt: str,
    input_image: str | None,
    outputs: list[GalleryItem],
    settings: str,
) -> History:
    entry: HistoryEntry = {
        "id": str(time.time_ns()),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "operation": operation,
        "prompt": prompt,
        "input_image": input_image,
        "outputs": list(outputs),
        "settings": settings,
    }
    return [entry, *history][:MAX_HISTORY_ITEMS]


def history_choices(history: History) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for entry in history:
        prompt = " ".join(entry["prompt"].split())
        if len(prompt) > 60:
            prompt = f"{prompt[:57]}..."
        label = f'{entry["created_at"]} · {entry["operation"]} · {prompt}'
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
        f'**{entry["operation"]}** · {entry["created_at"]}\n\n'
        f'{entry["settings"]}'
    )
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


def edit_image_flow(
    image_path: str, prompt: str, models: list[ModelName], history: History
):
    # Disable the button while work is in flight; yielding ensures the UI
    # reflects this state before the long-running call starts.
    yield (
        gr.update(),
        gr.update(interactive=False, value="Editing..."),
        history,
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )
    try:
        results = edit_image(image_path, prompt, models)
    except BaseException:
        # Re-enable the button on any failure (including gr.Error) so the UI
        # doesn't get stuck in the "Editing..." state, then re-raise to let
        # Gradio surface the error to the user.
        yield (
            gr.update(),
            gr.update(interactive=True, value="Edit Image"),
            history,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
        raise

    updated_history = add_history_entry(
        history,
        operation="Edit",
        prompt=prompt,
        input_image=image_path,
        outputs=results,
        settings=f'Models: {", ".join(models)}',
    )
    yield (
        results,
        gr.update(interactive=True, value="Edit Image"),
        updated_history,
        *history_view(updated_history),
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


def generate_image_flow(
    prompt: str,
    model_name: ModelName,
    aspect_ratio: str,
    num_images: str,
    history: History,
):
    yield (
        gr.update(),
        gr.update(interactive=False, value="Generating..."),
        history,
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )
    try:
        results = generate_image(
            prompt,
            model_name,
            aspect_ratio,
            int(num_images),
        )
    except BaseException:
        yield (
            gr.update(),
            gr.update(interactive=True, value="Generate"),
            history,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
        raise

    updated_history = add_history_entry(
        history,
        operation="Generate",
        prompt=prompt,
        input_image=None,
        outputs=results,
        settings=(
            f"Model: {model_name} · Aspect ratio: {aspect_ratio} · "
            f"Images: {num_images}"
        ),
    )
    yield (
        results,
        gr.update(interactive=True, value="Generate"),
        updated_history,
        *history_view(updated_history),
    )


with gr.Blocks(title="Image Studio") as demo:
    gr.Markdown("# Image Studio")
    history_state = gr.State([])

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
        history_state,
        history_selector,
        history_details,
        history_prompt,
        history_input,
        history_gallery,
    ]
    edit_btn.click(
        fn=edit_image_flow,
        inputs=[input_image, edit_prompt, models, history_state],
        outputs=[edit_gallery, edit_btn, *history_outputs],
    )
    gen_btn.click(
        fn=generate_image_flow,
        inputs=[gen_prompt, gen_model, gen_ratio, gen_num, history_state],
        outputs=[gen_gallery, gen_btn, *history_outputs],
    )
    gen_model.change(
        fn=generation_aspect_ratio_update,
        inputs=[gen_model, gen_ratio],
        outputs=[gen_ratio],
    )
    history_selector.change(
        fn=history_entry_view,
        inputs=[history_state, history_selector],
        outputs=[
            history_details,
            history_prompt,
            history_input,
            history_gallery,
        ],
    )
    history_refresh.click(
        fn=history_view,
        inputs=[history_state],
        outputs=[
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
