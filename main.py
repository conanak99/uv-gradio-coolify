import concurrent.futures
import os
import time
from typing import Any

from dotenv import load_dotenv
import fal_client
import gradio as gr
from PIL import Image

load_dotenv()

type ImageUrl = str
type ModelName = str
type GalleryItem = tuple[ImageUrl, ModelName]

MODEL_MAP: dict[ModelName, str] = {
    "Qwen Image Edit": "fal-ai/qwen-image-edit-2511",
    "FLUX.2 Klein 9B Edit": "fal-ai/flux-2/klein/9b/edit",
    "Grok Imagine Image Edit": "xai/grok-imagine-image/edit",
}


MAX_DIMENSION = 2048


def resize_if_needed(image_path: str) -> str:
    img = Image.open(image_path)
    w, h = img.size
    if w <= MAX_DIMENSION and h <= MAX_DIMENSION:
        return image_path
    scale = MAX_DIMENSION / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    img = img.resize(new_size, Image.LANCZOS)
    resized_path = image_path + "_resized.png"
    img.save(resized_path)
    return resized_path


def upload_image_to_fal(image_path: str) -> ImageUrl:
    image_path = resize_if_needed(image_path)
    return fal_client.upload_file(image_path)


def poll_result(request_id: str, model_id: str) -> dict[str, Any]:
    while True:
        status = fal_client.queue.status(model_id, request_id, with_logs=False)
        if status["status"] == "COMPLETED":
            return fal_client.queue.result(model_id, request_id)
        if status["status"] == "FAILED":
            raise RuntimeError(f"Request failed: {status}")
        time.sleep(1)


def build_arguments(model_id: str, image_url: ImageUrl, prompt: str) -> dict[str, Any]:
    args: dict[str, Any] = {
        "prompt": prompt,
        "image_urls": [image_url],
        "sync_mode": False,
        "num_images": 1,
    }
    # Grok Imagine doesn't accept `enable_safety_checker`; only add it for
    # models that do support it.
    if not model_id.startswith("xai/"):
        args["enable_safety_checker"] = False
    return args


def run_model(
    model_name: ModelName, image_url: ImageUrl, prompt: str
) -> GalleryItem | None:
    model_id = MODEL_MAP[model_name]
    result: dict[str, Any] = fal_client.subscribe(
        model_id,
        arguments=build_arguments(model_id, image_url, prompt),
    )
    images: list[dict[str, Any]] = result.get("images", [])
    if images:
        return (images[0]["url"], model_name)
    return None


def edit_image(
    image_path: str, prompt: str, models: list[ModelName]
) -> list[GalleryItem]:
    if not image_path or not prompt:
        raise gr.Error("Please provide both an image and a prompt.")
    if not models:
        raise gr.Error("Please select at least one model.")

    image_url: ImageUrl = upload_image_to_fal(image_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures: dict[concurrent.futures.Future[GalleryItem | None], ModelName] = {
            executor.submit(run_model, name, image_url, prompt): name for name in models
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


def edit_image_flow(
    image_path: str, prompt: str, models: list[ModelName]
):
    # Disable the button while work is in flight; yielding ensures the UI
    # reflects this state before the long-running call starts.
    yield gr.update(), gr.update(interactive=False, value="Editing...")
    try:
        results = edit_image(image_path, prompt, models)
    except BaseException:
        # Re-enable the button on any failure (including gr.Error) so the UI
        # doesn't get stuck in the "Editing..." state, then re-raise to let
        # Gradio surface the error to the user.
        yield gr.update(), gr.update(interactive=True, value="Edit Image")
        raise
    yield results, gr.update(interactive=True, value="Edit Image")


with gr.Blocks(title="Image Editor - fal.ai") as demo:
    gr.Markdown("# Image Editor with fal.ai")
    gr.Markdown("Upload an image, describe the edit, and choose a model.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Input Image", height="50vh")
            prompt = gr.Textbox(
                label="Edit Prompt",
                placeholder="Describe what you want to change...",
            )
            models = gr.CheckboxGroup(
                choices=list(MODEL_MAP.keys()),
                value=list(MODEL_MAP.keys()),
                label="Models",
            )
            submit_btn = gr.Button("Edit Image", variant="primary")

        with gr.Column():
            output_gallery = gr.Gallery(
                label="Edited Images", columns=2, object_fit="contain"
            )

    submit_btn.click(
        fn=edit_image_flow,
        inputs=[input_image, prompt, models],
        outputs=[output_gallery, submit_btn],
    )


if __name__ == "__main__":
    port_env = os.environ.get("PORT")
    port = int(port_env) if port_env else None
    demo.launch(server_name="0.0.0.0", server_port=port)
