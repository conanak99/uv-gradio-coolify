import base64
import concurrent.futures
import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
    "Seedream 5.0 Pro Edit (NanoGPT)": "bytedance/seedream-v5.0-pro/edit",
}

NANO_GPT_MODELS = {"Seedream 5.0 Pro Edit (NanoGPT)"}
NANO_GPT_IMAGES_URL = "https://nano-gpt.com/api/v1/images"
NANO_GPT_MAX_INPUT_BYTES = 10 * 1024 * 1024

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


def image_to_data_url(image_path: str) -> ImageUrl:
    image_path = resize_if_needed(image_path)
    with Image.open(image_path) as image:
        mime_type = Image.MIME.get(image.format or "", "image/png")
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    if len(image_bytes) > NANO_GPT_MAX_INPUT_BYTES:
        raise RuntimeError("NanoGPT input image must be 10 MB or smaller after resizing.")

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def run_nano_gpt_model(
    model_name: ModelName, image_url: ImageUrl, prompt: str
) -> GalleryItem | None:
    api_key = os.environ.get("NANO_GPT_KEY")
    if not api_key:
        raise RuntimeError("NANO_GPT_KEY is not configured.")

    payload = {
        "model": MODEL_MAP[model_name],
        "prompt": prompt,
        "input_references": [image_url],
        "n": 1,
    }
    request = Request(
        NANO_GPT_IMAGES_URL,
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
    if not images:
        return None
    if image_url := images[0].get("url"):
        return (image_url, model_name)
    if image_base64 := images[0].get("b64_json"):
        return (f"data:image/png;base64,{image_base64}", model_name)
    return None


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
    if model_name in NANO_GPT_MODELS:
        return run_nano_gpt_model(model_name, image_url, prompt)

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

    image_urls: dict[ModelName, ImageUrl] = {}
    if any(model not in NANO_GPT_MODELS for model in models):
        fal_image_url = upload_image_to_fal(image_path)
        image_urls.update(
            (model, fal_image_url)
            for model in models
            if model not in NANO_GPT_MODELS
        )
    if any(model in NANO_GPT_MODELS for model in models):
        nano_gpt_image_url = image_to_data_url(image_path)
        image_urls.update(
            (model, nano_gpt_image_url) for model in models if model in NANO_GPT_MODELS
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures: dict[concurrent.futures.Future[GalleryItem | None], ModelName] = {
            executor.submit(run_model, name, image_urls[name], prompt): name
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


def generate_image(prompt: str, aspect_ratio: str, num_images: int) -> list[GalleryItem]:
    if not prompt:
        raise gr.Error("Please provide a prompt.")

    result: dict[str, Any] = fal_client.subscribe(
        "xai/grok-imagine-image",
        arguments={
            "prompt": prompt,
            "num_images": num_images,
            "aspect_ratio": aspect_ratio,
            "resolution": "1k",
            "output_format": "jpeg",
            "sync_mode": False,
        },
    )
    images: list[dict[str, Any]] = result.get("images", [])
    if not images:
        raise gr.Error("No images returned from the model.")
    return [(img["url"], f"Grok Imagine ({aspect_ratio})") for img in images]


def generate_image_flow(prompt: str, aspect_ratio: str, num_images: str):
    yield gr.update(), gr.update(interactive=False, value="Generating...")
    try:
        results = generate_image(prompt, aspect_ratio, int(num_images))
    except BaseException:
        yield gr.update(), gr.update(interactive=True, value="Generate")
        raise
    yield results, gr.update(interactive=True, value="Generate")


with gr.Blocks(title="Image Studio") as demo:
    gr.Markdown("# Image Studio")

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

            edit_btn.click(
                fn=edit_image_flow,
                inputs=[input_image, edit_prompt, models],
                outputs=[edit_gallery, edit_btn],
            )

        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column():
                    gen_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image you want to generate...",
                    )
                    gen_ratio = gr.Radio(
                        choices=["16:9", "4:3", "1:1", "3:4", "9:16"],
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

            gen_btn.click(
                fn=generate_image_flow,
                inputs=[gen_prompt, gen_ratio, gen_num],
                outputs=[gen_gallery, gen_btn],
            )


if __name__ == "__main__":
    port_env = os.environ.get("PORT")
    port = int(port_env) if port_env else None
    demo.launch(server_name="0.0.0.0", server_port=port)
