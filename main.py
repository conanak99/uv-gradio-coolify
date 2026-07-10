import logging
import os

from dotenv import load_dotenv
import gradio as gr

from clients.nano_gpt import OUTPUT_CACHE_PATH
from history import refresh_history, stored_history_entry_view
from workflows import (
    GENERATE_ASPECT_RATIOS,
    GENERATE_MODEL_MAP,
    MODEL_MAP,
    edit_image_flow,
    generate_image_flow,
)


load_dotenv()

DEFAULT_EDIT_MODELS = list(MODEL_MAP.keys())
DEFAULT_GENERATE_MODELS = list(GENERATE_MODEL_MAP.keys())
BROWSER_STATE_SECRET = os.environ.get(
    "BROWSER_STATE_SECRET",
    "image-studio-browser-state-v1",
)


def valid_model_selection(
    saved_selection: object,
    available_models: list[str],
) -> list[str]:
    if not isinstance(saved_selection, list | tuple):
        return list(available_models)

    saved_models = {
        model_name for model_name in saved_selection if isinstance(model_name, str)
    }
    selected_models = [
        model_name for model_name in available_models if model_name in saved_models
    ]
    return selected_models or list(available_models)


def load_model_preferences(
    saved_edit_models: object,
    saved_generate_models: object,
) -> tuple[list[str], list[str]]:
    return (
        valid_model_selection(saved_edit_models, DEFAULT_EDIT_MODELS),
        valid_model_selection(saved_generate_models, DEFAULT_GENERATE_MODELS),
    )


def save_edit_model_preference(selected_models: object) -> list[str]:
    return valid_model_selection(selected_models, DEFAULT_EDIT_MODELS)


def save_generate_model_preference(selected_models: object) -> list[str]:
    return valid_model_selection(selected_models, DEFAULT_GENERATE_MODELS)


def edit_prompt_preference(saved_prompt: object) -> str:
    return saved_prompt if isinstance(saved_prompt, str) else ""

log_level = getattr(
    logging,
    os.environ.get("LOG_LEVEL", "INFO").upper(),
    logging.INFO,
)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
for logger_name in ("clients.fal", "clients.nano_gpt", "history", "workflows"):
    logging.getLogger(logger_name).setLevel(log_level)

PROMPT_CSS = """
.prompt-input textarea,
.prompt-input input {
    font-size: 16px !important;
}
.history-select input,
.history-select select {
    font-size: 16px !important;
}
"""


with gr.Blocks(title="Image Studio") as demo:
    gr.Markdown("# Image Studio")
    edit_models_preference = gr.BrowserState(
        DEFAULT_EDIT_MODELS,
        storage_key="image-studio-edit-models",
        secret=BROWSER_STATE_SECRET,
    )
    generate_models_preference = gr.BrowserState(
        DEFAULT_GENERATE_MODELS,
        storage_key="image-studio-generate-models",
        secret=BROWSER_STATE_SECRET,
    )
    edit_prompt_state = gr.BrowserState(
        "",
        storage_key="image-studio-edit-prompt",
        secret=BROWSER_STATE_SECRET,
    )

    with gr.Tabs():
        with gr.Tab("Edit"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        type="filepath",
                        label="Input Image",
                        height="50vh",
                    )
                    edit_prompt = gr.Textbox(
                        label="Edit Prompt",
                        placeholder="Describe what you want to change...",
                        elem_classes=["prompt-input"],
                    )
                    models = gr.CheckboxGroup(
                        choices=DEFAULT_EDIT_MODELS,
                        value=DEFAULT_EDIT_MODELS,
                        label="Models",
                    )
                    edit_btn = gr.Button("Edit Image", variant="primary")

                with gr.Column():
                    edit_gallery = gr.Gallery(
                        label="Edited Images",
                        columns=2,
                        object_fit="contain",
                    )

        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column():
                    gen_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image you want to generate...",
                        elem_classes=["prompt-input"],
                    )
                    gen_models = gr.CheckboxGroup(
                        choices=DEFAULT_GENERATE_MODELS,
                        value=DEFAULT_GENERATE_MODELS,
                        label="Models",
                    )
                    gen_ratio = gr.Radio(
                        choices=GENERATE_ASPECT_RATIOS,
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
                        label="Generated Images",
                        columns=2,
                        object_fit="contain",
                    )

        with gr.Tab("History"):
            with gr.Row():
                history_selector = gr.Dropdown(
                    choices=[],
                    label="Latest 10 requests",
                    interactive=True,
                    elem_classes=["history-select"],
                    allow_custom_value=True,
                )
                history_refresh = gr.Button("Refresh")

            history_details = gr.Markdown("No history yet.")
            history_prompt = gr.Textbox(
                label="Prompt",
                interactive=False,
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Input Image")
                    history_input = gr.HTML()
                with gr.Column():
                    gr.Markdown("### Outputs")
                    history_gallery = gr.HTML("<p>No outputs yet.</p>")

    history_outputs = [
        history_selector,
        history_details,
        history_prompt,
        history_input,
        history_gallery,
    ]
    edit_btn.click(
        fn=edit_image_flow,
        inputs=[input_image, edit_prompt, models],
        outputs=[edit_gallery, edit_btn, *history_outputs],
    )
    models.change(
        fn=save_edit_model_preference,
        inputs=[models],
        outputs=[edit_models_preference],
        queue=False,
        show_progress="hidden",
    )
    edit_prompt.change(
        fn=edit_prompt_preference,
        inputs=[edit_prompt],
        outputs=[edit_prompt_state],
        queue=False,
        show_progress="hidden",
    )
    gen_btn.click(
        fn=generate_image_flow,
        inputs=[gen_prompt, gen_models, gen_ratio, gen_num],
        outputs=[gen_gallery, gen_btn, *history_outputs],
    )
    gen_models.change(
        fn=save_generate_model_preference,
        inputs=[gen_models],
        outputs=[generate_models_preference],
        queue=False,
        show_progress="hidden",
    )
    history_selector.change(
        fn=stored_history_entry_view,
        inputs=[history_selector],
        outputs=[
            history_details,
            history_prompt,
            history_input,
            history_gallery,
        ],
        queue=False,
        show_progress="hidden",
    )
    history_refresh.click(
        fn=refresh_history,
        outputs=[
            history_selector,
            history_details,
            history_prompt,
            history_input,
            history_gallery,
        ],
        queue=False,
        show_progress="hidden",
    )
    demo.load(
        fn=refresh_history,
        outputs=[
            history_selector,
            history_details,
            history_prompt,
            history_input,
            history_gallery,
        ],
        queue=False,
        show_progress="hidden",
    )
    demo.load(
        fn=load_model_preferences,
        inputs=[edit_models_preference, generate_models_preference],
        outputs=[models, gen_models],
        queue=False,
        show_progress="hidden",
    )
    demo.load(
        fn=edit_prompt_preference,
        inputs=[edit_prompt_state],
        outputs=[edit_prompt],
        queue=False,
        show_progress="hidden",
    )


def launch_app(port: int | None = None):
    return demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        css=PROMPT_CSS,
        allowed_paths=[str(OUTPUT_CACHE_PATH)],
    )


if __name__ == "__main__":
    port_env = os.environ.get("PORT")
    port = int(port_env) if port_env else None
    launch_app(port)
