import os

from dotenv import load_dotenv
import gradio as gr

from history import refresh_history, stored_history_entry_view
from workflows import (
    GENERATE_ASPECT_RATIOS,
    GENERATE_MODEL_MAP,
    MODEL_MAP,
    edit_image_flow,
    generate_image_flow,
)


load_dotenv()

PROMPT_CSS = """
.prompt-input textarea,
.prompt-input input {
    font-size: 16px !important;
}
"""


with gr.Blocks(title="Image Studio", css=PROMPT_CSS) as demo:
    gr.Markdown("# Image Studio")

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
                        choices=list(MODEL_MAP.keys()),
                        value=list(MODEL_MAP.keys()),
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
                        choices=list(GENERATE_MODEL_MAP.keys()),
                        value=list(GENERATE_MODEL_MAP.keys()),
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
    gen_btn.click(
        fn=generate_image_flow,
        inputs=[gen_prompt, gen_models, gen_ratio, gen_num],
        outputs=[gen_gallery, gen_btn, *history_outputs],
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


if __name__ == "__main__":
    port_env = os.environ.get("PORT")
    port = int(port_env) if port_env else None
    demo.launch(server_name="0.0.0.0", server_port=port)
