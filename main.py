import os

import gradio as gr


def greet(name):
    return f"Hello {name}!"


demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    title="Hello World",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 0))
    demo.launch(server_port=port)
