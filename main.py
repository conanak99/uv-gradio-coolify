import os

import gradio as gr


def greet(name):
    return f"Hello {name}!"


demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    title="Hello World Updated",
)

if __name__ == "__main__":
    port_env = os.environ.get("PORT")
    port = int(port_env) if port_env else None
    demo.launch(server_port=port)
