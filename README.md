# uv-gradio-coolify

A minimal [Gradio](https://www.gradio.app/) "Hello World" app, packaged with [uv](https://docs.astral.sh/uv/) and ready to deploy to a PaaS like [Coolify](https://coolify.io/) (or Railway, Fly.io, Render, etc.).

## Requirements

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) installed

## Run locally

```bash
uv sync
uv run python main.py
```

Then open the URL printed in the terminal (defaults to <http://127.0.0.1:7860>). The app also binds to `0.0.0.0` so other devices on your network can reach it.

## Notes

- **Image resizing**: Input images with any dimension larger than 2048px are automatically scaled down (preserving aspect ratio) before being sent to the model, which supports a maximum of 2048×2048.
- **History**: Each browser session keeps its latest 10 successful edit and generation requests in memory. History is cleared when the session or app restarts.

## Configuration

| Env var | Default | Description |
| --- | --- | --- |
| `PORT` | auto (`7860` if free) | Port the Gradio server binds to. Most PaaS platforms inject this automatically. |
| `FAL_KEY` | — | fal.ai API key used by the fal.ai edit and generation models. |
| `NANO_GPT_KEY` | — | NanoGPT API key used by the Seedream 5.0 Pro Edit model. |

## Deploying

The app reads `PORT` from the environment and binds to `0.0.0.0`, so it works out of the box on platforms that:

1. Inject a `PORT` env var
2. Expect the process to listen on all interfaces

### Coolify

1. Create a new **Application** and point it at this repo.
2. Build pack: **Nixpacks** or **Dockerfile** (Nixpacks autodetects `uv` via `pyproject.toml`).
3. Start command: `uv run python main.py`
4. Expose the port Coolify assigned (it sets `PORT` for you).

### Generic Docker

```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY . .
RUN uv sync --frozen --no-dev
EXPOSE 7860
CMD ["uv", "run", "python", "main.py"]
```

## Project layout

```
.
├── main.py           # Gradio app entrypoint
├── pyproject.toml    # Project + dependencies (managed by uv)
├── uv.lock           # Locked dependency versions
└── README.md
```
