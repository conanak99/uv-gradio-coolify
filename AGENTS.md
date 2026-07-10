# AGENTS.md

## Cursor Cloud specific instructions

This is a single-service [Gradio](https://www.gradio.app/) app ("Image Studio") managed with [uv](https://docs.astral.sh/uv/). It proxies image edit/generation requests to external paid APIs (fal.ai and NanoGPT).

### Running / testing / building
- Standard commands live in `README.md` and `pyproject.toml`. Dependencies are installed via `uv sync` (handled by the startup update script).
- Run the app (dev): `uv run python main.py`. It binds `0.0.0.0` and defaults to port `7860` (override with `PORT`).
- Run tests: `uv run python -m unittest test_main.py`. The suite is `unittest`-based and fully mocks external HTTP calls, so it needs no API keys or network.
- There is no build step (pure Python) and no linter configured (no ruff/flake8 config, no dev dependencies).

### Non-obvious caveats
- Core functionality (Edit/Generate tabs) requires `FAL_KEY` and/or `NANO_GPT_KEY` env vars. Without them the UI loads and requests run, but every model call fails gracefully with an error notification naming the missing key. The tests do NOT need these keys.
- History is in-memory only (latest 10 jobs) and is cleared on restart.
