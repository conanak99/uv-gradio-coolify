from html import escape
import threading
import time
from typing import Any, TypedDict
from urllib.parse import quote, urlparse
import uuid

import gradio as gr


type GalleryItem = tuple[str, str]
type History = list["HistoryEntry"]


class HistoryEntry(TypedDict):
    id: str
    created_at: str
    operation: str
    prompt: str
    input_image: str | None
    outputs: list[GalleryItem]
    settings: str


MAX_HISTORY_ITEMS = 10
HISTORY_STORE: History = []
HISTORY_LOCK = threading.Lock()


def get_history() -> History:
    with HISTORY_LOCK:
        return [
            {**entry, "outputs": list(entry["outputs"])}
            for entry in HISTORY_STORE
        ]


def add_history_entry(
    *,
    operation: str,
    prompt: str,
    input_image: str | None,
    outputs: list[GalleryItem],
    settings: str,
) -> str:
    entry: HistoryEntry = {
        "id": uuid.uuid4().hex,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "operation": operation,
        "prompt": prompt,
        "input_image": input_image,
        "outputs": list(outputs),
        "settings": settings,
    }
    with HISTORY_LOCK:
        HISTORY_STORE[:] = [entry, *HISTORY_STORE][:MAX_HISTORY_ITEMS]
    return entry["id"]


def history_choices(history: History) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for entry in history:
        prompt = " ".join(entry["prompt"].split())
        if len(prompt) > 60:
            prompt = f"{prompt[:57]}..."
        label = f'{entry["created_at"]} · {entry["operation"]} · {prompt}'
        choices.append((label, entry["id"]))
    return choices


def history_media_source(media: str) -> str:
    parsed = urlparse(media)
    if parsed.scheme in {"http", "https"} or media.startswith("data:image/"):
        return media
    return f"/gradio_api/file={quote(media, safe='/')}"


def history_image_html(media: str, caption: str) -> str:
    source = escape(history_media_source(media), quote=True)
    safe_caption = escape(caption)
    return (
        '<figure style="margin:0;min-width:0">'
        f'<a href="{source}" target="_blank" rel="noopener noreferrer">'
        f'<img src="{source}" alt="{safe_caption}" loading="lazy" '
        'style="display:block;width:100%;max-height:60vh;object-fit:contain;'
        'border-radius:8px;background:#f4f4f5">'
        "</a>"
        f'<figcaption style="margin-top:0.4rem">{safe_caption}</figcaption>'
        "</figure>"
    )


def history_outputs_html(outputs: list[GalleryItem]) -> str:
    if not outputs:
        return "<p>No outputs yet.</p>"
    images = "".join(
        history_image_html(image_url, caption)
        for image_url, caption in outputs
    )
    return (
        '<div style="display:grid;grid-template-columns:'
        'repeat(auto-fit,minmax(220px,1fr));gap:1rem">'
        f"{images}</div>"
    )


def history_entry_view(
    history: History, entry_id: str | None
) -> tuple[str, str, str, str]:
    if not history:
        return "No history yet.", "", "", "<p>No outputs yet.</p>"

    entry = next(
        (item for item in history if item["id"] == entry_id),
        history[0],
    )
    details = (
        f'**{entry["operation"]}** · {entry["created_at"]}\n\n'
        f'{entry["settings"]}'
    )
    input_html = (
        history_image_html(entry["input_image"], "Input image")
        if entry["input_image"]
        else "<p>No input image for this request.</p>"
    )
    return (
        details,
        entry["prompt"],
        input_html,
        history_outputs_html(entry["outputs"]),
    )


def history_view(
    history: History, entry_id: str | None = None
) -> tuple[Any, str, str, str, str]:
    if not history:
        return (
            gr.update(choices=[], value=None),
            "No history yet.",
            "",
            "",
            "<p>No outputs yet.</p>",
        )

    selected_id = entry_id if any(
        item["id"] == entry_id for item in history
    ) else history[0]["id"]
    return (
        gr.update(choices=history_choices(history), value=selected_id),
        *history_entry_view(history, selected_id),
    )


def stored_history_entry_view(
    entry_id: str | None,
) -> tuple[str, str, str, str]:
    return history_entry_view(get_history(), entry_id)


def refresh_history() -> tuple[Any, str, str, str, str]:
    return history_view(get_history())


def unchanged_history_view() -> tuple[Any, Any, Any, Any, Any]:
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )
