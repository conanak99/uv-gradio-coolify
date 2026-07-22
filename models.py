"""Registry of the edit and generation models offered in the UI.

Adding a new model is a single entry in EDIT_MODELS or GENERATE_MODELS.
Adding a new provider means a new Provider member plus routing branches in
workflows.py (edit input preparation and client dispatch).
"""

from dataclasses import dataclass, field
from enum import Enum


class Provider(Enum):
    """External API that serves a model."""

    FAL = "fal.ai"
    NANO_GPT = "NanoGPT"


# Aspect ratios offered in the Generate tab.
GENERATE_ASPECT_RATIOS = ["16:9", "4:3", "1:1", "3:4", "9:16"]

# Output ratios accepted by Seedream edit; inputs are cropped to the closest.
SEEDREAM_EDIT_ASPECT_RATIOS = ("1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4")


@dataclass(frozen=True)
class EditModel:
    name: str
    provider: Provider
    model_id: str
    # When set, the input image is center-cropped to the closest of these
    # ratios and that ratio is sent to the API as the output size.
    crop_aspect_ratios: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerateModel:
    name: str
    provider: Provider
    model_id: str
    # UI aspect ratio -> (closest supported ratio, API resolution value).
    # Ratios not listed are passed to the API unchanged.
    ratio_overrides: dict[str, tuple[str, str]] = field(default_factory=dict)

    def resolve_aspect_ratio(self, aspect_ratio: str) -> tuple[str, str]:
        return self.ratio_overrides.get(aspect_ratio, (aspect_ratio, aspect_ratio))


_SEEDREAM_LITE_RATIOS = {
    "16:9": ("16:9", "2560x1440"),
    "4:3": ("3:2", "3072x2048"),
    "1:1": ("1:1", "2048x2048"),
    "3:4": ("2:3", "2048x3072"),
    "9:16": ("9:16", "1440x2560"),
}
_WAN_27_RATIOS = {
    "16:9": ("16:9", "1280*720"),
    "4:3": ("3:2", "1536*1024"),
    "1:1": ("1:1", "1024*1024"),
    "3:4": ("2:3", "1024*1536"),
    "9:16": ("9:16", "720*1280"),
}


def _by_name[Model: (EditModel, GenerateModel)](
    models: list[Model],
) -> dict[str, Model]:
    return {model.name: model for model in models}


EDIT_MODELS = _by_name(
    [
        EditModel(
            "Qwen Image Edit",
            Provider.FAL,
            "fal-ai/qwen-image-edit-2511",
        ),
        EditModel(
            "FLUX.2 Klein 9B Edit",
            Provider.FAL,
            "fal-ai/flux-2/klein/9b/edit",
        ),
        EditModel(
            "Grok Imagine Image Edit",
            Provider.FAL,
            "xai/grok-imagine-image/edit",
        ),
        EditModel(
            "Seedream 5.0 Pro Edit (NanoGPT)",
            Provider.NANO_GPT,
            "bytedance/seedream-v5.0-pro/edit",
            crop_aspect_ratios=SEEDREAM_EDIT_ASPECT_RATIOS,
        ),
        EditModel(
            "WAN 2.6 Image Edit (NanoGPT)",
            Provider.NANO_GPT,
            "wan-2.6-image-edit",
        ),
        # No crop ratios: NanoGPT defaults this model's resolution to "auto",
        # which matches the input image ratio (same behavior as WAN edit).
        EditModel(
            "P-Image Edit (NanoGPT)",
            Provider.NANO_GPT,
            "pruna-ai/p-image/edit",
        ),
    ]
)

GENERATE_MODELS = _by_name(
    [
        GenerateModel(
            "Grok Imagine (fal.ai)",
            Provider.FAL,
            "xai/grok-imagine-image",
        ),
        GenerateModel(
            "Seedream 5.0 Lite (NanoGPT)",
            Provider.NANO_GPT,
            "seedream-v5.0-lite",
            ratio_overrides=_SEEDREAM_LITE_RATIOS,
        ),
        GenerateModel(
            "Seedream 5.0 Pro (NanoGPT)",
            Provider.NANO_GPT,
            "bytedance/seedream-v5.0-pro",
        ),
        GenerateModel(
            "WAN 2.7 Image (NanoGPT)",
            Provider.NANO_GPT,
            "wan2.7-image",
            ratio_overrides=_WAN_27_RATIOS,
        ),
        GenerateModel(
            "WAN 2.7 Image Pro (NanoGPT)",
            Provider.NANO_GPT,
            "wan2.7-image-pro",
            ratio_overrides=_WAN_27_RATIOS,
        ),
    ]
)
