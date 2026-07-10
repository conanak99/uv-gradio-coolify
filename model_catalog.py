from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class Provider(str, Enum):
    FAL = "fal.ai"
    NANO_GPT = "NanoGPT"


class Operation(str, Enum):
    EDIT = "Edit"
    GENERATE = "Generate"

    @property
    def log_name(self) -> str:
        return self.value.lower()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    api_id: str
    provider: Provider
    operation: Operation
    edit_aspect_ratios: tuple[str, ...] = ()
    aspect_ratio_map: Mapping[str, str] = field(default_factory=dict)
    resolutions: Mapping[str, str] = field(default_factory=dict)

    def generation_parameters(self, requested_ratio: str) -> tuple[str, str]:
        output_ratio = self.aspect_ratio_map.get(requested_ratio, requested_ratio)
        resolution = self.resolutions.get(output_ratio, output_ratio)
        return output_ratio, resolution


SEEDREAM_PRO_EDIT_MODEL_ID = "bytedance/seedream-v5.0-pro/edit"
WAN_26_EDIT_MODEL_ID = "wan-2.6-image-edit"
GROK_GENERATE_MODEL_ID = "xai/grok-imagine-image"
SEEDREAM_LITE_GENERATE_MODEL_ID = "seedream-v5.0-lite"
SEEDREAM_PRO_GENERATE_MODEL_ID = "bytedance/seedream-v5.0-pro"
WAN_27_IMAGE_MODEL_ID = "wan2.7-image"
WAN_27_IMAGE_PRO_MODEL_ID = "wan2.7-image-pro"

SEEDREAM_PRO_EDIT_MODEL = "Seedream 5.0 Pro Edit (NanoGPT)"
WAN_26_EDIT_MODEL = "WAN 2.6 Image Edit (NanoGPT)"
GROK_GENERATE_MODEL = "Grok Imagine (fal.ai)"
SEEDREAM_LITE_GENERATE_MODEL = "Seedream 5.0 Lite (NanoGPT)"
SEEDREAM_PRO_GENERATE_MODEL = "Seedream 5.0 Pro (NanoGPT)"
WAN_27_GENERATE_MODEL = "WAN 2.7 Image (NanoGPT)"
WAN_27_PRO_GENERATE_MODEL = "WAN 2.7 Image Pro (NanoGPT)"

GENERATE_ASPECT_RATIOS = ("16:9", "4:3", "1:1", "3:4", "9:16")
SEEDREAM_PRO_EDIT_ASPECT_RATIOS = (
    "1:1",
    "16:9",
    "9:16",
    "3:2",
    "2:3",
    "4:3",
    "3:4",
)
NEAREST_GENERATE_RATIO = {
    "16:9": "16:9",
    "4:3": "3:2",
    "1:1": "1:1",
    "3:4": "2:3",
    "9:16": "9:16",
}
SEEDREAM_LITE_RESOLUTIONS = {
    "1:1": "2048x2048",
    "16:9": "2560x1440",
    "9:16": "1440x2560",
    "3:2": "3072x2048",
    "2:3": "2048x3072",
}
WAN_27_RESOLUTIONS = {
    "1:1": "1024*1024",
    "16:9": "1280*720",
    "9:16": "720*1280",
    "3:2": "1536*1024",
    "2:3": "1024*1536",
}

MODEL_SPECS = (
    ModelSpec(
        name="Qwen Image Edit",
        api_id="fal-ai/qwen-image-edit-2511",
        provider=Provider.FAL,
        operation=Operation.EDIT,
    ),
    ModelSpec(
        name="FLUX.2 Klein 9B Edit",
        api_id="fal-ai/flux-2/klein/9b/edit",
        provider=Provider.FAL,
        operation=Operation.EDIT,
    ),
    ModelSpec(
        name="Grok Imagine Image Edit",
        api_id="xai/grok-imagine-image/edit",
        provider=Provider.FAL,
        operation=Operation.EDIT,
    ),
    ModelSpec(
        name=SEEDREAM_PRO_EDIT_MODEL,
        api_id=SEEDREAM_PRO_EDIT_MODEL_ID,
        provider=Provider.NANO_GPT,
        operation=Operation.EDIT,
        edit_aspect_ratios=SEEDREAM_PRO_EDIT_ASPECT_RATIOS,
    ),
    ModelSpec(
        name=WAN_26_EDIT_MODEL,
        api_id=WAN_26_EDIT_MODEL_ID,
        provider=Provider.NANO_GPT,
        operation=Operation.EDIT,
    ),
    ModelSpec(
        name=GROK_GENERATE_MODEL,
        api_id=GROK_GENERATE_MODEL_ID,
        provider=Provider.FAL,
        operation=Operation.GENERATE,
    ),
    ModelSpec(
        name=SEEDREAM_LITE_GENERATE_MODEL,
        api_id=SEEDREAM_LITE_GENERATE_MODEL_ID,
        provider=Provider.NANO_GPT,
        operation=Operation.GENERATE,
        aspect_ratio_map=NEAREST_GENERATE_RATIO,
        resolutions=SEEDREAM_LITE_RESOLUTIONS,
    ),
    ModelSpec(
        name=SEEDREAM_PRO_GENERATE_MODEL,
        api_id=SEEDREAM_PRO_GENERATE_MODEL_ID,
        provider=Provider.NANO_GPT,
        operation=Operation.GENERATE,
    ),
    ModelSpec(
        name=WAN_27_GENERATE_MODEL,
        api_id=WAN_27_IMAGE_MODEL_ID,
        provider=Provider.NANO_GPT,
        operation=Operation.GENERATE,
        aspect_ratio_map=NEAREST_GENERATE_RATIO,
        resolutions=WAN_27_RESOLUTIONS,
    ),
    ModelSpec(
        name=WAN_27_PRO_GENERATE_MODEL,
        api_id=WAN_27_IMAGE_PRO_MODEL_ID,
        provider=Provider.NANO_GPT,
        operation=Operation.GENERATE,
        aspect_ratio_map=NEAREST_GENERATE_RATIO,
        resolutions=WAN_27_RESOLUTIONS,
    ),
)


def models_for(operation: Operation) -> dict[str, ModelSpec]:
    return {
        model.name: model
        for model in MODEL_SPECS
        if model.operation is operation
    }


EDIT_MODELS = models_for(Operation.EDIT)
GENERATE_MODELS = models_for(Operation.GENERATE)
