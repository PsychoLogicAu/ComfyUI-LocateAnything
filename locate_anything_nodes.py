# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""
Locate Anything custom nodes for ComfyUI bounding box detection.

This module provides custom nodes for:
- Object detection
- Text grounding
- Phrase grounding
- Pointing
- GUI grounding
- Debugging
"""

import os
import json
import logging
import numpy as np
from PIL import Image, ImageDraw
import torch

from .locateanything_worker import LocateAnythingModel, map_dtype_to_torch

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache: model_path:dtype -> LocateAnythingModel instance
_WORKER_CACHE: dict[str, "LocateAnythingModel"] = {}


def _get_or_create_model(
    model_path: str,
    dtype_str: str,
    trust_remote_code: bool,
    attention_implementation: str,
) -> "LocateAnythingModel":
    """Get or create a LocateAnythingModel from the cache."""
    cache_key = f"{model_path}:{dtype_str}:{attention_implementation}"

    if cache_key not in _WORKER_CACHE:
        dtype = map_dtype_to_torch(dtype_str)
        model = LocateAnythingModel(
            model_path=model_path,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attention_implementation=attention_implementation,
        )
        _WORKER_CACHE[cache_key] = model
        logger.info(f"Loaded LocateAnythingModel: {model_path}")
    return _WORKER_CACHE[cache_key]


# ──────────────────────────────────────────────────────────────────────────────
# Tensor / PIL helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI image tensor to PIL Image (RGB)."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    elif tensor.dim() != 3:
        raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {tensor.dim()}")

    tensor = tensor.clamp(0, 1).mul(255).byte()
    numpy_image = tensor.cpu().numpy().astype(np.uint8)
    return Image.fromarray(numpy_image, mode="RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image back to ComfyUI tensor format [1, H, W, C]."""
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _draw_boxes(
    image: Image.Image, boxes: list[dict], color: str = "#FF0000", width: int = 3
) -> Image.Image:
    """Draw bounding boxes on image."""
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=width)
    return image


def _draw_points(
    image: Image.Image, points: list[dict], fill: str = "#FF0000", outline: str = "#00FF00"
) -> Image.Image:
    """Draw points on image."""
    draw = ImageDraw.Draw(image)
    for pt in points:
        x, y = int(pt["x"]), int(pt["y"])
        r = 10
        draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=outline, fill=fill, width=2)
    return image


# ──────────────────────────────────────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────────────────────────────────────

class LocateAnythingLoader:
    """Load Locate Anything model.

    Attention implementation is set here and baked into the model config.
    There is no need to specify it again on inference nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "nvidia/LocateAnything-3B",
                    "multiline": False,
                    "placeholder": "Model path or HuggingFace repo ID",
                }),
                "dtype": (
                    ["auto", "bfloat16", "float16", "float32"],
                    {"default": "auto"},
                ),
                "trust_remote_code": ("BOOLEAN", {"default": True}),
                "attention_implementation": (
                    ["sdpa", "flash_attention_2", "magi_attention", "eager"],
                    {
                        "default": "sdpa",
                        "tooltip": "Attention implementation. Baked into model config at load time.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("locate_anything_model",)
    RETURN_NAMES = ("locate_anything",)
    FUNCTION = "load_model"
    CATEGORY = "Locate Anything/Loader"

    def load_model(self, model_path, dtype, trust_remote_code, attention_implementation="sdpa"):
        model = _get_or_create_model(model_path, dtype, trust_remote_code, attention_implementation)
        return (model,)


class LocateAnythingConfig:
    """Configure inference parameters.

    Attention implementation cannot be changed here — it's set on the Loader
    and baked into the model config at load time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "generation_mode": (["hybrid", "direct"], {"default": "hybrid"}),
                "use_cache": ("BOOLEAN", {"default": True}),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("locate_anything_config",)
    RETURN_NAMES = ("config",)
    FUNCTION = "configure"
    CATEGORY = "Locate Anything/Config"

    def configure(
        self,
        max_new_tokens,
        temperature,
        top_p,
        do_sample,
        generation_mode,
        use_cache,
        repetition_penalty,
    ):
        config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "generation_mode": generation_mode,
            "use_cache": use_cache,
            "repetition_penalty": repetition_penalty,
        }
        return (config,)


class _InferenceNode:
    """Mixin that resolves the model + config and runs inference."""

    @staticmethod
    def _resolve_model_and_config(locate_anything, config):
        """Return (LocateAnythingModel, inference_kwargs)."""
        if config is None:
            config = {}

        inference_kw = {
            "max_new_tokens": config.get("max_new_tokens", 2048),
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.9),
            "do_sample": config.get("do_sample", True),
            "generation_mode": config.get("generation_mode", "hybrid"),
            "use_cache": config.get("use_cache", True),
            "repetition_penalty": config.get("repetition_penalty", 1.1),
        }
        return locate_anything, inference_kw


class LocateAnythingDetector(_InferenceNode):
    """Detect objects in image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "locate_anything": ("locate_anything_model",),
                "image": ("IMAGE",),
                "categories": (
                    "STRING",
                    {
                        "default": "chair, person, car, dog, laptop",
                        "multiline": True,
                        "placeholder": "Comma-separated categories",
                    },
                ),
            },
            "optional": {"config": ("locate_anything_config",)},
        }

    RETURN_TYPES = ("text", "text", "IMAGE")
    RETURN_NAMES = ("detection_result", "parsed_boxes", "annotated_image")
    FUNCTION = "detect"
    CATEGORY = "Locate Anything/Detection"

    def detect(self, locate_anything, image, categories, config=None):
        model, kw = self._resolve_model_and_config(locate_anything, config)
        cat_list = [c.strip() for c in categories.split(",")]
        pil = _tensor_to_pil(image[0])
        result = model.detect(pil, categories=cat_list, **kw)
        boxes = LocateAnythingModel.parse_boxes(result["answer"], pil.size[0], pil.size[1])
        annotated = _pil_to_tensor(_draw_boxes(pil, boxes))
        return (result["answer"], json.dumps(boxes, indent=2), annotated)


class LocateAnythingGroundPhrase(_InferenceNode):
    """Ground phrase instances in image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "locate_anything": ("locate_anything_model",),
                "image": ("IMAGE",),
                "phrase": (
                    "STRING",
                    {"default": "chair", "multiline": False, "placeholder": "Object to ground"},
                ),
            },
            "optional": {"config": ("locate_anything_config",)},
        }

    RETURN_TYPES = ("text", "text", "IMAGE")
    RETURN_NAMES = ("grounding_result", "parsed_boxes", "annotated_image")
    FUNCTION = "ground_phrase"
    CATEGORY = "Locate Anything/Grounding/Phrase"

    def ground_phrase(self, locate_anything, image, phrase, config=None):
        model, kw = self._resolve_model_and_config(locate_anything, config)
        pil = _tensor_to_pil(image[0])
        result = model.ground_multi(pil, phrase=phrase, **kw)
        boxes = LocateAnythingModel.parse_boxes(result["answer"], pil.size[0], pil.size[1])
        annotated = _pil_to_tensor(_draw_boxes(pil, boxes))
        return (result["answer"], json.dumps(boxes, indent=2), annotated)


class LocateAnythingGroundText(_InferenceNode):
    """Ground text in images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "locate_anything": ("locate_anything_model",),
                "image": ("IMAGE",),
                "phrase": (
                    "STRING",
                    {
                        "default": "text",
                        "multiline": False,
                        "placeholder": "Text to locate",
                    },
                ),
            },
            "optional": {"config": ("locate_anything_config",)},
        }

    RETURN_TYPES = ("text", "text", "IMAGE")
    RETURN_NAMES = ("grounding_result", "parsed_boxes", "annotated_image")
    FUNCTION = "ground_text"
    CATEGORY = "Locate Anything/Grounding/Text"

    def ground_text(self, locate_anything, image, phrase, config=None):
        model, kw = self._resolve_model_and_config(locate_anything, config)
        pil = _tensor_to_pil(image[0])
        result = model.ground_text(pil, phrase=phrase, **kw)
        boxes = LocateAnythingModel.parse_boxes(result["answer"], pil.size[0], pil.size[1])
        annotated = _pil_to_tensor(_draw_boxes(pil, boxes))
        return (result["answer"], json.dumps(boxes, indent=2), annotated)


class LocateAnythingPoint(_InferenceNode):
    """Point to objects in images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "locate_anything": ("locate_anything_model",),
                "image": ("IMAGE",),
                "phrase": (
                    "STRING",
                    {
                        "default": "the traffic light",
                        "multiline": False,
                        "placeholder": "Object to point to",
                    },
                ),
            },
            "optional": {"config": ("locate_anything_config",)},
        }

    RETURN_TYPES = ("text", "text", "IMAGE")
    RETURN_NAMES = ("pointing_result", "parsed_points", "annotated_image")
    FUNCTION = "point"
    CATEGORY = "Locate Anything/Pointing"

    def point(self, locate_anything, image, phrase, config=None):
        model, kw = self._resolve_model_and_config(locate_anything, config)
        pil = _tensor_to_pil(image[0])
        result = model.point(pil, phrase=phrase, **kw)
        pts = LocateAnythingModel.parse_points(result["answer"], pil.size[0], pil.size[1])
        annotated = _pil_to_tensor(_draw_points(pil, pts))
        return (result["answer"], json.dumps(pts, indent=2), annotated)


class LocateAnythingGUIGround(_InferenceNode):
    """Ground UI elements for GUI grounding."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "locate_anything": ("locate_anything_model",),
                "image": ("IMAGE",),
                "phrase": (
                    "STRING",
                    {
                        "default": "the search button",
                        "multiline": False,
                        "placeholder": "UI element description",
                    },
                ),
                "output_type": (["box", "point"], {"default": "box"}),
            },
            "optional": {"config": ("locate_anything_config",)},
        }

    RETURN_TYPES = ("text", "text", "IMAGE")
    RETURN_NAMES = ("gui_result", "parsed_output", "annotated_image")
    FUNCTION = "gui_ground"
    CATEGORY = "Locate Anything/GUI Grounding"

    def gui_ground(
        self,
        locate_anything,
        image,
        phrase,
        output_type,
        config=None,
    ):
        model, kw = self._resolve_model_and_config(locate_anything, config)
        pil = _tensor_to_pil(image[0])
        result = model.ground_gui(pil, phrase=phrase, output_type=output_type, **kw)
        answer = result["answer"]
        if output_type == "box":
            parsed = LocateAnythingModel.parse_boxes(answer, pil.size[0], pil.size[1])
            annotated = _pil_to_tensor(_draw_boxes(pil, parsed))
        else:
            parsed = LocateAnythingModel.parse_points(answer, pil.size[0], pil.size[1])
            annotated = _pil_to_tensor(_draw_points(pil, parsed))
        return (answer, json.dumps(parsed, indent=2), annotated)


class LocateAnythingDebug(_InferenceNode):
    """Debug node to inspect model and inference state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "locate_anything": ("locate_anything_model",),
                "image": ("IMAGE",),
                "test_phrase": (
                    "STRING",
                    {"default": "chair", "multiline": False, "placeholder": "Test phrase"},
                ),
                "test_categories": (
                    "STRING",
                    {
                        "default": "chair, person, car, dog, laptop",
                        "multiline": False,
                        "placeholder": "Test categories",
                    },
                ),
            }
        }

    RETURN_TYPES = ("text", "IMAGE", "text")
    RETURN_NAMES = ("model_info", "debug_image", "debug_output")
    FUNCTION = "debug_model"
    CATEGORY = "Locate Anything/Debug"

    def debug_model(self, locate_anything, image, test_phrase, test_categories):
        model_info = {
            "model_path": locate_anything.model_path,
            "load_device": str(locate_anything.load_device),
            "offload_device": str(locate_anything.offload_device),
            "dtype": str(locate_anything.dtype),
        }

        model, kw = self._resolve_model_and_config(
            locate_anything, {"max_new_tokens": 256, "temperature": 0.0}
        )
        pil = _tensor_to_pil(image[0])
        debug_text = ""

        try:
            result = model.ground_multi(pil, phrase=test_phrase, **kw)
            boxes = LocateAnythingModel.parse_boxes(result["answer"], pil.size[0], pil.size[1])
            debug_image = _draw_boxes(pil, boxes)
            debug_text = f"Phrase Grounding Test:\nInput: '{test_phrase}'\nOutput: {result['answer'][:200]}..."
        except Exception as e:
            debug_image = pil
            debug_text = f"Phrase Grounding Error: {e}"

        try:
            result = model.detect(pil, categories=test_categories.split(","), **kw)
            debug_text += f"\n\nObject Detection Test:\nCategories: {test_categories}\nOutput: {result['answer'][:200]}..."
        except Exception as e:
            debug_text += f"\n\nObject Detection Error: {e}"

        return (json.dumps(model_info, indent=2), _pil_to_tensor(debug_image), debug_text)


# ──────────────────────────────────────────────────────────────────────────────
# Node Registration
# ──────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LocateAnythingLoader": LocateAnythingLoader,
    "LocateAnythingConfig": LocateAnythingConfig,
    "LocateAnythingDetector": LocateAnythingDetector,
    "LocateAnythingGroundPhrase": LocateAnythingGroundPhrase,
    "LocateAnythingGroundText": LocateAnythingGroundText,
    "LocateAnythingPoint": LocateAnythingPoint,
    "LocateAnythingGUIGround": LocateAnythingGUIGround,
    "LocateAnythingDebug": LocateAnythingDebug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LocateAnythingLoader": "Load Locate Anything Model",
    "LocateAnythingConfig": "Configure Inference",
    "LocateAnythingDetector": "Detect Objects",
    "LocateAnythingGroundPhrase": "Ground Phrase",
    "LocateAnythingGroundText": "Ground Text",
    "LocateAnythingPoint": "Point to Object",
    "LocateAnythingGUIGround": "GUI Grounding",
    "LocateAnythingDebug": "Debug Model",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]