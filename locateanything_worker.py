"""
Locate Anything Worker module.

This module provides:
- LocateAnythingModel wrapper class (ComfyUI-compatible memory management)
- Worker class for inference
- Object detection, phrase grounding, text grounding, pointing, GUI grounding
"""

import re
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

import comfy.model_management
import comfy.model_patcher


def map_dtype_to_torch(dtype_str):
    """Map dtype config string to torch.dtype.

    Args:
        dtype_str: String like "auto", "fp32", "fp16", "bf16", "int8", or None

    Returns:
        torch.dtype or None (to let ComfyUI decide)
    """
    if dtype_str is None or dtype_str == "auto":
        return None  # Let ComfyUI decide

    dtype_lower = dtype_str.lower()

    if dtype_lower in ("fp32", "float32", "float"):
        return torch.float32
    elif dtype_lower in ("fp16", "float16", "half"):
        return torch.float16
    elif dtype_lower in ("bf16", "bfloat16", "bfloat"):
        return torch.bfloat16
    elif dtype_lower == "int8":
        return torch.int8

    return None


class _HFModelProxy:
    """Thin wrapper around a HuggingFace model that provides a settable 'device' property.

    CoreModelPatcher.load() expects to set model.device = device_to to move the
    model between GPU and CPU. HF models expose 'device' as a read-only property
    derived from their parameters, so we intercept the setter and call .to(device).
    All other attribute access is forwarded transparently.
    """

    def __init__(self, model):
        object.__setattr__(self, "_model", model)

    @property
    def device(self):
        return self._model.device

    @device.setter
    def device(self, value):
        self._model.to(value)

    @property
    def training(self):
        return self._model.training

    def __getattr__(self, name):
        return getattr(self._model, name)

class LocateAnythingModel:
    """ComfyUI-compatible wrapper for LocateAnything model.

    Follows the same pattern as BackgroundRemovalModel (comfy/bg_removal_model.py):
    - Uses CoreModelPatcher for proper GPU memory tracking and eviction.
    - Uses model_management for device/dtype selection.
    - Model is loaded on offload_device and moved to GPU only when needed.
    """

    def __init__(self, model_path, dtype=None, trust_remote_code=True,
                 attention_implementation=None):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.attention_implementation = attention_implementation

        # --- Device / dtype selection via ComfyUI model_management ---
        # LocateAnything is a large multimodal encoder ~3B params, similar in size
        # to text encoders, so we use the text_encoder device/dtype helpers.
        self.load_device = comfy.model_management.text_encoder_device()
        self.offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)

        # If the user explicitly requested a different dtype, honour it.
        if dtype is not None:
            self.dtype = dtype

        # --- Load tokenizer and processor (lightweight, keep on CPU) ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code)

        # --- Load config and override attention implementation ---
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code)

        if attention_implementation:
            config._attn_implementation = attention_implementation
        else:
            config._attn_implementation = "sdpa"

        # --- Load model on the offload_device (CPU) ---
        hf_model = AutoModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=self.dtype,
            trust_remote_code=trust_remote_code,
        ).to(self.offload_device).eval()

        # Wrap in proxy so CoreModelPatcher can set model.device = device_to.
        # HF models expose 'device' as a read-only property, so the proxy
        # intercepts the setter and delegates to model.to(device).
        self.model = _HFModelProxy(hf_model)

        # --- Wrap in CoreModelPatcher for ComfyUI memory management ---
        self.patcher = comfy.model_patcher.CoreModelPatcher(
            self.model,
            load_device=self.load_device,
            offload_device=self.offload_device,
        )

    def _get_model(self):
        """Return the raw model (used internally after loading to GPU)."""
        return self.model

    @torch.no_grad()
    def predict(
        self,
        image,
        question,
        generation_mode="hybrid",
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        verbose=True,
    ):
        """Run inference with ComfyUI-compatible memory management.

        Calls ``load_model_gpu()`` before inference so ComfyUI can manage
        GPU memory (evict other models if needed, track usage, etc.).
        """
        # --- Load model to GPU via ComfyUI memory management ---
        comfy.model_management.load_model_gpu(self.patcher)

        model = self._get_model()

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.load_device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,  # model's generate() asserts use_cache=True
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            verbose=verbose,
        )

        result = {"answer": response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        return result

    # ---- Convenience methods for each task ----

    def detect(self, image, categories, **kwargs):
        """Object detection / document layout analysis."""
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def ground_single(self, image, phrase, **kwargs):
        """Phrase grounding — single instance."""
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_multi(self, image, phrase, **kwargs):
        """Phrase grounding — multiple instances."""
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_text(self, image, phrase, **kwargs):
        """Text grounding."""
        prompt = f"Please locate the text referred as {phrase}."
        return self.predict(image, prompt, **kwargs)

    def detect_text(self, image, **kwargs):
        """Scene text detection."""
        prompt = "Detect all the text in box format."
        return self.predict(image, prompt, **kwargs)

    def ground_gui(self, image, phrase, output_type="box", **kwargs):
        """GUI grounding (box or point)."""
        if output_type == "point":
            prompt = f"Point to: {phrase}."
        else:
            prompt = f"Locate the region that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def point(self, image, phrase, **kwargs):
        """Pointing."""
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)

    # ---- Utility: parse model output ----

    @staticmethod
    def parse_boxes(answer, image_width, image_height):
        """Parse model output into pixel-coordinate bounding boxes.

        Coordinates in model output are normalized integers in [0, 1000].
        Output coordinates are rounded to integers.
        """
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append({
                "x1": int(round(x1 / 1000 * image_width)),
                "y1": int(round(y1 / 1000 * image_height)),
                "x2": int(round(x2 / 1000 * image_width)),
                "y2": int(round(y2 / 1000 * image_height)),
            })
        return boxes

    @staticmethod
    def parse_points(answer, image_width, image_height):
        """Parse model output into pixel-coordinate points."""
        points = []
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append({
                "x": x / 1000 * image_width,
                "y": y / 1000 * image_height,
            })
        return points

    @staticmethod
    def parse_boxes_with_labels(answer, image_width, image_height):
        """Parse model output into labeled bounding boxes grouped by category.

        Returns a dict: {"label": [{"x1": int, "y1": int, "x2": int, "y2": int}, ...], ...}
        Coordinates are rounded to integers.
        """
        result = {}
        current_label = "unknown"

        # Track all <ref> tags to know when we switch labels
        refs = list(re.finditer(r"<ref>([^<]+)</ref>", answer))
        boxes = list(re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer))

        box_idx = 0
        for ref_match in refs:
            label = ref_match.group(1).strip()
            # Find all boxes that appear after this ref and before the next ref
            label_boxes = []
            next_ref_start = refs[refs.index(ref_match) + 1].start() if ref_match != refs[-1] else len(answer)
            
            while box_idx < len(boxes) and boxes[box_idx].start() < next_ref_start:
                b = boxes[box_idx]
                x1, y1, x2, y2 = [int(g) for g in b.groups()]
                label_boxes.append({
                    "x1": int(round(x1 / 1000 * image_width)),
                    "y1": int(round(y1 / 1000 * image_height)),
                    "x2": int(round(x2 / 1000 * image_width)),
                    "y2": int(round(y2 / 1000 * image_height)),
                })
                box_idx += 1

            if label_boxes:
                result[label] = label_boxes

        # Handle boxes after the last ref (use last label)
        if box_idx < len(boxes) and refs:
            last_label = refs[-1].group(1).strip()
            existing = result.get(last_label, [])
            while box_idx < len(boxes):
                b = boxes[box_idx]
                x1, y1, x2, y2 = [int(g) for g in b.groups()]
                existing.append({
                    "x1": int(round(x1 / 1000 * image_width)),
                    "y1": int(round(y1 / 1000 * image_height)),
                    "x2": int(round(x2 / 1000 * image_width)),
                    "y2": int(round(y2 / 1000 * image_height)),
                })
                box_idx += 1
            result[last_label] = existing

        return result


# ─────────────────────────────────────────────────────────────
# Legacy alias: LocateAnythingWorker -> LocateAnythingModel
# Kept for backward compatibility with existing node code.
# ─────────────────────────────────────────────────────────────

class LocateAnythingWorker(LocateAnythingModel):
    """Deprecated alias for LocateAnythingModel.

    This exists purely for backward compatibility. New code should use
    LocateAnythingModel directly.
    """
    pass