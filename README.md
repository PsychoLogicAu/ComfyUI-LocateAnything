# ComfyUI-LocateAnything

ComfyUI custom nodes for NVIDIA's [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) model - a vision-language model for zero-shot object detection and localization with bounding boxes.

**LocateAnything-3B** is part of the [EAGLE](https://github.com/NVlabs/Eagle) project from [NVIDIA AI Research](https://www.nvidia.com/en-us/research/), which provides a unified, zero-shot grounding model capable of detecting and localizing objects described by natural language prompts.

## References

| Resource | Link |
|----------|------|
| **HuggingFace Model** | [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) |
| **NVIDIA Eagle GitHub** | [NVlabs/Eagle](https://github.com/NVlabs/Eagle) |
| **EAGLE Paper** | [arXiv:2501.14288](https://arxiv.org/abs/2501.14288) |

## Features

- **Automatic Model Download**: Downloads the LocateAnything-3B model from HuggingFace on first use
- **Zero-Shot Object Detection**: Detect objects using text prompts without training
- **Phrase Grounding**: Locate all instances matching a natural language phrase
- **Text Grounding**: Find and localize text within images
- **Pointing**: Get point coordinates for objects described by phrases
- **GUI Grounding**: Detect and localize UI elements in screenshots
- **Configurable Inference**: Adjustable temperature, top_p, max tokens, generation mode, and more
- **Multiple Backend Support**: SDPA, Flash Attention 2, Magi Attention, and eager modes
- **Model Caching**: Models are cached to avoid redundant loading across workflows
- **Debug Visualization**: Overlay bounding boxes and points for visual verification

## Nodes

### Loader & Config

| Node | Display Name | Description |
|------|-------|-------------|
| **LocateAnythingLoader** | Load Locate Anything Model | Loads the LocateAnything-3B model from HuggingFace with configurable device, dtype, and attention backend |
| **LocateAnythingConfig** | Configure Inference | Creates inference configuration with parameters for max tokens, temperature, top_p, sampling, generation mode, caching, and repetition penalty |

### Detection

| Node | Display Name | Description |
|------|-------|-------------|
| **LocateAnythingDetector** | Detect Objects | Detects objects in an image using comma-separated category names (e.g., "chair, person, car, dog, laptop") |

### Grounding

| Node | Display Name | Description |
|------|-------|-------------|
| **LocateAnythingGroundPhrase** | Ground Phrase | Locates all instances in the image matching a given phrase (e.g., "chair", "the red car") |
| **LocateAnythingGroundText** | Ground Text | Finds and localizes text within images (e.g., "STOP" or generic "text") |

### Pointing & GUI

| Node | Display Name | Description |
|------|-------|-------------|
| **LocateAnythingPoint** | Point to Object | Returns point coordinates for objects described by a phrase (e.g., "the traffic light") |
| **LocateAnythingGUIGround** | GUI Grounding | Detects UI elements in screenshots, with configurable output as bounding box or point |

### Debug

| Node | Display Name | Description |
|------|-------|-------------|
| **LocateAnythingDebug** | Debug Model | Inspects model loading state and runs test grounding/detection with visualized results |

## Installation

1. Clone the repository into your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-username/ComfyUI-LocateAnything.git
   cd ComfyUI-LocateAnything
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

4. Load the model in your workflow:
   - Add a **Load Locate Anything Model** node
   - Configure model path (default: `nvidia/LocateAnything-3B`), device, and dtype
   - The model will be downloaded from HuggingFace on first use

## Usage

### Basic Object Detection Workflow

1. Add a **Load Locate Anything Model** node and configure:
   - `model_path`: `nvidia/LocateAnything-3B` (or a local path)
   - `device`: `cuda` (or `cpu`)
   - `dtype`: `auto` (recommended), `bfloat16`, `float16`, or `float32`
   - `attention_implementation`: `sdpa` (default), `flash_attention_2`, `magi_attention`, or `eager`

2. (Optional) Add a **Configure Inference** node to customize:
   - `max_new_tokens`: Max output tokens (default: 2048)
   - `temperature`: Sampling temperature (default: 0.7)
   - `top_p`: Nucleus sampling threshold (default: 0.9)
   - `do_sample`: Enable sampling (default: true)
   - `generation_mode`: `hybrid` or `direct`
   - `use_cache`: Enable KV caching (default: true)
   - `repetition_penalty`: Penalty for repeated tokens (default: 1.1)

3. Add a **Detect Objects** node:
   - Connect the model output and your image
   - Set `categories` to comma-separated object names (e.g., "person, car, dog")

4. The node outputs:
   - `detection_result`: Raw model output text
   - `parsed_boxes`: JSON-formatted bounding box coordinates
   - `annotated_image`: Image with bounding boxes drawn in red

### Phrase Grounding Workflow

Use the **Ground Phrase** node to find all instances matching a natural language description:

1. Connect model and image inputs
2. Set `phrase` to the object description (e.g., "the red car", "a chair")
3. Adjust `confidence` threshold if needed

### Text Grounding Workflow

Use the **Ground Text** node to locate text within images:

1. Connect model and image inputs
2. Set `phrase` to specific text to find (e.g., "STOP") or use generic "text"

### Pointing Workflow

Use the **Point to Object** node to get point coordinates:

1. Connect model and image inputs
2. Set `phrase` to describe the target object
3. Points are drawn as red circles with green outlines on the annotated image

### GUI Grounding Workflow

Use the **GUI Grounding** node to detect UI elements:

1. Connect model and image inputs
2. Set `phrase` to describe the UI element (e.g., "the search button")
3. Choose `output_type`: `box` for bounding boxes or `point` for coordinates

## Output Format

### Bounding Box Output (JSON)

The `parsed_boxes` output is a JSON array of bounding box objects:

```json
[
  {
    "x1": 120,
    "y1": 80,
    "x2": 340,
    "y2": 520,
    "label": "person"
  },
  ...
]
```

Coordinates are in pixel space relative to the input image dimensions.

### Point Output (JSON)

The `parsed_points` output is a JSON array of point objects:

```json
[
  {
    "x": 256,
    "y": 180,
    "label": "the traffic light"
  },
  ...
]
```

## Model Information

- **Model**: NVIDIA LocateAnything-3B
- **HuggingFace**: [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)
- **Upstream Repo**: [NVlabs/Eagle](https://github.com/NVlabs/Eagle)
- **Paper**: [EAGLE: Embodied Generalist with Action Generation and Language Grounding](https://arxiv.org/abs/2501.14288)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- Pillow (PIL)
- torchvision
- torchtyping
- sentencepiece

## License

This project is provided as-is for use with ComfyUI. The LocateAnything model is licensed under its original license from NVIDIA. See the [HuggingFace model card](https://huggingface.co/nvidia/LocateAnything-3B) for licensing details.

## Troubleshooting

- **Model download fails**: Check your internet connection and ensure you have access to the HuggingFace model
- **CUDA out of memory**: Try using `float16` dtype, switch to CPU inference, or reduce `max_new_tokens`
- **Flash Attention errors**: Fall back to `sdpa` or `eager` attention implementation
- **No detections**: Try a simpler prompt, increase `max_new_tokens`, or adjust `temperature`

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.