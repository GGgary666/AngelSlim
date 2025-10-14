# AngelSlim Diffusion Model Compression Example

AngelSlim provides flexible and efficient tools for compressing DiT diffusion models. Our quantization utilities are highly modular, allowing seamless integration into various inference pipelines.

## Quick Example: FP8 Quantization for Diffusion

```python
import torch
from diffusers import FluxPipeline
from angelslim.compressor.diffusion import quantize_model_to_fp8

# We recommend loading the model using torch.bfloat16 for best compatibility
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)

# Choose quantization type: "fp8-per-tensor", "fp8-per-token", or "fp8-per-block"
quantize_model_to_fp8(pipe.transformer, "fp8-per-token")

# Move pipeline to GPU (optional)
pipe.to("cuda")

# Run inference as usual
image = pipe(
    "A cat holding a sign that says hello world",
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]
image.save("flux-schnell_fp8_per_token.png")
```

## Flexible Quantization Layer Filtering

To ensure optimal accuracy, AngelSlim supports flexible filtering for quantized linear layers. You can customize which layers are quantized using filters or pattern matching.

```python
# Method 1: Default filtering rules (quantizes typical linear layers)
quantize_model_to_fp8(model, "fp8-per-tensor")

# Method 2: Specify include/exclude patterns as strings
quantize_model_to_fp8(
    model,
    "fp8-per-tensor",
    include_patterns=["linear", "attention"],
    exclude_patterns=["embed", "norm"]
)

# Method 3: Use regular expression patterns (automatically recognized)
quantize_model_to_fp8(
    model,
    "fp8-per-tensor",
    include_patterns=[r".*\.linear\d+", r".*\.attn.*"],
    exclude_patterns=[r".*embed.*"]
)

# Method 4: Mix string and regex patterns for more fine-grained control
quantize_model_to_fp8(
    model,
    "fp8-per-tensor",
    include_patterns=["linear", r".*\.attn.*"],
    exclude_patterns=["embed", r".*norm.*"]
)
```

> **Tip:** You may also provide a custom Python function as a filter (see the detailed API docs for more options).