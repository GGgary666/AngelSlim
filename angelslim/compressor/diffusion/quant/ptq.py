# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import re
from typing import List, Optional, Union

import torch
import tqdm

from .modules import FP8DynamicLinear
from .quantizers import (
    fp8_per_block_quant,
    fp8_per_tensor_quant,
    fp8_per_token_group_quant,
)
from .utils import (
    _ensure_deep_gemm,
    cleanup_memory,
    create_default_layer_filter,
    replace_module,
    should_quantize_layer,
)

__all__ = ["quantize_model_to_fp8"]


def quantize_model_to_fp8(
    model: torch.nn.Module,
    quant_type: str = "fp8-per-tensor",
    layer_filter: Optional[callable] = None,
    include_patterns: Optional[List[Union[str, re.Pattern]]] = None,
    exclude_patterns: Optional[List[Union[str, re.Pattern]]] = None,
):
    """
    Quantize a PyTorch model to FP8 formats.

    Args:
        model: The PyTorch model to be quantized.
        quant_type: Quantization type, one of
            "fp8-per-tensor", "fp8-per-token", "fp8-per-block".
        layer_filter: A callable filter for deciding whether a
            layer should be quantized.
        include_patterns: List of patterns (string or regex) to include.
        exclude_patterns: List of patterns (string or regex) to exclude.

    Examples:
        # Default filtering rules
        quantize_model_to_fp8(model, "fp8-per-tensor")

        # String patterns (special chars auto-escaped)
        quantize_model_to_fp8(
            model,
            "fp8-per-tensor",
            include_patterns=["linear", "attention"],
            exclude_patterns=["embed", "norm"],
        )

        # Regex patterns (auto-detected)
        quantize_model_to_fp8(
            model,
            "fp8-per-tensor",
            include_patterns=[".*\\.linear\\d+", ".*\\.attn.*"],
            exclude_patterns=[".*embed.*"]
        )

        # Mixed string and regex patterns
        quantize_model_to_fp8(
            model,
            "fp8-per-tensor",
            include_patterns=["linear", ".*\\.attn.*"],
            exclude_patterns=["embed", ".*norm.*"]
        )

        # Custom filter function
        custom_filter = lambda name: "attention" in name and "norm" not in name
        quantize_model_to_fp8(model, "fp8-per-tensor", layer_filter=custom_filter)
    """
    assert quant_type in [
        "fp8-per-tensor",
        "fp8-per-token",
        "fp8-per-block",
    ], f"Invalid quant_type: {quant_type}"
    # if quant_type == "fp8-per-token":
    # from abo_kernel_lib import abo_fp8_cutlass_scaled_mm

    native_fp8_support = (
        torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)
    )

    # Set layer filter
    if layer_filter is None:
        if include_patterns is not None or exclude_patterns is not None:

            def _layer_filter(name: str) -> bool:
                return should_quantize_layer(name, include_patterns, exclude_patterns)

            layer_filter = _layer_filter
        else:
            layer_filter = create_default_layer_filter()

    # Convert model to bfloat16
    model.to(torch.bfloat16)

    named_modules = list(model.named_modules())

    for name, linear in tqdm.tqdm(named_modules, desc="Quantizing weights"):
        if isinstance(linear, torch.nn.Linear):
            if layer_filter(name):
                print(f"Quantizing {name}")
                if quant_type == "fp8-per-tensor":
                    quant_weight, weight_scale = fp8_per_tensor_quant(linear.weight)
                elif quant_type == "fp8-per-token":
                    quant_weight, weight_scale = fp8_per_token_group_quant(
                        linear.weight, linear.weight.shape[-1]
                    )
                    weight_scale = weight_scale.t()
                elif quant_type == "fp8-per-block":
                    if native_fp8_support:
                        _ = (
                            _ensure_deep_gemm()
                        )  # checked import; error if not available
                    quant_weight, weight_scale = fp8_per_block_quant(linear.weight)
                else:
                    raise ValueError(f"Invalid quant_type: {quant_type}")

                bias = copy.deepcopy(linear.bias) if linear.bias is not None else None

                quant_linear = FP8DynamicLinear(
                    weight=quant_weight,
                    weight_scale=weight_scale,
                    bias=bias,
                    native_fp8_support=native_fp8_support,
                    quant_type=quant_type,
                )

                replace_module(model, name, quant_linear)
                del linear.weight
                del linear.bias
                del linear

    cleanup_memory()
