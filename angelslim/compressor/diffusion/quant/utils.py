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


import gc
import re
from functools import lru_cache
from typing import List, Optional, Union

import torch

__all__ = [
    "replace_module",
    "cleanup_memory",
    "should_quantize_layer",
    "create_default_layer_filter",
    "_compile_pattern",
    "_ensure_deep_gemm",
]


def replace_module(model: torch.nn.Module, name: str, new_module: torch.nn.Module):
    """
    Replace a submodule in the model with a new module by name.
    """
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_name = name[len(parent_name) + 1 :]
        parent = model.get_submodule(parent_name)
    else:
        parent_name = ""
        parent = model
        child_name = name

    setattr(parent, child_name, new_module)


def cleanup_memory():
    """
    Run garbage collection and clear CUDA memory cache.
    """
    gc.collect()
    torch.cuda.empty_cache()


def _compile_pattern(
    pattern: Union[str, re.Pattern], case_sensitive: bool = False
) -> re.Pattern:
    """
    Compile a pattern (string or pre-compiled pattern) into a regex pattern object.

    Args:
        pattern: String pattern or already-compiled regex pattern.
        case_sensitive: Whether the match is case sensitive.

    Returns:
        Compiled regex pattern object.
    """
    if isinstance(pattern, str):
        # If the string contains special regex characters, treat as regex.
        if any(char in pattern for char in ".*+?^${}[]|()\\"):
            flags = 0 if case_sensitive else re.IGNORECASE
            return re.compile(pattern, flags)
        else:
            # Escape regular string so it matches literally.
            escaped = re.escape(pattern)
            flags = 0 if case_sensitive else re.IGNORECASE
            return re.compile(escaped, flags)
    else:
        # Already a compiled pattern.
        return pattern


def should_quantize_layer(
    layer_name: str,
    include_patterns: Optional[List[Union[str, re.Pattern]]] = None,
    exclude_patterns: Optional[List[Union[str, re.Pattern]]] = None,
    case_sensitive: bool = False,
) -> bool:
    """
    Decide whether a layer should be quantized based on inclusion/exclusion patterns.

    Args:
        layer_name: Name of the layer.
        include_patterns: List of patterns (str or regex) to include.
        exclude_patterns: List of patterns (str or regex) to exclude.
        case_sensitive: Whether patterns are matched case sensitively.

    Returns:
        bool: Whether this layer should be quantized.

    Note:
        String patterns are automatically detected as regex if they include special chars:
        - If contains any . * + ? ^ $ { } [ ] | ( ) \\ it's treated as regex,
        - Otherwise, it's escaped and matched literally.
    """
    if include_patterns is None:
        include_patterns = []
    if exclude_patterns is None:
        exclude_patterns = []

    # Check exclusion patterns
    for pattern in exclude_patterns:
        compiled_pattern = _compile_pattern(pattern, case_sensitive)
        if compiled_pattern.search(layer_name):
            return False

    # If no include patterns, default is to include all layers
    if not include_patterns:
        return True

    # Check inclusion patterns
    for pattern in include_patterns:
        compiled_pattern = _compile_pattern(pattern, case_sensitive)
        if compiled_pattern.search(layer_name):
            return True

    return False


def create_default_layer_filter():
    """
    Create a default layer name filter for quantization.
    Returns a preconfigured filter function.
    """
    include_patterns = ["wrapped_module", "block", "lin", "img", "txt"]
    exclude_patterns = ["embed"]

    return lambda name: should_quantize_layer(name, include_patterns, exclude_patterns)


_deep_gemm_cached = None


def _ensure_deep_gemm():
    """
    Lazy, safe import of deep_gemm with process-level caching. Returns the module
    if available, otherwise raises a clear error.
    """
    global _deep_gemm_cached
    if _deep_gemm_cached is not None:
        return _deep_gemm_cached
    try:
        import deep_gemm

        _deep_gemm_cached = deep_gemm
        return _deep_gemm_cached
    except ImportError as e:
        raise ImportError(
            "deep_gemm is required for 'fp8-per-block' quantization with native_fp8_support, "
            "but was not found. Please install deep_gemm first."
        ) from e


if __name__ == "__main__":
    """Test layer name filtering functionality: All English comments and outputs."""
    print("=== Test: Layer Filtering Functionality ===\n")

    # Test Case 1: Basic string match
    print("1. Test case: Basic string matching:")
    test_cases = [
        ("transformer.block.0.attention.linear_q", True),
        ("transformer.block.0.ffn.linear_1", True),
        ("transformer.embedding.word_embed", False),
        ("transformer.norm.final_norm", False),
    ]

    for layer_name, expected in test_cases:
        result = should_quantize_layer(
            layer_name,
            include_patterns=["linear", "attention"],
            exclude_patterns=["embed", "norm"],
        )
        status = "✓" if result == expected else "✗"
        print(f"  {status} {layer_name}: {result} (Expected: {expected})")

    print("\n2. Test: Regex pattern auto detection:")
    # Test regex auto-detection
    regex_cases = [
        ("model.linear1.weight", True),  # matches .*\.linear\d+
        ("model.linear2.weight", True),  # matches .*\.linear\d+
        ("model.linear.weight", False),  # doesn't match \d+
        ("model.attn.linear.weight", True),  # matches .*\.attn.*
        ("model.attention.linear.weight", False),  # doesn't match .attn.
        ("model.embed.word_embed.weight", False),  # excluded
    ]

    for layer_name, expected in regex_cases:
        result = should_quantize_layer(
            layer_name,
            include_patterns=[r".*\.linear\d+", r".*\.attn.*"],
            exclude_patterns=["embed"],
        )
        status = "✓" if result == expected else "✗"
        print(f"  {status} {layer_name}: {result} (Expected: {expected})")

    print("\n3. Test: Mixed string and regex patterns:")
    mixed_cases = [
        ("model.linear.weight", True),  # matches string "linear"
        ("model.linear1.weight", True),  # matches regex ".*\.linear\d+"
        ("model.attn.linear.weight", True),  # matches regex ".*\.attn.*"
        ("model.norm.weight", False),  # excluded
        ("model.embed.weight", False),  # excluded
    ]

    for layer_name, expected in mixed_cases:
        result = should_quantize_layer(
            layer_name,
            include_patterns=["linear", r".*\.attn.*"],  # mixed
            exclude_patterns=["norm", r".*embed.*"],
        )
        status = "✓" if result == expected else "✗"
        print(f"  {status} {layer_name}: {result} (Expected: {expected})")

    print("\n4. Test: Default filter:")
    default_filter = create_default_layer_filter()
    default_cases = [
        ("model.wrapped_module.linear", True),
        ("model.block.attention", True),
        ("model.lin.projection", True),
        ("model.img.encoder", True),
        ("model.txt.decoder", True),
        ("model.embedding.word_embed", False),
    ]

    for layer_name, expected in default_cases:
        result = default_filter(layer_name)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {layer_name}: {result} (Expected: {expected})")

    print("\n=== All tests finished! ===")
