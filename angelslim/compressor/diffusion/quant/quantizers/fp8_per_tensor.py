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

from typing import Tuple

import torch

FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)


# quant function for per-tensor fp8
# modified from https://github.com/neuralmagic/AutoFP8/blob/main/auto_fp8/quantize.py
def fp8_per_tensor_quant(x: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Dynamically Quantize a tensor using per-tensor static scaling factor.
    Args:
        x: The input tensor.
    """
    if x.numel() == 0:
        min_val, max_val = (
            torch.tensor(-16.0, dtype=x.dtype),
            torch.tensor(16.0, dtype=x.dtype),
        )
    else:
        min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = FP8_MAX / amax.clamp(min=1e-12)
    qx = (x * scale).clamp(min=FP8_MIN, max=FP8_MAX)
    qx = qx.to(torch.float8_e4m3fn)
    scale = scale.float().reciprocal()
    return qx, scale


if __name__ == "__main__":
    # Test both implementations
    x = torch.randn(1024, 1024).cuda()

    # Test PyTorch implementation
    qx_torch, scale_torch = fp8_per_tensor_quant(x)
    print("PyTorch implementation:")
    print(f"Quantized tensor shape: {qx_torch.shape}, dtype: {qx_torch.dtype}")
    print(f"Scale: {scale_torch}")
