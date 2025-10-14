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

from typing import Optional

import torch
import triton
import triton.language as tl

from ..quantizers import *
from ..utils import _ensure_deep_gemm

# modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py
fp8_gemm_configs = [
    triton.Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=fp8_gemm_configs, key=["N", "K"])
@triton.jit
def _fp8_gemm_triton_block_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


# modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py
def fp8_gemm_triton_block(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    out_dtype=torch.bfloat16,
    bias=None,
) -> torch.Tensor:
    """
    Perform a matrix multiplication using FP8 precision.
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=out_dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _fp8_gemm_triton_block_kernel[grid](a, b, c, a_s, b_s, M, N, K)

    if bias is not None:
        c += bias

    return c


def fp8_gemm_deepgemm_block(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    out_dtype=torch.bfloat16,
    bias=None,
    origin_shape=None,
) -> torch.Tensor:
    a_fp8 = (a, a_s)
    b_fp8 = (b, b_s)
    out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.bfloat16)

    _deep_gemm = _ensure_deep_gemm()
    _deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, out)

    if origin_shape is not None:
        out = out.reshape([*origin_shape[:-1], b.shape[0]])
    if bias is not None:
        out += bias

    return out.to(out_dtype)


def fp8_gemm_torch_tensor_token(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    out_dtype=torch.bfloat16,
    bias=None,
) -> torch.Tensor:
    need_reshape = a.dim() == 3
    if need_reshape:
        batch_size = a.shape[0]
        A_input = a.reshape(-1, a.shape[-1])
    else:
        batch_size = None
        A_input = a

    output = torch._scaled_mm(
        A_input,
        b.t(),
        out_dtype=out_dtype,
        scale_a=a_s,
        scale_b=b_s,
        bias=bias,
    )
    # If output is a tuple, take the first element
    if isinstance(output, tuple):
        output = output[0]

    if need_reshape:
        output = output.reshape(
            batch_size, output.shape[0] // batch_size, output.shape[1]
        )

    return output


def fp8_gemm(
    A,
    A_scale,
    B,
    B_scale,
    bias,
    out_dtype,
    native_fp8_support=False,
    quant_type="fp8-per-tensor",
    origin_shape=None,
):
    if A.numel() == 0:
        return torch.empty(size=(0, B.shape[0]), dtype=out_dtype, device=A.device)

    if native_fp8_support and quant_type == "fp8-per-tensor":
        output = fp8_gemm_torch_tensor_token(A, A_scale, B, B_scale, out_dtype, bias)
    elif native_fp8_support and quant_type == "fp8-per-token":
        output = fp8_gemm_torch_tensor_token(A, A_scale, B, B_scale, out_dtype, bias)
    elif native_fp8_support and quant_type == "fp8-per-block":
        output = fp8_gemm_deepgemm_block(
            A, A_scale, B, B_scale, out_dtype, bias, origin_shape
        )
    elif not native_fp8_support and quant_type == "fp8-per-block":
        output = fp8_gemm_triton_block(A, A_scale, B, B_scale, out_dtype, bias)
    else:
        output = torch.nn.functional.linear(
            A.to(out_dtype) * A_scale,
            B.to(out_dtype) * B_scale.to(out_dtype),
            bias=bias,
        )

    return output


# modified from https://github.com/neuralmagic/AutoFP8/blob/main/auto_fp8/quantize.py
class FP8DynamicLinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.nn.Parameter,
        native_fp8_support: bool = False,
        quant_type: str = "fp8-per-tensor",
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias
        self.native_fp8_support = native_fp8_support
        self.quant_type = quant_type

    @torch.compiler.disable(recursive=True)
    def forward(self, x):
        ori_dtype = x.dtype
        assert ori_dtype in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ], "x.dtype must be float32, bfloat16, or float16"

        if ori_dtype == torch.float32:
            x = x.to(torch.bfloat16)

        if self.quant_type == "fp8-per-tensor":
            origin_shape = None
            qinput, x_scale = fp8_per_tensor_quant(x)
        elif self.quant_type == "fp8-per-token":
            origin_shape = None
            x_2d = x.view(-1, x.shape[-1])
            qinput, x_scale = fp8_per_token_group_quant(x_2d, x_2d.shape[-1])
        elif self.quant_type == "fp8-per-block" and self.native_fp8_support:
            origin_shape = x.shape
            x = x.view(-1, x.shape[-1])
            qinput, x_scale = fp8_per_token_group_quant(
                x, group_size=128, column_major_scales=True, scale_tma_aligned=True
            )
        elif self.quant_type == "fp8-per-block" and not self.native_fp8_support:
            origin_shape = None
            qinput, x_scale = fp8_per_block_quant(x, block_size=128)
        else:
            raise ValueError(f"Invalid quant_type: {self.quant_type}")

        output = fp8_gemm(
            A=qinput,
            A_scale=x_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
            native_fp8_support=self.native_fp8_support,
            quant_type=self.quant_type,
            origin_shape=origin_shape,
        )

        if self.quant_type == "fp8-per-token" and x.dim() == 3 and output.dim() == 2:
            output = output.unsqueeze(0)

        return output


if __name__ == "__main__":
    weight = torch.randn(1024, 1024).to(torch.float8_e4m3fn).cuda()
    weight_scale = torch.randn(1024).float().cuda()
    bias = torch.randn(1024).cuda()
    linear = FP8DynamicLinear(
        weight,
        weight_scale,
        bias,
        native_fp8_support=False,
        quant_type="fp8-per-tensor",
    )
    x = torch.randn(1024, 1024).to(torch.bfloat16).cuda()
    output = linear(x)
    print(output.shape)
