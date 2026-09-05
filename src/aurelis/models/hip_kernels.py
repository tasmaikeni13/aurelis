"""Accelerated HIP/ROCm kernels and fused operators for AMD Instinct MI300X."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Ensure ROCm extension builds target the local AMD Instinct MI300X architecture only
if "PYTORCH_ROCM_ARCH" not in os.environ:
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"

_HIP_MODULE = None
_HIP_ATTEMPTED = False


def _build_hip_kernels():
    global _HIP_MODULE, _HIP_ATTEMPTED
    if _HIP_ATTEMPTED:
        return _HIP_MODULE
    _HIP_ATTEMPTED = True

    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        return None

    try:
        from torch.utils.cpp_extension import load_inline

        hip_sources = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Accelerated recurrent selective scan: h_t = decay_t * h_{t-1} + x_t
__global__ void recurrent_scan_f32_kernel(
    const float* __restrict__ x,       // [B, H, L, D]
    const float* __restrict__ decay,   // [B, H, L, D]
    float* __restrict__ out,           // [B, H, L, D]
    int B, int H, int L, int D
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || h >= H || d >= D) return;

    int bh_offset = (b * H + h) * L * D;
    float state = 0.0f;

    for (int t = 0; t < L; ++t) {
        int idx = bh_offset + t * D + d;
        float a = decay[idx];
        float input_val = x[idx];
        state = a * state + input_val;
        out[idx] = state;
    }
}

// Fused uncertainty residual gate: y = remote + g * (vbar - mapped_kbar)
__global__ void fused_residual_gate_f32_kernel(
    const float* __restrict__ remote,       // [B, H, L, D]
    const float* __restrict__ vbar,         // [B, H, L, D]
    const float* __restrict__ mapped_kbar,  // [B, H, L, D]
    const float* __restrict__ gate,         // [B, H, L]
    float* __restrict__ out,                // [B, H, L, D]
    int B, int H, int L, int D
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int l_d = blockIdx.x * blockDim.x + threadIdx.x;
    int total_ld = L * D;
    if (b >= B || h >= H || l_d >= total_ld) return;

    int l = l_d / D;
    int d = l_d % D;

    int elem_idx = (b * H + h) * total_ld + l_d;
    int gate_idx = (b * H + h) * L + l;

    float g = gate[gate_idx];
    float r = remote[elem_idx];
    float v = vbar[elem_idx];
    float mk = mapped_kbar[elem_idx];

    out[elem_idx] = r + g * (v - mk);
}

void launch_recurrent_scan_f32(at::Tensor x, at::Tensor decay, at::Tensor out) {
    int B = x.size(0);
    int H = x.size(1);
    int L = x.size(2);
    int D = x.size(3);

    int threads = 256;
    int blocks_x = (D + threads - 1) / threads;
    dim3 grid(blocks_x, H, B);
    dim3 block(threads);

    hipLaunchKernelGGL(
        recurrent_scan_f32_kernel, grid, block, 0, 0,
        x.data_ptr<float>(), decay.data_ptr<float>(), out.data_ptr<float>(),
        B, H, L, D
    );
}

void launch_fused_residual_gate_f32(
    at::Tensor remote, at::Tensor vbar, at::Tensor mapped_kbar, at::Tensor gate, at::Tensor out
) {
    int B = remote.size(0);
    int H = remote.size(1);
    int L = remote.size(2);
    int D = remote.size(3);

    int total_ld = L * D;
    int threads = 256;
    int blocks_x = (total_ld + threads - 1) / threads;
    dim3 grid(blocks_x, H, B);
    dim3 block(threads);

    hipLaunchKernelGGL(
        fused_residual_gate_f32_kernel, grid, block, 0, 0,
        remote.data_ptr<float>(), vbar.data_ptr<float>(), mapped_kbar.data_ptr<float>(),
        gate.data_ptr<float>(), out.data_ptr<float>(),
        B, H, L, D
    );
}
"""
        cpp_sources = """
void launch_recurrent_scan_f32(at::Tensor x, at::Tensor decay, at::Tensor out);
void launch_fused_residual_gate_f32(
    at::Tensor remote, at::Tensor vbar, at::Tensor mapped_kbar, at::Tensor gate, at::Tensor out
);
"""
        _HIP_MODULE = load_inline(
            name="aurelis_hip_kernels",
            cpp_sources=cpp_sources,
            cuda_sources=hip_sources,
            functions=["launch_recurrent_scan_f32", "launch_fused_residual_gate_f32"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx942"],
        )
        return _HIP_MODULE
    except Exception as err:
        logger.warning("Could not build inline HIP kernels: %s. Using PyTorch reference paths.", err)
        return None


def hip_recurrent_scan(x: Tensor, decay: Tensor) -> Tensor:
    """Fast recurrent scan along the sequence dimension: h_t = decay_t * h_{t-1} + x_t."""
    # Input shapes: [B, H, L, D]
    if x.is_cuda and getattr(torch.version, "hip", None) and x.dtype == torch.float32:
        mod = _build_hip_kernels()
        if mod is not None:
            out = torch.empty_like(x)
            mod.launch_recurrent_scan_f32(x.contiguous(), decay.contiguous(), out)
            return out

    # PyTorch reference path
    B, H, L, D = x.shape
    out = torch.empty_like(x)
    curr = torch.zeros(B, H, D, dtype=x.dtype, device=x.device)
    for t in range(L):
        curr = decay[:, :, t, :] * curr + x[:, :, t, :]
        out[:, :, t, :] = curr
    return out


def hip_fused_residual_gate(
    remote: Tensor,
    vbar: Tensor,
    mapped_kbar: Tensor,
    gate: Tensor,
) -> Tensor:
    """Fused evaluation of y = remote + gate * (vbar - mapped_kbar)."""
    # Shapes: remote, vbar, mapped_kbar: [B, H, L, D]
    B, H, L, D = remote.shape
    gate_3d = gate.view(B, H, L)

    if (
        remote.is_cuda
        and getattr(torch.version, "hip", None)
        and remote.dtype == torch.float32
        and gate_3d.dtype == torch.float32
    ):
        mod = _build_hip_kernels()
        if mod is not None:
            out = torch.empty_like(remote)
            mod.launch_fused_residual_gate_f32(
                remote.contiguous(),
                vbar.contiguous(),
                mapped_kbar.contiguous(),
                gate_3d.contiguous(),
                out,
            )
            return out

    # PyTorch fallback: broadcast [B, H, L, 1]
    g = gate_3d.unsqueeze(-1)
    return remote + g * (vbar - mapped_kbar)

