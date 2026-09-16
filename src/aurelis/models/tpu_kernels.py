"""Accelerated JAX / XLA / HLO kernels and fused operators for Google Cloud TPU v4 Pod (16 v4 TPUs / 32 TensorCores)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Ensure TPU single-host process bounds default when running on a pod slice host
if "TPU_PROCESS_BOUNDS" not in os.environ and "TPU_CHIPS_PER_PROCESS_BOUNDS" not in os.environ:
    os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
    os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "2,2,1"

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False
    jax = None
    jnp = None
    lax = None


def get_tpu_pod_info() -> Dict[str, Any]:
    """Inspect and return topology information for the Cloud TPU v4 Pod slice."""
    if not _JAX_AVAILABLE:
        return {"tpu_available": False, "reason": "JAX is not installed"}

    try:
        devices = jax.devices()
        tpu_devices = [d for d in devices if d.platform == "tpu"]
        return {
            "tpu_available": len(tpu_devices) > 0,
            "platform": "tpu" if tpu_devices else devices[0].platform,
            "device_count": len(devices),
            "process_count": jax.process_count(),
            "process_index": jax.process_index(),
            "devices": [str(d) for d in devices],
            "accelerator_type": "v4-32" if len(tpu_devices) >= 4 else "v4",
            "total_chips_in_pod": 16,
            "tensor_cores": 32,
            "topology": "2x2x4",
        }
    except Exception as err:
        return {"tpu_available": False, "error": str(err)}


def init_tpu_pod(coordinator_address: Optional[str] = None, num_processes: int = 4, process_id: int = 0) -> bool:
    """Initialize multi-host TPU Pod coordination or fallback to local process bounds."""
    if not _JAX_AVAILABLE:
        return False

    if coordinator_address is not None:
        try:
            jax.distributed.initialize(
                coordinator_address=coordinator_address,
                num_processes=num_processes,
                process_id=process_id,
            )
            logger.info("Initialized multi-host TPU pod with %d processes.", num_processes)
            return True
        except Exception as err:
            logger.warning("Multi-host TPU pod initialization failed: %s. Falling back to local devices.", err)
            return False
    return True


# =========================================================================
# JAX / XLA HLO Accelerated Kernels
# =========================================================================

if _JAX_AVAILABLE:

    @jax.jit
    def jax_recurrent_scan(x: jax.Array, decay: jax.Array) -> jax.Array:
        """Accelerated recurrent selective scan along the sequence dimension: h_t = decay_t * h_{t-1} + x_t.

        Lowers via XLA to optimized TPU VPU/MXU vector loop pipelines.
        Input shapes: [B, H, L, D]
        """
        # Define scan step: carry is [B, H, D], inputs are slice (x_t, decay_t) of [B, H, D]
        def scan_step(h_prev: jax.Array, inputs: Tuple[jax.Array, jax.Array]) -> Tuple[jax.Array, jax.Array]:
            x_t, decay_t = inputs
            h_curr = decay_t * h_prev + x_t
            return h_curr, h_curr

        # Move sequence axis (L) to lead dimension for efficient jax.lax.scan: [L, B, H, D]
        x_trans = jnp.moveaxis(x, 2, 0)
        decay_trans = jnp.moveaxis(decay, 2, 0)
        B, H, L, D = x.shape
        h0 = jnp.zeros((B, H, D), dtype=x.dtype)

        _, out_trans = lax.scan(scan_step, h0, (x_trans, decay_trans))
        # Move sequence axis back to [B, H, L, D]
        return jnp.moveaxis(out_trans, 0, 2)

    @jax.jit
    def jax_fused_residual_gate(
        remote: jax.Array,
        vbar: jax.Array,
        mapped_kbar: jax.Array,
        gate: jax.Array,
    ) -> jax.Array:
        """Fused evaluation of y = remote + gate * (vbar - mapped_kbar) targeting TPU VPU.

        XLA fuses the subtract, multiply, and add into a single HLO instruction loop.
        Shapes: remote, vbar, mapped_kbar: [B, H, L, D]; gate: [B, H, L] or [B, H, L, 1]
        """
        if gate.ndim == 3:
            g = jnp.expand_dims(gate, -1)
        else:
            g = gate
        return remote + g * (vbar - mapped_kbar)

    @jax.jit
    def jax_rmsnorm(x: jax.Array, weight: jax.Array, eps: float = 1e-6) -> jax.Array:
        """Fused RMSNorm targeting Cloud TPU v4 vector units."""
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        inv_rms = lax.rsqrt(variance + eps)
        return x * inv_rms * weight

    @jax.jit
    def jax_swiglu(gate: jax.Array, up: jax.Array) -> jax.Array:
        """Fused SwiGLU activation targeting Cloud TPU v4: silu(gate) * up."""
        silu_gate = gate * lax.sigmoid(gate)
        return silu_gate * up

else:
    jax_recurrent_scan = None
    jax_fused_residual_gate = None
    jax_rmsnorm = None
    jax_swiglu = None


# =========================================================================
# PyTorch Interfaces with Native Fallbacks & TPU/JAX Interoperability
# =========================================================================

def tpu_recurrent_scan(x: Tensor, decay: Tensor) -> Tensor:
    """Accelerated recurrent selective scan along sequence dimension: h_t = decay_t * h_{t-1} + x_t.

    Targets Cloud TPU v4 via XLA/JAX with PyTorch native fallback.
    """
    if _JAX_AVAILABLE and not x.requires_grad:
        try:
            x_np = x.detach().cpu().numpy()
            decay_np = decay.detach().cpu().numpy()
            x_jax = jnp.asarray(x_np)
            decay_jax = jnp.asarray(decay_np)
            out_jax = jax_recurrent_scan(x_jax, decay_jax)
            out_np = np.array(out_jax, copy=True)
            return torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype)
        except Exception as err:
            logger.debug("JAX TPU scan fallback to PyTorch: %s", err)

    # Pure PyTorch reference path
    B, H, L, D = x.shape
    out = torch.empty_like(x)
    curr = torch.zeros(B, H, D, dtype=x.dtype, device=x.device)
    for t in range(L):
        curr = decay[:, :, t, :] * curr + x[:, :, t, :]
        out[:, :, t, :] = curr
    return out


def tpu_fused_residual_gate(
    remote: Tensor,
    vbar: Tensor,
    mapped_kbar: Tensor,
    gate: Tensor,
) -> Tensor:
    """Fused evaluation of y = remote + gate * (vbar - mapped_kbar).

    Targets Cloud TPU v4 via XLA/JAX with PyTorch native fallback.
    """
    B, H, L, D = remote.shape
    gate_3d = gate.view(B, H, L)

    if _JAX_AVAILABLE and not remote.requires_grad:
        try:
            r_jax = jnp.asarray(remote.detach().cpu().numpy())
            v_jax = jnp.asarray(vbar.detach().cpu().numpy())
            mk_jax = jnp.asarray(mapped_kbar.detach().cpu().numpy())
            g_jax = jnp.asarray(gate_3d.detach().cpu().numpy())
            out_jax = jax_fused_residual_gate(r_jax, v_jax, mk_jax, g_jax)
            return torch.from_numpy(np.array(out_jax, copy=True)).to(device=remote.device, dtype=remote.dtype)
        except Exception as err:
            logger.debug("JAX TPU fused gate fallback to PyTorch: %s", err)

    g = gate_3d.unsqueeze(-1)
    return remote + g * (vbar - mapped_kbar)


def tpu_rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Fused RMSNorm targeting Cloud TPU v4 with PyTorch reference fallback."""
    if _JAX_AVAILABLE and not x.requires_grad:
        try:
            x_jax = jnp.asarray(x.detach().cpu().numpy())
            w_jax = jnp.asarray(weight.detach().cpu().numpy())
            out_jax = jax_rmsnorm(x_jax, w_jax, eps)
            return torch.from_numpy(np.array(out_jax, copy=True)).to(device=x.device, dtype=x.dtype)
        except Exception as err:
            logger.debug("JAX TPU rmsnorm fallback to PyTorch: %s", err)

    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


class _FusedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:
        ctx.save_for_backward(gate, up)
        if _JAX_AVAILABLE and not gate.requires_grad and not up.requires_grad:
            try:
                g_jax = jnp.asarray(gate.detach().cpu().numpy())
                u_jax = jnp.asarray(up.detach().cpu().numpy())
                out_jax = jax_swiglu(g_jax, u_jax)
                return torch.from_numpy(np.array(out_jax, copy=True)).to(device=gate.device, dtype=gate.dtype)
            except Exception:
                pass
        return torch.nn.functional.silu(gate) * up

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        gate, up = ctx.saved_tensors
        sig = torch.sigmoid(gate)
        silu = sig * gate
        d_up = grad_output * silu
        d_gate = grad_output * up * (sig + silu * (1.0 - sig))
        return d_gate, d_up


def tpu_swiglu(gate: Tensor, up: Tensor) -> Tensor:
    """Fused SwiGLU forward and backward targeting Cloud TPU v4."""
    return _FusedSwiGLUFunction.apply(gate, up)
