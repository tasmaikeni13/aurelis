from __future__ import annotations

import pytest
import torch

from aurelis import prepared_aurelis_head


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_eager_and_compiled_forward_backward_agree() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and torch.version.hip else "cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    matrix = torch.randn(1, 2, 4, 4, device=device, dtype=dtype)
    precision = matrix @ matrix.mT + 0.5 * torch.eye(4, device=device, dtype=dtype)
    inputs = (
        precision,
        torch.randn(1, 2, 3, 4, device=device, dtype=dtype),
        torch.randn(1, 2, 3, 4, device=device, dtype=dtype),
        torch.randn(1, 2, 3, 3, device=device, dtype=dtype),
        torch.rand(1, 2, 3, device=device, dtype=dtype) + 0.2,
        torch.randn(1, 2, 4, device=device, dtype=dtype),
        torch.rand(1, 2, device=device, dtype=dtype) + 0.2,
        torch.ones(1, 2, device=device, dtype=dtype),
    )
    eager_inputs = tuple(item.detach().clone().requires_grad_() for item in inputs)
    eager_tuple = prepared_aurelis_head(*eager_inputs)
    eager = eager_tuple[2]
    eager_grad = torch.autograd.grad(
        eager_tuple[2].sum() + eager_tuple[3].sum(), eager_inputs
    )
    compiled = torch.compile(
        prepared_aurelis_head, backend="inductor", fullgraph=True
    )
    compiled_inputs = tuple(item.detach().clone().requires_grad_() for item in inputs)
    try:
        actual_tuple = compiled(*compiled_inputs)
    except Exception as exc:
        pytest.fail(f"TorchInductor compile failed: {exc!r}")
    actual = actual_tuple[2]
    actual_grad = torch.autograd.grad(
        actual_tuple[2].sum() + actual_tuple[3].sum(), compiled_inputs
    )
    tolerance = 3e-5 if dtype == torch.float32 else 1e-9
    for actual_item, expected_item in zip(actual_tuple, eager_tuple, strict=True):
        torch.testing.assert_close(
            actual_item, expected_item, rtol=tolerance, atol=tolerance
        )
    for actual_item, expected_item in zip(actual_grad, eager_grad, strict=True):
        torch.testing.assert_close(
            actual_item, expected_item, rtol=tolerance, atol=tolerance
        )
