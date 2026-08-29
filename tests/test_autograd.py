from __future__ import annotations

import torch

from aurelis import AurelisProjectionBlock, vectorized_reference


def test_gradcheck_all_declared_differentiable_inputs() -> None:
    batch, heads, length, d_key, d_value = 1, 1, 3, 2, 2
    keys = torch.randn(batch, heads, length, d_key, requires_grad=True)
    values = torch.randn(batch, heads, length, d_value, requires_grad=True)
    evidence_raw = torch.randn(batch, heads, length, requires_grad=True)
    queries = torch.randn(batch, heads, length, d_key, requires_grad=True)
    temperature_raw = torch.randn(1, heads, 1, requires_grad=True)
    episodic_raw = torch.randn(batch, heads, length, requires_grad=True)

    def function(
        k: torch.Tensor,
        q: torch.Tensor,
        v: torch.Tensor,
        beta_raw: torch.Tensor,
        tau_raw: torch.Tensor,
        e_raw: torch.Tensor,
    ) -> torch.Tensor:
        result = vectorized_reference(
            k,
            v,
            torch.nn.functional.softplus(beta_raw) + 0.2,
            q,
            window=2,
            prior=0.7,
            temperature=torch.nn.functional.softplus(tau_raw) + 0.1,
            episodic_responsibility=torch.sigmoid(e_raw),
        )
        return result.episodic

    assert torch.autograd.gradcheck(
        function,
        (keys, queries, values, evidence_raw, temperature_raw, episodic_raw),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
        fast_mode=False,
    )


def test_projection_parameters_receive_finite_gradients() -> None:
    model = AurelisProjectionBlock(6, 2, 3, 2, 3, prior=0.5).double()
    hidden = torch.randn(2, 5, 6, requires_grad=True)
    output, sequence = model(hidden)
    loss = output.square().mean() + sequence.g_B.mean()
    loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

