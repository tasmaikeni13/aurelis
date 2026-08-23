from __future__ import annotations

import torch

from csm import FP64GaussMarkovMemory

DTYPE = torch.float64


def test_autograd_reaches_all_continuous_inputs() -> None:
    steps, d_key, d_value = 4, 3, 2
    memory = FP64GaussMarkovMemory(d_key, d_value, epsilon=0.4)
    keys = torch.randn(steps, d_key, dtype=DTYPE, requires_grad=True)
    values = torch.randn(steps, d_value, dtype=DTYPE, requires_grad=True)
    beta = (0.5 + torch.rand(steps, dtype=DTYPE)).requires_grad_()
    decay = (0.7 + 0.2 * torch.rand(steps, dtype=DTYPE)).requires_grad_()
    query = torch.randn(d_key, dtype=DTYPE, requires_grad=True)

    state = memory.run(keys, values, beta, decay)
    read, variance = memory.read_with_confidence(state, query)
    loss = read.square().sum() + 0.2 * variance
    gradients = torch.autograd.grad(loss, (keys, values, beta, decay, query))

    for gradient, source in zip(gradients, (keys, values, beta, decay, query)):
        assert gradient.shape == source.shape
        assert torch.isfinite(gradient).all()


def test_gradcheck_of_recurrence_cholesky_read_and_confidence() -> None:
    steps, d_key, d_value = 3, 2, 2
    memory = FP64GaussMarkovMemory(d_key, d_value, epsilon=0.7)

    def function(keys, values, beta, decay, query):
        state = memory.run(keys, values, beta, decay)
        read, variance = memory.read_with_confidence(state, query)
        return torch.cat((read, variance.unsqueeze(0)))

    arguments = (
        torch.randn(steps, d_key, dtype=DTYPE, requires_grad=True),
        torch.randn(steps, d_value, dtype=DTYPE, requires_grad=True),
        (0.5 + torch.rand(steps, dtype=DTYPE)).requires_grad_(),
        (0.75 + 0.1 * torch.rand(steps, dtype=DTYPE)).requires_grad_(),
        torch.randn(d_key, dtype=DTYPE, requires_grad=True),
    )
    assert torch.autograd.gradcheck(
        function, arguments, eps=1e-6, atol=2e-5, rtol=2e-4, fast_mode=False
    )

