"""Mathematically transparent Gauss–Markov CSM memory.

The module implements Definition 5.1 without learned encoders, scan kernels, or
approximate inverses. State objects are immutable containers so recurrence
graphs remain visible to autograd and experiments cannot accidentally carry
state between trials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

SolveMethod = Literal["cholesky", "solve"]


@dataclass(frozen=True)
class CSMState:
    """The sufficient statistics ``S`` and ``C`` at one time step."""

    S: Tensor
    C: Tensor

    @property
    def key_dimension(self) -> int:
        return self.S.shape[-1]

    @property
    def value_dimension(self) -> int:
        return self.C.shape[-2]


def _outer(left: Tensor, right: Tensor) -> Tensor:
    return left.unsqueeze(-1) * right.unsqueeze(-2)


def _as_scalar(value: Tensor | float, reference: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _validate_state(state: CSMState, d_key: int, d_value: int) -> None:
    if state.S.shape != (d_key, d_key):
        raise ValueError(f"S must have shape {(d_key, d_key)}, got {state.S.shape}")
    if state.C.shape != (d_value, d_key):
        raise ValueError(f"C must have shape {(d_value, d_key)}, got {state.C.shape}")


def _solve(A: Tensor, q: Tensor, method: SolveMethod) -> Tensor:
    """Solve ``A x = q`` without ever constructing ``A^{-1}``."""

    if method == "cholesky":
        factor = torch.linalg.cholesky(A)
        return torch.cholesky_solve(q.unsqueeze(-1), factor).squeeze(-1)
    if method == "solve":
        return torch.linalg.solve(A, q)
    raise ValueError(f"unknown solve method: {method}")


class GaussMarkovMemory(nn.Module):
    """Reference recurrence and exact linear-system read.

    Parameters are structural only; there are no learned parameters. The
    default dtype is fp64 as required by Phase 1.
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        epsilon: float = 1e-3,
        *,
        dtype: torch.dtype = torch.float64,
        solve_method: SolveMethod = "cholesky",
    ) -> None:
        super().__init__()
        if d_key < 1 or d_value < 1:
            raise ValueError("key and value dimensions must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be strictly positive")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point")
        if solve_method not in ("cholesky", "solve"):
            raise ValueError("solve_method must be 'cholesky' or 'solve'")
        self.d_key = d_key
        self.d_value = d_value
        self.epsilon = float(epsilon)
        self.dtype = dtype
        self.solve_method = solve_method

    def initial_state(self, *, device: torch.device | str | None = None) -> CSMState:
        return CSMState(
            S=torch.zeros((self.d_key, self.d_key), dtype=self.dtype, device=device),
            C=torch.zeros((self.d_value, self.d_key), dtype=self.dtype, device=device),
        )

    def write(
        self,
        state: CSMState,
        key: Tensor,
        value: Tensor,
        beta: Tensor | float,
        decay: Tensor | float,
    ) -> CSMState:
        """Apply ``S'=λS+βkkᵀ`` and ``C'=λC+βvkᵀ``."""

        _validate_state(state, self.d_key, self.d_value)
        if key.shape != (self.d_key,):
            raise ValueError(f"key must have shape {(self.d_key,)}, got {key.shape}")
        if value.shape != (self.d_value,):
            raise ValueError(
                f"value must have shape {(self.d_value,)}, got {value.shape}"
            )
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise TypeError(f"keys and values must use {self.dtype}")
        beta_t = _as_scalar(beta, key)
        decay_t = _as_scalar(decay, key)
        if beta_t.ndim != 0 or decay_t.ndim != 0:
            raise ValueError("beta and decay must be scalars")
        return CSMState(
            S=decay_t * state.S + beta_t * _outer(key, key),
            C=decay_t * state.C + beta_t * _outer(value, key),
        )

    def run(
        self,
        keys: Tensor,
        values: Tensor,
        beta: Tensor,
        decay: Tensor,
        *,
        initial_state: CSMState | None = None,
    ) -> CSMState:
        """Evaluate the recurrence sequentially while preserving autograd."""

        if keys.ndim != 2 or keys.shape[1] != self.d_key:
            raise ValueError(f"keys must have shape [T, {self.d_key}]")
        steps = keys.shape[0]
        if values.shape != (steps, self.d_value):
            raise ValueError(f"values must have shape [{steps}, {self.d_value}]")
        if beta.shape != (steps,) or decay.shape != (steps,):
            raise ValueError("beta and decay must each have shape [T]")
        state = initial_state or self.initial_state(device=keys.device)
        for step in range(steps):
            state = self.write(
                state, keys[step], values[step], beta[step], decay[step]
            )
        return state

    def system_matrix(self, state: CSMState) -> Tensor:
        _validate_state(state, self.d_key, self.d_value)
        identity = torch.eye(self.d_key, dtype=state.S.dtype, device=state.S.device)
        return state.S + self.epsilon * identity

    def solve(self, state: CSMState, query: Tensor) -> Tensor:
        if query.shape != (self.d_key,):
            raise ValueError(f"query must have shape {(self.d_key,)}")
        return _solve(self.system_matrix(state), query, self.solve_method)

    def read(self, state: CSMState, query: Tensor) -> Tensor:
        """Return ``C (S + εI)^{-1} q`` through a stable solve."""

        return state.C @ self.solve(state, query)

    def confidence(self, state: CSMState, query: Tensor) -> Tensor:
        """Return manuscript ``c(q)`` (epistemic variance; lower is surer)."""

        solved = self.solve(state, query)
        return torch.dot(query, solved)

    def read_with_confidence(
        self, state: CSMState, query: Tensor
    ) -> tuple[Tensor, Tensor]:
        solved = self.solve(state, query)
        return state.C @ solved, torch.dot(query, solved)


class FP64GaussMarkovMemory(GaussMarkovMemory):
    """Explicit fp64 specialization used as the Phase 1 reference."""

    def __init__(
        self,
        d_key: int,
        d_value: int,
        epsilon: float = 1e-3,
        *,
        solve_method: SolveMethod = "cholesky",
    ) -> None:
        super().__init__(
            d_key,
            d_value,
            epsilon,
            dtype=torch.float64,
            solve_method=solve_method,
        )


def recompute_state(
    keys: Tensor,
    values: Tensor,
    beta: Tensor,
    decay: Tensor,
) -> CSMState:
    """Naively recompute the final sufficient statistics from full history.

    A write at index ``s`` has final weight
    ``beta[s] * product(decay[s+1:])``. This implementation constructs those
    weighted historical terms independently of the sequential recurrence.
    """

    if keys.ndim != 2 or values.ndim != 2:
        raise ValueError("keys and values must be rank-2 tensors")
    steps, d_key = keys.shape
    if values.shape[0] != steps:
        raise ValueError("keys and values must have the same sequence length")
    if beta.shape != (steps,) or decay.shape != (steps,):
        raise ValueError("beta and decay must each have shape [T]")
    if steps == 0:
        return CSMState(
            S=keys.new_zeros((d_key, d_key)),
            C=values.new_zeros((values.shape[1], d_key)),
        )

    weighted_S: list[Tensor] = []
    weighted_C: list[Tensor] = []
    for write_index in range(steps):
        suffix = decay[write_index + 1 :]
        suffix_decay = suffix.prod() if suffix.numel() else decay.new_ones(())
        weight = beta[write_index] * suffix_decay
        weighted_S.append(weight * _outer(keys[write_index], keys[write_index]))
        weighted_C.append(weight * _outer(values[write_index], keys[write_index]))
    return CSMState(
        S=torch.stack(weighted_S).sum(0), C=torch.stack(weighted_C).sum(0)
    )


def direct_inverse_oracle(
    state: CSMState,
    query: Tensor,
    epsilon: float,
    *,
    max_dimension: int = 32,
) -> tuple[Tensor, Tensor]:
    """Tiny-matrix inverse oracle, forbidden from production-style paths."""

    d_key = state.key_dimension
    if d_key > max_dimension:
        raise ValueError(
            f"direct inverse oracle is limited to d_key <= {max_dimension}, got {d_key}"
        )
    identity = torch.eye(d_key, dtype=state.S.dtype, device=state.S.device)
    inverse = torch.linalg.inv(state.S + epsilon * identity)
    solved = inverse @ query
    return state.C @ solved, torch.dot(query, solved)

