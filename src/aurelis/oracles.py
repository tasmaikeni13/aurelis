"""Independent mathematics reference oracles (fp64) for AURELIS equations (2)-(13).

Contains two completely independent implementations:
1. ScalarOracle: Pure Python scalar arithmetic, nested lists, explicit loops.
   No reliance on PyTorch, BLAS, or tensor libraries.
2. TensorOracle: PyTorch float64 tensor operations, linear algebra, and einsum.

No shared math helpers to avoid circular agreement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence
import torch
from torch import Tensor


# ==============================================================================
# 1. SCALAR REFERENCE ORACLE (Pure Python float / lists / loops)
# ==============================================================================


@dataclass(frozen=True)
class ScalarCertificateResult:
    bound: float
    selected_mass: float
    estimated_unread_mass: float
    unread_mass_lower: float
    unread_mass_upper: float
    residual_bound: float
    denominator_floor: float


@dataclass(frozen=True)
class ScalarPageEnvelope:
    count: int
    key_min: list[float]
    key_max: list[float]
    center: list[float]
    radius: float
    ell: float
    u: float
    L: float
    U: float
    b: float


class ScalarOracle:
    """Independent scalar CPU oracle using primitive floats and loops."""

    @staticmethod
    def norm2(vec: Sequence[float]) -> float:
        """Euclidean norm of a 1D sequence."""
        return math.sqrt(sum(x * x for x in vec))

    @staticmethod
    def normalize_key(key: Sequence[float]) -> list[float]:
        """Normalize key vector to norm <= 1 with zero norm floor (Eq. 2)."""
        n2 = math.sqrt(sum(x * x for x in key))
        if n2 <= 1.0:
            return [float(x) for x in key]
        return [float(x) / n2 for x in key]

    @staticmethod
    def zero_matrix(d_v: int, d_k: int) -> list[list[float]]:
        """Return zero matrix S in R^{d_v x d_k}."""
        return [[0.0 for _ in range(d_k)] for _ in range(d_v)]

    @classmethod
    def gated_delta_update(
        cls,
        S: Sequence[Sequence[float]],
        key: Sequence[float],
        value: Sequence[float],
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> list[list[float]]:
        """Evaluate solve-free gated delta recurrent update (Eq. 2).

        S_tilde = alpha * S
        S^+ = S_tilde + beta * (v - S_tilde @ k) k^T
        """
        d_v = len(S)
        d_k = len(S[0])
        k = cls.normalize_key(key)
        v = [float(x) for x in value]
        a = float(alpha)
        b = float(beta)

        # S_tilde = alpha * S
        S_tilde = [[a * S[i][j] for j in range(d_k)] for i in range(d_v)]

        # pred = S_tilde @ k  (dimension d_v)
        pred = [0.0 for _ in range(d_v)]
        for i in range(d_v):
            pred[i] = sum(S_tilde[i][j] * k[j] for j in range(d_k))

        # error = v - pred
        error = [v[i] - pred[i] for i in range(d_v)]

        # S^+ = S_tilde + beta * error @ k^T
        S_next = [[0.0 for _ in range(d_k)] for _ in range(d_v)]
        for i in range(d_v):
            for j in range(d_k):
                S_next[i][j] = S_tilde[i][j] + b * error[i] * k[j]

        return S_next

    @classmethod
    def local_attention(
        cls,
        keys: Sequence[Sequence[float]],
        values: Sequence[Sequence[float]],
        query: Sequence[float],
        kappa: float = 1.0,
    ) -> tuple[list[float], list[float], list[float]]:
        """Compute local window attention weights and barycenters.

        Returns (weights, kbar, vbar).
        """
        w = len(keys)
        d_k = len(query)
        if w == 0:
            d_v = len(values[0]) if len(values) > 0 else 0
            return [], [0.0] * d_k, [0.0] * d_v

        d_v = len(values[0])

        # Dot scores: s_i = kappa * sum_d q_d * k_id
        scores = [
            kappa * sum(query[d] * keys[i][d] for d in range(d_k))
            for i in range(w)
        ]

        # Softmax with max subtraction
        max_s = max(scores)
        exp_s = [math.exp(s - max_s) for s in scores]
        sum_exp = sum(exp_s)
        weights = [e / sum_exp for e in exp_s]

        # Barycenters
        kbar = [sum(weights[i] * keys[i][d] for i in range(w)) for d in range(d_k)]
        vbar = [sum(weights[i] * values[i][v] for i in range(w)) for v in range(d_v)]

        return weights, kbar, vbar

    @classmethod
    def bounded_read(
        cls,
        query: Sequence[float],
        keys: Sequence[Sequence[float]],
        values: Sequence[Sequence[float]],
        S: Sequence[Sequence[float]],
        kappa: float = 1.0,
    ) -> list[float]:
        """Evaluate bounded read (Eq. 3): r(q) = vbar_L + S_t(q - kbar_L)."""
        d_v = len(S)
        d_k = len(S[0])
        _, kbar, vbar = cls.local_attention(keys, values, query, kappa=kappa)

        diff = [query[j] - kbar[j] for j in range(d_k)]
        # S @ diff
        transport = [sum(S[i][j] * diff[j] for j in range(d_k)) for i in range(d_v)]
        return [vbar[i] + transport[i] for i in range(d_v)]

    @classmethod
    def full_softmax(
        cls,
        keys: Sequence[Sequence[float]],
        values: Sequence[Sequence[float]],
        query: Sequence[float],
        kappa: float = 1.0,
    ) -> list[float]:
        """Explicit full softmax attention (Eq. 7) ground truth y_*."""
        t = len(keys)
        d_k = len(query)
        d_v = len(values[0])
        if t == 0:
            return [0.0] * d_v

        scores = [
            kappa * sum(query[d] * keys[i][d] for d in range(d_k))
            for i in range(t)
        ]
        max_s = max(scores)
        exp_s = [math.exp(s - max_s) for s in scores]
        sum_exp = sum(exp_s)

        out = [0.0] * d_v
        for i in range(t):
            weight = exp_s[i] / sum_exp
            for v in range(d_v):
                out[v] += weight * values[i][v]
        return out

    @classmethod
    def completed_read(
        cls,
        selected_mass: float,
        estimated_unread_mass: float,
        selected_numerator: Sequence[float],
        prior: Sequence[float],
    ) -> list[float]:
        """Evaluate normalized archive completion (Eq. 8).

        y_hat = (N_A + Z_hat_O * r) / (Z_A + Z_hat_O)
        """
        denom = selected_mass + estimated_unread_mass
        return [
            (selected_numerator[v] + estimated_unread_mass * prior[v]) / denom
            for v in range(len(prior))
        ]

    @classmethod
    def compute_page_envelopes(
        cls,
        page_keys: Sequence[Sequence[float]],
        page_values: Sequence[Sequence[float]],
        query: Sequence[float],
        prior: Sequence[float],
        kappa: float = 1.0,
    ) -> ScalarPageEnvelope:
        """Compute page envelopes and bounds (Eqs. 11, 12)."""
        n = len(page_keys)
        d_k = len(query)
        d_v = len(prior)

        # Coordinate bounds k^- and k^+
        k_min = [min(page_keys[i][d] for i in range(n)) for d in range(d_k)]
        k_max = [max(page_keys[i][d] for i in range(n)) for d in range(d_k)]

        # Center c_j = mean(values)
        center = [sum(page_values[i][v] for i in range(n)) / float(n) for v in range(d_v)]

        # Radius rho_j = max ||v_i - c_j||_2
        radius = 0.0
        for i in range(n):
            dist = math.sqrt(sum((page_values[i][v] - center[v]) ** 2 for v in range(d_v)))
            if dist > radius:
                radius = dist

        # ell_j and u_j (Eq. 11)
        # ell_j = kappa * sum_d min(q_d k^-_d, q_d k^+_d)
        # u_j   = kappa * sum_d max(q_d k^-_d, q_d k^+_d)
        ell = kappa * sum(min(query[d] * k_min[d], query[d] * k_max[d]) for d in range(d_k))
        u = kappa * sum(max(query[d] * k_min[d], query[d] * k_max[d]) for d in range(d_k))

        # L_j = n * exp(ell), U_j = n * exp(u)
        L = float(n) * math.exp(ell)
        U = float(n) * math.exp(u)

        # b_j(r) = U_j * (||c_j - r||_2 + rho_j) (Eq. 12)
        diff_r = math.sqrt(sum((center[v] - prior[v]) ** 2 for v in range(d_v)))
        b = U * (diff_r + radius)

        return ScalarPageEnvelope(
            count=n,
            key_min=k_min,
            key_max=k_max,
            center=center,
            radius=radius,
            ell=ell,
            u=u,
            L=L,
            U=U,
            b=b,
        )

    @classmethod
    def residual_certificate(
        cls,
        selected_mass: float,
        unread_lower: float,
        unread_upper: float,
        residual_bound: float,
        prior: Sequence[float],
        completed: Sequence[float],
    ) -> ScalarCertificateResult:
        """Evaluate deterministic residual certificate (Eq. 10).

        E_A = (B_O + eta * ||r - y_hat_A||_2) / (Z_A + L_O)
        """
        eta = (unread_upper - unread_lower) / 2.0
        est_unread = (unread_lower + unread_upper) / 2.0
        prior_diff = math.sqrt(sum((prior[v] - completed[v]) ** 2 for v in range(len(prior))))

        denom = selected_mass + unread_lower
        if denom <= 0.0:
            bound = float("inf")
        else:
            bound = (residual_bound + eta * prior_diff) / denom

        return ScalarCertificateResult(
            bound=bound,
            selected_mass=selected_mass,
            estimated_unread_mass=est_unread,
            unread_mass_lower=unread_lower,
            unread_mass_upper=unread_upper,
            residual_bound=residual_bound,
            denominator_floor=denom,
        )

    @classmethod
    def grouped_completed_read(
        cls,
        selected_mass: float,
        selected_numerator: Sequence[float],
        page_estimates: Sequence[float],
        page_predictors: Sequence[Sequence[float]],
    ) -> list[float]:
        """Mass-consistent grouped completion (Eq. 13).

        y_hat = (N_A + sum_j Z_hat_j p_j) / (Z_A + sum_j Z_hat_j)
        """
        d_v = len(selected_numerator)
        sum_zh = sum(page_estimates)
        denom = selected_mass + sum_zh

        pred_num = [0.0] * d_v
        for j, zh in enumerate(page_estimates):
            for v in range(d_v):
                pred_num[v] += zh * page_predictors[j][v]

        return [(selected_numerator[v] + pred_num[v]) / denom for v in range(d_v)]

    @classmethod
    def grouped_residual_certificate(
        cls,
        selected_mass: float,
        unread_lowers: Sequence[float],
        unread_uppers: Sequence[float],
        page_bounds: Sequence[float],
        page_predictors: Sequence[Sequence[float]],
        completed: Sequence[float],
    ) -> float:
        """Evaluate grouped comparator certificate bound (Eq. 13).

        bound = sum_j [B_j + eta_j ||p_j - y_hat||_2] / (Z_A + sum_j L_j)
        """
        sum_L = sum(unread_lowers)
        denom = selected_mass + sum_L
        if denom <= 0.0:
            return float("inf")

        num = 0.0
        for j in range(len(unread_lowers)):
            eta_j = (unread_uppers[j] - unread_lowers[j]) / 2.0
            diff_j = math.sqrt(
                sum((page_predictors[j][v] - completed[v]) ** 2 for v in range(len(completed)))
            )
            num += page_bounds[j] + eta_j * diff_j

        return num / denom


# ==============================================================================
# 2. TENSOR REFERENCE ORACLE (PyTorch float64 operations)
# ==============================================================================


@dataclass(frozen=True)
class TensorCertificateResult:
    bound: float
    selected_mass: float
    estimated_unread_mass: float
    unread_mass_lower: float
    unread_mass_upper: float
    residual_bound: float
    denominator_floor: float


@dataclass(frozen=True)
class TensorPageEnvelope:
    count: int
    key_min: Tensor
    key_max: Tensor
    center: Tensor
    radius: float
    ell: float
    u: float
    L: float
    U: float
    b: float


class TensorOracle:
    """Independent tensor CPU oracle using PyTorch float64 and linear algebra."""

    @staticmethod
    def normalize_key(key: Tensor) -> Tensor:
        """Normalize key vector to Euclidean norm <= 1 (Eq. 2)."""
        k = key.to(dtype=torch.float64)
        norm = torch.linalg.vector_norm(k, dim=-1, keepdim=True)
        scale = torch.clamp(norm, min=1.0)
        return k / scale

    @staticmethod
    def zero_matrix(d_v: int, d_k: int) -> Tensor:
        """Return zero matrix S in R^{d_v x d_k} with dtype float64."""
        return torch.zeros((d_v, d_k), dtype=torch.float64)

    @classmethod
    def gated_delta_update(
        cls,
        S: Tensor,
        key: Tensor,
        value: Tensor,
        alpha: float | Tensor = 1.0,
        beta: float | Tensor = 1.0,
    ) -> Tensor:
        """Evaluate solve-free gated delta recurrent update (Eq. 2)."""
        S_64 = S.to(dtype=torch.float64)
        k = cls.normalize_key(key)
        v = value.to(dtype=torch.float64)
        a = torch.as_tensor(alpha, dtype=torch.float64)
        b = torch.as_tensor(beta, dtype=torch.float64)

        S_tilde = a * S_64
        pred = torch.matmul(S_tilde, k)
        err = v - pred
        delta = b * torch.outer(err, k)
        return S_tilde + delta

    @classmethod
    def local_attention(
        cls,
        keys: Tensor,
        values: Tensor,
        query: Tensor,
        kappa: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute local window attention weights and barycenters in fp64."""
        k_64 = keys.to(dtype=torch.float64)
        v_64 = values.to(dtype=torch.float64)
        q_64 = query.to(dtype=torch.float64)

        w = k_64.shape[0]
        if w == 0:
            d_k = q_64.shape[-1]
            d_v = v_64.shape[-1] if v_64.ndim > 1 else 0
            return (
                torch.empty(0, dtype=torch.float64),
                torch.zeros(d_k, dtype=torch.float64),
                torch.zeros(d_v, dtype=torch.float64),
            )

        # Dot product scores s_i = kappa * q^T k_i
        scores = kappa * torch.mv(k_64, q_64)
        weights = torch.softmax(scores, dim=-1)

        kbar = torch.matmul(weights, k_64)
        vbar = torch.matmul(weights, v_64)
        return weights, kbar, vbar

    @classmethod
    def bounded_read(
        cls,
        query: Tensor,
        keys: Tensor,
        values: Tensor,
        S: Tensor,
        kappa: float = 1.0,
    ) -> Tensor:
        """Evaluate bounded read (Eq. 3): r(q) = vbar_L + S_t(q - kbar_L)."""
        S_64 = S.to(dtype=torch.float64)
        q_64 = query.to(dtype=torch.float64)
        _, kbar, vbar = cls.local_attention(keys, values, q_64, kappa=kappa)

        diff = q_64 - kbar
        transport = torch.matmul(S_64, diff)
        return vbar + transport

    @classmethod
    def full_softmax(
        cls,
        keys: Tensor,
        values: Tensor,
        query: Tensor,
        kappa: float = 1.0,
    ) -> Tensor:
        """Explicit full softmax attention (Eq. 7) ground truth y_*."""
        k_64 = keys.to(dtype=torch.float64)
        v_64 = values.to(dtype=torch.float64)
        q_64 = query.to(dtype=torch.float64)

        scores = kappa * torch.mv(k_64, q_64)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v_64)

    @classmethod
    def completed_read(
        cls,
        selected_mass: float | Tensor,
        estimated_unread_mass: float | Tensor,
        selected_numerator: Tensor,
        prior: Tensor,
    ) -> Tensor:
        """Evaluate normalized archive completion (Eq. 8)."""
        zs = torch.as_tensor(selected_mass, dtype=torch.float64)
        zh = torch.as_tensor(estimated_unread_mass, dtype=torch.float64)
        num = selected_numerator.to(dtype=torch.float64)
        r = prior.to(dtype=torch.float64)
        return (num + zh * r) / (zs + zh)

    @classmethod
    def compute_page_envelopes(
        cls,
        page_keys: Tensor,
        page_values: Tensor,
        query: Tensor,
        prior: Tensor,
        kappa: float = 1.0,
    ) -> TensorPageEnvelope:
        """Compute page envelopes and bounds (Eqs. 11, 12)."""
        pk = page_keys.to(dtype=torch.float64)
        pv = page_values.to(dtype=torch.float64)
        q = query.to(dtype=torch.float64)
        r = prior.to(dtype=torch.float64)

        n = pk.shape[0]
        k_min = torch.min(pk, dim=0).values
        k_max = torch.max(pk, dim=0).values

        center = torch.mean(pv, dim=0)
        radii = torch.linalg.vector_norm(pv - center, dim=-1)
        radius = float(torch.max(radii).item())

        # Coordinate box score interval (Eq. 11)
        qk_min = q * k_min
        qk_max = q * k_max
        coord_min = torch.minimum(qk_min, qk_max)
        coord_max = torch.maximum(qk_min, qk_max)

        ell = float((kappa * torch.sum(coord_min)).item())
        u = float((kappa * torch.sum(coord_max)).item())

        L = float(n) * math.exp(ell)
        U = float(n) * math.exp(u)

        diff_r = float(torch.linalg.vector_norm(center - r).item())
        b = U * (diff_r + radius)

        return TensorPageEnvelope(
            count=n,
            key_min=k_min,
            key_max=k_max,
            center=center,
            radius=radius,
            ell=ell,
            u=u,
            L=L,
            U=U,
            b=b,
        )

    @classmethod
    def residual_certificate(
        cls,
        selected_mass: float,
        unread_lower: float,
        unread_upper: float,
        residual_bound: float,
        prior: Tensor,
        completed: Tensor,
    ) -> TensorCertificateResult:
        """Evaluate deterministic residual certificate (Eq. 10)."""
        eta = (unread_upper - unread_lower) / 2.0
        est_unread = (unread_lower + unread_upper) / 2.0

        r = prior.to(dtype=torch.float64)
        y = completed.to(dtype=torch.float64)
        prior_diff = float(torch.linalg.vector_norm(r - y).item())

        denom = selected_mass + unread_lower
        if denom <= 0.0:
            bound = float("inf")
        else:
            bound = (residual_bound + eta * prior_diff) / denom

        return TensorCertificateResult(
            bound=bound,
            selected_mass=selected_mass,
            estimated_unread_mass=est_unread,
            unread_mass_lower=unread_lower,
            unread_mass_upper=unread_upper,
            residual_bound=residual_bound,
            denominator_floor=denom,
        )

    @classmethod
    def grouped_completed_read(
        cls,
        selected_mass: float | Tensor,
        selected_numerator: Tensor,
        page_estimates: Sequence[float] | Tensor,
        page_predictors: Tensor,
    ) -> Tensor:
        """Mass-consistent grouped completion (Eq. 13)."""
        zs = torch.as_tensor(selected_mass, dtype=torch.float64)
        zh = torch.as_tensor(page_estimates, dtype=torch.float64)
        num = selected_numerator.to(dtype=torch.float64)
        p = page_predictors.to(dtype=torch.float64)

        # sum_j Z_hat_j * p_j
        pred_num = torch.einsum("j,jv->v", zh, p)
        denom = zs + torch.sum(zh)
        return (num + pred_num) / denom

    @classmethod
    def grouped_residual_certificate(
        cls,
        selected_mass: float,
        unread_lowers: Sequence[float] | Tensor,
        unread_uppers: Sequence[float] | Tensor,
        page_bounds: Sequence[float] | Tensor,
        page_predictors: Tensor,
        completed: Tensor,
    ) -> float:
        """Evaluate grouped comparator certificate bound (Eq. 13)."""
        L = torch.as_tensor(unread_lowers, dtype=torch.float64)
        U = torch.as_tensor(unread_uppers, dtype=torch.float64)
        B = torch.as_tensor(page_bounds, dtype=torch.float64)
        p = page_predictors.to(dtype=torch.float64)
        y = completed.to(dtype=torch.float64)

        denom = selected_mass + float(torch.sum(L).item())
        if denom <= 0.0:
            return float("inf")

        eta = (U - L) / 2.0
        diff_norms = torch.linalg.vector_norm(p - y, dim=-1)
        num = torch.sum(B + eta * diff_norms)
        return float((num / denom).item())


# ==============================================================================
# 3. STREAMING ORACLES (fp64) FOR DISJOINT CAUSAL HANDOFF
# ==============================================================================


class ScalarStreamingOracle:
    """Stateful scalar streaming oracle with explicit ring buffer and recurrence."""

    def __init__(self, d_v: int, d_k: int, window: int) -> None:
        self.d_v = d_v
        self.d_k = d_k
        self.window = window
        self.S = ScalarOracle.zero_matrix(d_v, d_k)
        self.buffer_keys: list[list[float]] = []
        self.buffer_values: list[list[float]] = []
        self.buffer_alphas: list[float] = []
        self.buffer_betas: list[float] = []
        self.buffer_ids: list[int] = []
        self.evicted_ids: list[int] = []
        self.evicted_keys: list[list[float]] = []
        self.evicted_values: list[list[float]] = []
        self.step_count = 0

    def consume(
        self,
        key: Sequence[float],
        value: Sequence[float],
        alpha: float = 1.0,
        beta: float = 1.0,
        occurrence_id: int | None = None,
    ) -> None:
        oid = occurrence_id if occurrence_id is not None else self.step_count
        self.step_count += 1

        if len(self.buffer_ids) == self.window:
            # Evict oldest
            evicted_k = self.buffer_keys.pop(0)
            evicted_v = self.buffer_values.pop(0)
            evicted_a = self.buffer_alphas.pop(0)
            evicted_b = self.buffer_betas.pop(0)
            evicted_id = self.buffer_ids.pop(0)

            # Update S
            self.S = ScalarOracle.gated_delta_update(
                self.S, evicted_k, evicted_v, alpha=evicted_a, beta=evicted_b
            )
            self.evicted_ids.append(evicted_id)
            self.evicted_keys.append(evicted_k)
            self.evicted_values.append(evicted_v)

        # Append to window
        self.buffer_keys.append([float(x) for x in key])
        self.buffer_values.append([float(x) for x in value])
        self.buffer_alphas.append(float(alpha))
        self.buffer_betas.append(float(beta))
        self.buffer_ids.append(oid)

    def bounded_read(self, query: Sequence[float], kappa: float = 1.0) -> list[float]:
        return ScalarOracle.bounded_read(
            query, self.buffer_keys, self.buffer_values, self.S, kappa=kappa
        )

    def get_partition(self) -> tuple[list[int], list[int]]:
        return list(self.evicted_ids), list(self.buffer_ids)


class TensorStreamingOracle:
    """Stateful tensor streaming oracle (torch.float64) with ring buffer."""

    def __init__(self, d_v: int, d_k: int, window: int) -> None:
        self.d_v = d_v
        self.d_k = d_k
        self.window = window
        self.S = TensorOracle.zero_matrix(d_v, d_k)
        self.buffer_keys: list[Tensor] = []
        self.buffer_values: list[Tensor] = []
        self.buffer_alphas: list[float] = []
        self.buffer_betas: list[float] = []
        self.buffer_ids: list[int] = []
        self.evicted_ids: list[int] = []
        self.evicted_keys: list[Tensor] = []
        self.evicted_values: list[Tensor] = []
        self.step_count = 0

    def consume(
        self,
        key: Tensor,
        value: Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
        occurrence_id: int | None = None,
    ) -> None:
        oid = occurrence_id if occurrence_id is not None else self.step_count
        self.step_count += 1

        k_64 = key.to(dtype=torch.float64)
        v_64 = value.to(dtype=torch.float64)

        if len(self.buffer_ids) == self.window:
            evicted_k = self.buffer_keys.pop(0)
            evicted_v = self.buffer_values.pop(0)
            evicted_a = self.buffer_alphas.pop(0)
            evicted_b = self.buffer_betas.pop(0)
            evicted_id = self.buffer_ids.pop(0)

            self.S = TensorOracle.gated_delta_update(
                self.S, evicted_k, evicted_v, alpha=evicted_a, beta=evicted_b
            )
            self.evicted_ids.append(evicted_id)
            self.evicted_keys.append(evicted_k)
            self.evicted_values.append(evicted_v)

        self.buffer_keys.append(k_64)
        self.buffer_values.append(v_64)
        self.buffer_alphas.append(float(alpha))
        self.buffer_betas.append(float(beta))
        self.buffer_ids.append(oid)

    def bounded_read(self, query: Tensor, kappa: float = 1.0) -> Tensor:
        if len(self.buffer_keys) == 0:
            keys_tensor = torch.empty((0, self.d_k), dtype=torch.float64)
            vals_tensor = torch.empty((0, self.d_v), dtype=torch.float64)
        else:
            keys_tensor = torch.stack(self.buffer_keys, dim=0)
            vals_tensor = torch.stack(self.buffer_values, dim=0)
        return TensorOracle.bounded_read(
            query, keys_tensor, vals_tensor, self.S, kappa=kappa
        )

    def get_partition(self) -> tuple[list[int], list[int]]:
        return list(self.evicted_ids), list(self.buffer_ids)

