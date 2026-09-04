"""Learned feature projections and episodic routing architectures for Phase 3."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .types import SequenceOutput


def compute_effective_rank(matrix: Tensor) -> float:
    """Compute Roy & Vetterli (2007) effective rank: exp(Shannon entropy of singular values)."""
    if matrix.ndim != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    s = torch.linalg.svdvals(matrix.detach().float())
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum().item()
    return math.exp(entropy)


class LearnedAurelisBlock(nn.Module):
    """Multi-head learned AURELIS memory block supporting shared/independent charts and gate modes."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        d_key: int,
        d_value: int,
        window: int,
        *,
        prior: float = 1.0,
        shared_charts: bool = True,
        learned_evidence: bool = True,
        gate_mode: Literal["aurelis_b", "aurelis_e", "fixed_0", "fixed_1", "learned_sigmoid"] = "aurelis_e",
        cache_overlap: bool = False,
        frozen_features: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_key = d_key
        self.d_value = d_value
        self.window = window
        self.prior = prior
        self.shared_charts = shared_charts
        self.learned_evidence = learned_evidence
        self.gate_mode = gate_mode
        self.cache_overlap = cache_overlap
        self.frozen_features = frozen_features

        # Key & Query projections
        if shared_charts:
            self.key_query = nn.Linear(d_model, heads * d_key, bias=False)
            self.key = None
            self.query = None
        else:
            self.key_query = None
            self.key = nn.Linear(d_model, heads * d_key, bias=False)
            self.query = nn.Linear(d_model, heads * d_key, bias=False)

        # Value projection
        self.value = nn.Linear(d_model, heads * d_value, bias=False)

        # Evidence projection
        if learned_evidence:
            self.evidence_proj = nn.Linear(d_model, heads, bias=True)
        else:
            self.evidence_proj = None

        # Episodic override projection
        self.episodic_proj = nn.Linear(d_model, heads, bias=True)
        self.episodic_proj.bias.data.fill_(-2.0)

        # Learned sigmoid gate projection (for ablation)
        if gate_mode == "learned_sigmoid":
            self.learned_gate_proj = nn.Linear(d_model, heads, bias=True)
        else:
            self.learned_gate_proj = None

        # Log temperature per head
        self.log_temperature = nn.Parameter(torch.zeros(heads))

        # Output projection
        self.output_proj = nn.Linear(heads * d_value, d_model, bias=False)

        if frozen_features:
            for p in self.parameters():
                p.requires_grad = False

    def _split_heads(self, tensor: Tensor, dim: int) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.heads, dim).transpose(1, 2)

    def forward(self, hidden: Tensor) -> tuple[Tensor, dict[str, Any]]:
        batch, length, _ = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        # Compute keys and queries
        if self.shared_charts:
            assert self.key_query is not None
            kq = self.key_query(hidden)
            keys = self._split_heads(kq, self.d_key)
            queries = self._split_heads(kq, self.d_key)
        else:
            assert self.key is not None and self.query is not None
            keys = self._split_heads(self.key(hidden), self.d_key)
            queries = self._split_heads(self.query(hidden), self.d_key)

        values = self._split_heads(self.value(hidden), self.d_value)

        # Evidence beta_t
        if self.learned_evidence:
            assert self.evidence_proj is not None
            evidence = F.softplus(self.evidence_proj(hidden)).transpose(1, 2) + 1e-6
        else:
            evidence = torch.ones(batch, self.heads, length, device=device, dtype=dtype)

        # Episodic responsibility e_t
        responsibility = torch.sigmoid(self.episodic_proj(hidden)).transpose(1, 2)

        # Remote Bayesian statistics
        outer_precision = torch.einsum("bhti,bhtj,bht->bhtij", keys, keys, evidence)
        outer_cross = torch.einsum("bhtv,bhtd,bht->bhtvd", values, keys, evidence)
        prefix_precision = torch.cumsum(outer_precision, dim=2)
        prefix_cross = torch.cumsum(outer_cross, dim=2)

        if self.cache_overlap:
            # Ablation: Known-invalid double counting without window delay
            remote_precision = prefix_precision
            remote_cross = prefix_cross
        else:
            # Delayed handoff: recent window is strictly excluded from remote store
            if self.window >= length:
                remote_precision = torch.zeros_like(prefix_precision)
                remote_cross = torch.zeros_like(prefix_cross)
            else:
                remote_precision = torch.cat(
                    (torch.zeros_like(prefix_precision[:, :, : self.window]), prefix_precision[:, :, : -self.window]),
                    dim=2,
                )
                remote_cross = torch.cat(
                    (torch.zeros_like(prefix_cross[:, :, : self.window]), prefix_cross[:, :, : -self.window]),
                    dim=2,
                )

        eye = torch.eye(self.d_key, dtype=dtype, device=device)
        precision = remote_precision + self.prior * eye.view(1, 1, 1, self.d_key, self.d_key)
        cross = remote_cross

        # Local causal softmax attention over recent window w
        tau = self.log_temperature.exp().view(1, self.heads, 1, 1)
        scores = torch.einsum("bhtd,bhsd->bhts", queries, keys) * tau
        positions = torch.arange(length, device=device)
        q_pos = positions[:, None]
        k_pos = positions[None, :]
        local_mask = (k_pos <= q_pos) & (k_pos > q_pos - self.window)
        scores = scores.masked_fill(~local_mask.view(1, 1, length, length), -torch.inf)
        attention = torch.softmax(scores, dim=-1)

        kbar = torch.einsum("bhts,bhsd->bhtd", attention, keys)
        vbar = torch.einsum("bhts,bhsv->bhtv", attention, values)
        h = torch.sum(attention.square() / evidence.unsqueeze(-2), dim=-1)

        residual_query = queries - kbar
        rhs = torch.stack((queries, kbar, residual_query), dim=-1)

        # Solve precision factor
        factors = torch.linalg.cholesky(precision)
        solved = torch.cholesky_solve(rhs, factors)
        p_query, p_kbar, p_residual = solved.unbind(dim=-1)

        remote = torch.einsum("bhtvd,bhtd->bhtv", cross, p_query)
        mapped_kbar = torch.einsum("bhtvd,bhtd->bhtv", cross, p_kbar)
        innovation = vbar - mapped_kbar
        full_residual = remote + innovation

        # Variances and Covariances
        V_R = torch.sum(queries * p_query, dim=-1)
        V_H = h + torch.sum(residual_query * p_residual, dim=-1)
        K_RH = torch.sum(queries * p_residual, dim=-1)
        denominator = (h + torch.sum(kbar * p_kbar, dim=-1)).clamp_min(1e-12)
        g_raw = torch.sum(queries * p_kbar, dim=-1) / denominator
        g_B = torch.clamp(g_raw, 0.0, 1.0)
        g_E_hard = torch.maximum(g_B, responsibility)
        g_E_soft = g_B + (1.0 - g_B) * responsibility
        g_E = g_E_hard + (g_E_soft - g_E_soft.detach())

        # Gate selection
        if self.gate_mode == "aurelis_b":
            gate = g_B
            read = remote + gate.unsqueeze(-1) * innovation
        elif self.gate_mode == "aurelis_e":
            gate = g_E
            read = remote + gate.unsqueeze(-1) * innovation
        elif self.gate_mode == "fixed_0":
            gate = torch.zeros_like(g_B)
            read = remote
        elif self.gate_mode == "fixed_1":
            gate = torch.ones_like(g_B)
            read = full_residual
        elif self.gate_mode == "learned_sigmoid":
            assert self.learned_gate_proj is not None
            gate = torch.sigmoid(self.learned_gate_proj(hidden)).transpose(1, 2)
            read = remote + gate.unsqueeze(-1) * innovation
        else:
            raise ValueError(f"Unknown gate mode: {self.gate_mode}")

        joined = read.transpose(1, 2).reshape(batch, length, self.heads * self.d_value)
        out = self.output_proj(joined)

        # Diagnostics
        diag = {
            "g_B": g_B,
            "g_E": g_E,
            "gate": gate,
            "e_t": responsibility,
            "V_R": V_R,
            "V_H": V_H,
            "K_RH": K_RH,
            "remote": remote,
            "full_residual": full_residual,
            "vbar": vbar,
            "kbar": kbar,
            "evidence": evidence,
            "temperature": self.log_temperature.exp(),
            "erank_kq": compute_effective_rank(self.key_query.weight if self.shared_charts else self.query.weight),
        }
        return out, diag


class LocalOnlyBlock(nn.Module):
    """Pure local causal softmax attention over recent window w."""

    def __init__(self, d_model: int, heads: int, d_key: int, d_value: int, window: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_key = d_key
        self.d_value = d_value
        self.window = window

        self.key_query = nn.Linear(d_model, heads * d_key, bias=False)
        self.value = nn.Linear(d_model, heads * d_value, bias=False)
        self.log_temperature = nn.Parameter(torch.zeros(heads))
        self.output_proj = nn.Linear(heads * d_value, d_model, bias=False)

    def _split_heads(self, tensor: Tensor, dim: int) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.heads, dim).transpose(1, 2)

    def forward(self, hidden: Tensor) -> tuple[Tensor, dict[str, Any]]:
        batch, length, _ = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        kq = self.key_query(hidden)
        keys = self._split_heads(kq, self.d_key)
        queries = self._split_heads(kq, self.d_key)
        values = self._split_heads(self.value(hidden), self.d_value)

        tau = self.log_temperature.exp().view(1, self.heads, 1, 1)
        scores = torch.einsum("bhtd,bhsd->bhts", queries, keys) * tau
        positions = torch.arange(length, device=device)
        q_pos = positions[:, None]
        k_pos = positions[None, :]
        local_mask = (k_pos <= q_pos) & (k_pos > q_pos - self.window)
        scores = scores.masked_fill(~local_mask.view(1, 1, length, length), -torch.inf)
        attention = torch.softmax(scores, dim=-1)

        vbar = torch.einsum("bhts,bhsv->bhtv", attention, values)
        joined = vbar.transpose(1, 2).reshape(batch, length, self.heads * self.d_value)
        out = self.output_proj(joined)
        diag = {
            "vbar": vbar,
            "attention": attention,
            "erank_kq": compute_effective_rank(self.key_query.weight),
        }
        return out, diag


class RemoteOnlyBlock(nn.Module):
    """Pure remote Bayesian ridge linear memory (g=0)."""

    def __init__(
        self, d_model: int, heads: int, d_key: int, d_value: int, window: int, *, prior: float = 1.0
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_key = d_key
        self.d_value = d_value
        self.window = window
        self.prior = prior

        self.key_query = nn.Linear(d_model, heads * d_key, bias=False)
        self.value = nn.Linear(d_model, heads * d_value, bias=False)
        self.evidence_proj = nn.Linear(d_model, heads, bias=True)
        self.output_proj = nn.Linear(heads * d_value, d_model, bias=False)

    def _split_heads(self, tensor: Tensor, dim: int) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.heads, dim).transpose(1, 2)

    def forward(self, hidden: Tensor) -> tuple[Tensor, dict[str, Any]]:
        batch, length, _ = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        kq = self.key_query(hidden)
        keys = self._split_heads(kq, self.d_key)
        queries = self._split_heads(kq, self.d_key)
        values = self._split_heads(self.value(hidden), self.d_value)
        evidence = F.softplus(self.evidence_proj(hidden)).transpose(1, 2) + 1e-6

        outer_precision = torch.einsum("bhti,bhtj,bht->bhtij", keys, keys, evidence)
        outer_cross = torch.einsum("bhtv,bhtd,bht->bhtvd", values, keys, evidence)
        prefix_precision = torch.cumsum(outer_precision, dim=2)
        prefix_cross = torch.cumsum(outer_cross, dim=2)

        if self.window >= length:
            remote_precision = torch.zeros_like(prefix_precision)
            remote_cross = torch.zeros_like(prefix_cross)
        else:
            remote_precision = torch.cat(
                (torch.zeros_like(prefix_precision[:, :, : self.window]), prefix_precision[:, :, : -self.window]),
                dim=2,
            )
            remote_cross = torch.cat(
                (torch.zeros_like(prefix_cross[:, :, : self.window]), prefix_cross[:, :, : -self.window]),
                dim=2,
            )

        eye = torch.eye(self.d_key, dtype=dtype, device=device)
        precision = remote_precision + self.prior * eye.view(1, 1, 1, self.d_key, self.d_key)
        factors = torch.linalg.cholesky(precision)
        p_query = torch.cholesky_solve(queries.unsqueeze(-1), factors).squeeze(-1)
        remote = torch.einsum("bhtvd,bhtd->bhtv", remote_cross, p_query)

        joined = remote.transpose(1, 2).reshape(batch, length, self.heads * self.d_value)
        out = self.output_proj(joined)
        diag = {
            "remote": remote,
            "erank_kq": compute_effective_rank(self.key_query.weight),
        }
        return out, diag


class LearnedSumBlock(nn.Module):
    """Learned convex combination of local attention and remote ridge memory."""

    def __init__(
        self, d_model: int, heads: int, d_key: int, d_value: int, window: int, *, prior: float = 1.0
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_key = d_key
        self.d_value = d_value
        self.window = window
        self.prior = prior

        self.key_query = nn.Linear(d_model, heads * d_key, bias=False)
        self.value = nn.Linear(d_model, heads * d_value, bias=False)
        self.evidence_proj = nn.Linear(d_model, heads, bias=True)
        self.log_temperature = nn.Parameter(torch.zeros(heads))
        self.alpha = nn.Parameter(torch.zeros(heads))  # logit for convex mixture
        self.output_proj = nn.Linear(heads * d_value, d_model, bias=False)

    def _split_heads(self, tensor: Tensor, dim: int) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.heads, dim).transpose(1, 2)

    def forward(self, hidden: Tensor) -> tuple[Tensor, dict[str, Any]]:
        batch, length, _ = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        kq = self.key_query(hidden)
        keys = self._split_heads(kq, self.d_key)
        queries = self._split_heads(kq, self.d_key)
        values = self._split_heads(self.value(hidden), self.d_value)
        evidence = F.softplus(self.evidence_proj(hidden)).transpose(1, 2) + 1e-6

        # Remote ridge
        outer_precision = torch.einsum("bhti,bhtj,bht->bhtij", keys, keys, evidence)
        outer_cross = torch.einsum("bhtv,bhtd,bht->bhtvd", values, keys, evidence)
        prefix_precision = torch.cumsum(outer_precision, dim=2)
        prefix_cross = torch.cumsum(outer_cross, dim=2)

        if self.window >= length:
            remote_precision = torch.zeros_like(prefix_precision)
            remote_cross = torch.zeros_like(prefix_cross)
        else:
            remote_precision = torch.cat(
                (torch.zeros_like(prefix_precision[:, :, : self.window]), prefix_precision[:, :, : -self.window]),
                dim=2,
            )
            remote_cross = torch.cat(
                (torch.zeros_like(prefix_cross[:, :, : self.window]), prefix_cross[:, :, : -self.window]),
                dim=2,
            )

        eye = torch.eye(self.d_key, dtype=dtype, device=device)
        precision = remote_precision + self.prior * eye.view(1, 1, 1, self.d_key, self.d_key)
        factors = torch.linalg.cholesky(precision)
        p_query = torch.cholesky_solve(queries.unsqueeze(-1), factors).squeeze(-1)
        remote = torch.einsum("bhtvd,bhtd->bhtv", remote_cross, p_query)

        # Local attention
        tau = self.log_temperature.exp().view(1, self.heads, 1, 1)
        scores = torch.einsum("bhtd,bhsd->bhts", queries, keys) * tau
        positions = torch.arange(length, device=device)
        q_pos = positions[:, None]
        k_pos = positions[None, :]
        local_mask = (k_pos <= q_pos) & (k_pos > q_pos - self.window)
        scores = scores.masked_fill(~local_mask.view(1, 1, length, length), -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        vbar = torch.einsum("bhts,bhsv->bhtv", attention, values)

        # Learned convex combination
        mix = torch.sigmoid(self.alpha).view(1, self.heads, 1, 1)
        combined = (1.0 - mix) * remote + mix * vbar

        joined = combined.transpose(1, 2).reshape(batch, length, self.heads * self.d_value)
        out = self.output_proj(joined)
        diag = {
            "remote": remote,
            "vbar": vbar,
            "alpha": torch.sigmoid(self.alpha),
            "erank_kq": compute_effective_rank(self.key_query.weight),
        }
        return out, diag


class GatedDeltaBlock(nn.Module):
    """Sequential Gated DeltaNet recurrent memory state."""

    def __init__(self, d_model: int, heads: int, d_key: int, d_value: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_key = d_key
        self.d_value = d_value

        self.key = nn.Linear(d_model, heads * d_key, bias=False)
        self.query = nn.Linear(d_model, heads * d_key, bias=False)
        self.value = nn.Linear(d_model, heads * d_value, bias=False)
        self.beta_proj = nn.Linear(d_model, heads, bias=True)
        self.decay_proj = nn.Linear(d_model, heads, bias=True)
        self.output_proj = nn.Linear(heads * d_value, d_model, bias=False)

    def _split_heads(self, tensor: Tensor, dim: int) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.heads, dim).transpose(1, 2)

    def forward(self, hidden: Tensor) -> tuple[Tensor, dict[str, Any]]:
        batch, length, _ = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        keys = self._split_heads(self.key(hidden), self.d_key)
        queries = self._split_heads(self.query(hidden), self.d_key)
        values = self._split_heads(self.value(hidden), self.d_value)
        betas = torch.sigmoid(self.beta_proj(hidden)).transpose(1, 2)
        decays = torch.sigmoid(self.decay_proj(hidden)).transpose(1, 2)

        # Recurrent state S [batch, heads, d_value, d_key]
        S = torch.zeros(batch, self.heads, self.d_value, self.d_key, device=device, dtype=dtype)
        out_list = []

        for t in range(length):
            k_t = keys[:, :, t, :]  # [B, H, d_k]
            q_t = queries[:, :, t, :]  # [B, H, d_k]
            v_t = values[:, :, t, :]  # [B, H, d_v]
            beta_t = betas[:, :, t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
            decay_t = decays[:, :, t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

            norm_k = torch.linalg.vector_norm(k_t, dim=-1, keepdim=True).clamp_min(1e-6)
            k_hat = k_t / norm_k  # [B, H, d_k]

            # Causal output at step t from state S_{t-1} or after update
            v_pred = torch.einsum("bhvd,bhd->bhv", S, k_hat)
            err = v_t - v_pred
            S = decay_t * S + beta_t * torch.einsum("bhv,bhd->bhvd", err, k_hat)

            y_t = torch.einsum("bhvd,bhd->bhv", S, q_t)
            out_list.append(y_t)

        out_tensor = torch.stack(out_list, dim=2)  # [B, H, L, d_v]
        joined = out_tensor.transpose(1, 2).reshape(batch, length, self.heads * self.d_value)
        out = self.output_proj(joined)
        diag = {
            "erank_kq": compute_effective_rank(self.key.weight),
        }
        return out, diag


class CumulativeLeastSquaresBlock(nn.Module):
    """Cumulative least squares (Mesa-style) over the entire prefix without delay."""

    def __init__(
        self, d_model: int, heads: int, d_key: int, d_value: int, *, prior: float = 1.0
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_key = d_key
        self.d_value = d_value
        self.prior = prior

        self.key_query = nn.Linear(d_model, heads * d_key, bias=False)
        self.value = nn.Linear(d_model, heads * d_value, bias=False)
        self.evidence_proj = nn.Linear(d_model, heads, bias=True)
        self.output_proj = nn.Linear(heads * d_value, d_model, bias=False)

    def _split_heads(self, tensor: Tensor, dim: int) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.heads, dim).transpose(1, 2)

    def forward(self, hidden: Tensor) -> tuple[Tensor, dict[str, Any]]:
        batch, length, _ = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        kq = self.key_query(hidden)
        keys = self._split_heads(kq, self.d_key)
        queries = self._split_heads(kq, self.d_key)
        values = self._split_heads(self.value(hidden), self.d_value)
        evidence = F.softplus(self.evidence_proj(hidden)).transpose(1, 2) + 1e-6

        # Cumulative sums over the entire prefix without delay (s <= t or s < t)
        # Causal prefix prior to current token: exclude current token or include
        # For standard cumulative linear regression: prefix includes all s < t
        outer_precision = torch.einsum("bhti,bhtj,bht->bhtij", keys, keys, evidence)
        outer_cross = torch.einsum("bhtv,bhtd,bht->bhtvd", values, keys, evidence)
        prefix_precision = torch.cumsum(outer_precision, dim=2)
        prefix_cross = torch.cumsum(outer_cross, dim=2)

        # Shift by 1 for strict causality
        causal_precision = torch.cat((torch.zeros_like(prefix_precision[:, :, :1]), prefix_precision[:, :, :-1]), dim=2)
        causal_cross = torch.cat((torch.zeros_like(prefix_cross[:, :, :1]), prefix_cross[:, :, :-1]), dim=2)

        eye = torch.eye(self.d_key, dtype=dtype, device=device)
        precision = causal_precision + self.prior * eye.view(1, 1, 1, self.d_key, self.d_key)
        factors = torch.linalg.cholesky(precision)
        p_query = torch.cholesky_solve(queries.unsqueeze(-1), factors).squeeze(-1)
        read = torch.einsum("bhtvd,bhtd->bhtv", causal_cross, p_query)

        joined = read.transpose(1, 2).reshape(batch, length, self.heads * self.d_value)
        out = self.output_proj(joined)
        diag = {
            "read": read,
            "erank_kq": compute_effective_rank(self.key_query.weight),
        }
        return out, diag


class Phase3SequenceModel(nn.Module):
    """Matched encoder-memory-decoder sequence model ensuring identical capacity and optimization opportunity."""

    def __init__(
        self,
        d_in: int,
        d_model: int,
        d_out: int,
        block: nn.Module,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.encoder = nn.Linear(d_in, d_model)
        self.block = block
        self.decoder = nn.Linear(d_model, d_out)

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Any]]:
        h = self.encoder(x)
        h_mem, diag = self.block(h)
        pred = self.decoder(h_mem)
        return pred, diag
