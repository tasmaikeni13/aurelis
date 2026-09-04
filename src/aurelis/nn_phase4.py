"""Learned feature projections, drift-aware routing, and compositional access for Phase 4."""

from __future__ import annotations

import math
import time
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


class DriftAwareAurelisBlock(nn.Module):
    """Multi-head AURELIS memory block supporting observable-cue decay, tempered evidence, and overrides."""

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
        gamma_min: float = 0.05,
        beta_min: float = 0.01,
        beta_max: float = 100.0,
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
        self.gamma_min = gamma_min
        self.beta_min = beta_min
        self.beta_max = beta_max

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

        # Log temperature per head
        self.log_temperature = nn.Parameter(torch.zeros(heads))

        # Output projection
        self.output_proj = nn.Linear(heads * d_value, d_model, bias=False)

    def _split_heads(self, tensor: Tensor, dim: int) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.heads, dim).transpose(1, 2)

    def forward(
        self,
        hidden: Tensor,
        *,
        cue: Tensor | None = None,
        override_evidence: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        batch, length, _ = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        # Compute keys and queries
        if self.shared_charts:
            assert self.key_query is not None
            kq = self.key_query(hidden)
            keys = self._split_heads(kq, self.d_key)  # [B, H, L, d_k]
            queries = self._split_heads(kq, self.d_key)  # [B, H, L, d_k]
        else:
            assert self.key is not None and self.query is not None
            keys = self._split_heads(self.key(hidden), self.d_key)
            queries = self._split_heads(self.query(hidden), self.d_key)

        values = self._split_heads(self.value(hidden), self.d_value)  # [B, H, L, d_v]

        # Evidence beta_t with tempering
        if override_evidence is not None:
            evidence = override_evidence.clamp(self.beta_min, self.beta_max)
        elif self.learned_evidence:
            assert self.evidence_proj is not None
            raw_ev = F.softplus(self.evidence_proj(hidden)).transpose(1, 2) + 1e-6
            evidence = raw_ev.clamp(self.beta_min, self.beta_max)  # [B, H, L]
        else:
            evidence = torch.ones(batch, self.heads, length, device=device, dtype=dtype)

        # Episodic responsibility e_t
        responsibility = torch.sigmoid(self.episodic_proj(hidden)).transpose(1, 2)  # [B, H, L]

        # Compute step decays gamma_t from observable cue
        # cue has shape [B, L] or [B, H, L] with values in [0, 1]
        if cue is not None:
            if cue.ndim == 2:
                c = cue.unsqueeze(1).expand(batch, self.heads, length)
            else:
                c = cue
            # gamma_t = 1 - cue_t * (1 - gamma_min)
            step_decay = (1.0 - c * (1.0 - self.gamma_min)).clamp(self.gamma_min, 1.0)
        else:
            step_decay = torch.ones(batch, self.heads, length, device=device, dtype=dtype)

        # Vectorized discounted outer products
        outer_precision = torch.einsum("bhti,bhtj,bht->bhtij", keys, keys, evidence)
        outer_cross = torch.einsum("bhtv,bhtd,bht->bhtvd", values, keys, evidence)

        # Cumulative discount factors D[b, h, t, s] = prod_{j=s+1}^t gamma_j
        # D[b, h, t, s] = exp(cumsum_log(t) - cumsum_log(s)) for s <= t
        log_decay = torch.log(step_decay.clamp_min(1e-6))
        cum_log = torch.cumsum(log_decay, dim=-1)  # [B, H, L]
        # Pairwise log discount: cum_log[t] - cum_log[s]
        pairwise_log = cum_log.unsqueeze(-1) - cum_log.unsqueeze(-2)  # [B, H, L, L] where (t, s)
        D = torch.exp(pairwise_log.clamp_max(0.0))  # [B, H, L, L]

        # Mask causal past and delayed window
        positions = torch.arange(length, device=device)
        t_pos = positions[:, None]
        s_pos = positions[None, :]
        causal_delayed_mask = (s_pos <= t_pos - self.window) & (s_pos >= 0)
        D_delayed = D.masked_fill(~causal_delayed_mask.view(1, 1, length, length), 0.0)

        # Contract over historical token s <= t - window
        remote_precision = torch.einsum("bhts,bhsij->bhtij", D_delayed, outer_precision)
        remote_cross = torch.einsum("bhts,bhsvd->bhtvd", D_delayed, outer_cross)

        eye = torch.eye(self.d_key, dtype=dtype, device=device)
        precision = remote_precision + self.prior * eye.view(1, 1, 1, self.d_key, self.d_key)
        cross = remote_cross

        # Local causal softmax attention over recent window w
        tau = self.log_temperature.exp().view(1, self.heads, 1, 1)
        scores = torch.einsum("bhtd,bhsd->bhts", queries, keys) * tau
        local_mask = (s_pos <= t_pos) & (s_pos > t_pos - self.window)
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
        else:
            raise ValueError(f"Unknown gate mode: {self.gate_mode}")

        joined = read.transpose(1, 2).reshape(batch, length, self.heads * self.d_value)
        out = self.output_proj(joined)

        diag = {
            "g_B": g_B,
            "g_E": g_E,
            "gate": gate,
            "e_t": responsibility,
            "V_R": V_R,
            "V_H": V_H,
            "K_RH": K_RH,
            "h": h,
            "attention": attention,
            "remote": remote,
            "full_residual": full_residual,
            "vbar": vbar,
            "kbar": kbar,
            "evidence": evidence,
            "precision": precision,
            "cross": cross,
            "factors": factors,
            "temperature": self.log_temperature.exp(),
            "erank_kq": compute_effective_rank(self.key_query.weight if self.shared_charts else self.query.weight),
        }
        return out, diag


class MultiHopPointerChaser:
    """Evaluates multi-hop composition and pointer chasing through the AURELIS memory state."""

    def __init__(
        self,
        block: DriftAwareAurelisBlock,
        d_feat: int,
        d_out: int,
        temperature: float = 8.0,
    ) -> None:
        self.block = block
        self.d_feat = d_feat
        self.d_out = d_out
        self.temperature = temperature

    def chase_pointers(
        self,
        hidden: Tensor,
        initial_query_feat: Tensor,
        max_hops: int = 16,
        *,
        adaptive: bool = False,
        tol: float = 0.05,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Perform multi-hop query chasing at the final position.

        Distinguishes adaptive round count, vector error, decoded success, operator norm,
        confidence, and actual measured latency.
        """
        device = hidden.device
        dtype = hidden.dtype
        B = hidden.shape[0]

        # First run forward pass to construct full memory state (precision, cross, cache)
        with torch.no_grad():
            _, diag = self.block(hidden)

        precision = diag["precision"][:, :, -1]  # [B, H, d_k, d_k]
        cross = diag["cross"][:, :, -1]  # [B, H, d_v, d_k]
        factors = diag["factors"][:, :, -1]  # [B, H, d_k, d_k]

        # Compute operator norm ||M|| per head: M = C P^-1
        eye_k = torch.eye(self.block.d_key, dtype=dtype, device=device).expand(B, self.block.heads, self.block.d_key, self.block.d_key)
        inv_P = torch.cholesky_solve(eye_k, factors)
        M = torch.einsum("bhvd,bhdj->bhvj", cross, inv_P)  # [B, H, d_v, d_k]
        operator_norms = torch.linalg.matrix_norm(M, ord=2).mean().item()

        # Cache items for local attention
        assert self.block.key_query is not None
        kq = self.block.key_query(hidden)
        keys_all = self.block._split_heads(kq, self.block.d_key)  # [B, H, L, d_k]
        vals_all = self.block._split_heads(self.block.value(hidden), self.block.d_value)  # [B, H, L, d_v]
        ev_all = diag["evidence"]  # [B, H, L]

        L = hidden.shape[1]
        w = self.block.window
        cache_start = max(0, L - w)
        keys_cache = keys_all[:, :, cache_start:, :]
        vals_cache = vals_all[:, :, cache_start:, :]
        ev_cache = ev_all[:, :, cache_start:]

        temp = temperature if temperature is not None else self.temperature
        tau = torch.full((1, self.block.heads, 1), temp, device=device, dtype=dtype)

        # Iterative pointer chasing
        q_feat = initial_query_feat.clone()  # [B, d_key]
        hop_outputs = []
        hop_confidences = []
        hop_latencies = []

        rounds_taken = max_hops
        start_time = time.perf_counter()

        for h_idx in range(max_hops):
            t0 = time.perf_counter()

            # Expand query to multi-head: [B, H, d_k]
            q_h = q_feat.unsqueeze(1).expand(B, self.block.heads, self.block.d_key)

            # 1. Local window read
            raw_sims = torch.einsum("bhnd,bhd->bhn", keys_cache, q_h)
            max_sim = raw_sims.amax(dim=-1)  # [B, H]
            scores = raw_sims * tau
            attn = torch.softmax(scores, dim=-1)
            kbar = torch.einsum("bhn,bhnd->bhd", attn, keys_cache)
            vbar = torch.einsum("bhn,bhnv->bhv", attn, vals_cache)
            h_var = torch.sum(attn.square() / ev_cache, dim=-1)

            # 2. Remote solve
            residual_q = q_h - kbar
            rhs = torch.stack((q_h, kbar, residual_q), dim=-1)
            solved = torch.cholesky_solve(rhs, factors)
            p_q, p_k, p_res = solved.unbind(dim=-1)

            remote = torch.einsum("bhvd,bhd->bhv", cross, p_q)
            mapped_k = torch.einsum("bhvd,bhd->bhv", cross, p_k)
            innov = vbar - mapped_k

            # Gate: Bayes posterior gated by cache presence discrimination
            denom = (h_var + torch.sum(kbar * p_k, dim=-1)).clamp_min(1e-12)
            g_raw = torch.sum(q_h * p_k, dim=-1) / denom
            g_B = torch.clamp(g_raw, 0.0, 1.0)
            cache_presence = torch.sigmoid((max_sim - 0.70) * 20.0)
            g = cache_presence * torch.maximum(g_B, torch.tensor(1.0, device=device))

            y_read = remote + g.unsqueeze(-1) * innov  # [B, H, d_v]
            y_mean = y_read.mean(dim=1)  # [B, d_v]

            t1 = time.perf_counter()
            hop_latencies.append((t1 - t0) * 1000.0)
            hop_outputs.append(y_mean)
            hop_confidences.append(g.mean().item())

            # Check adaptive stopping if requested
            if adaptive and h_idx > 0:
                step_diff = torch.linalg.vector_norm(y_mean - hop_outputs[-2], dim=-1).mean().item()
                if step_diff < tol:
                    rounds_taken = h_idx + 1
                    break

            # Update next query: normalize retrieved vector into key space
            q_feat = F.normalize(y_mean, dim=-1)

        total_latency = (time.perf_counter() - start_time) * 1000.0

        return {
            "hop_outputs": hop_outputs,
            "rounds_taken": rounds_taken,
            "hop_confidences": hop_confidences,
            "hop_latencies_ms": hop_latencies,
            "total_latency_ms": total_latency,
            "operator_norm": operator_norms,
        }


class Phase4SequenceModel(nn.Module):
    """Sequence model wrapping DriftAwareAurelisBlock with pointer chasing capabilities."""

    def __init__(
        self,
        d_in: int,
        d_model: int,
        d_out: int,
        block: DriftAwareAurelisBlock,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.encoder = nn.Linear(d_in, d_model)
        self.block = block
        self.decoder = nn.Linear(d_model, d_out)
        self.pointer_chaser = MultiHopPointerChaser(block, block.d_key, block.d_value)

    def forward(
        self,
        x: Tensor,
        *,
        cue: Tensor | None = None,
        override_evidence: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        h = self.encoder(x)
        h_mem, diag = self.block(h, cue=cue, override_evidence=override_evidence)
        pred = self.decoder(h_mem)
        return pred, diag

    def chase_pointers(
        self,
        x: Tensor,
        initial_query_feat: Tensor,
        max_hops: int = 16,
        *,
        adaptive: bool = False,
        tol: float = 0.05,
    ) -> dict[str, Any]:
        h = self.encoder(x)
        return self.pointer_chaser.chase_pointers(
            h, initial_query_feat, max_hops=max_hops, adaptive=adaptive, tol=tol
        )

