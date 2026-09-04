"""Suite generators and evaluators for Phase 4: Nonstationarity, composition, and capacity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class Phase4Batch:
    """Standard batch container for Phase 4 nonstationarity and compositional suites."""

    x: Tensor  # [batch, length, d_in]
    y: Tensor  # [batch, length, d_out]
    mask: Tensor  # [batch, length]
    cue: Tensor | None  # [batch, length] observable drift/changepoint cue
    true_precisions: Tensor  # [batch, length] ground truth evidence precisions
    corrupted_precisions: Tensor  # [batch, length] corrupted evidence precisions
    metadata: dict[str, Any]


class Phase4SuiteGenerator:
    """Generates synthetic tasks for the 7 required suites of Phase 4."""

    def __init__(
        self,
        d_in: int = 16,
        d_out: int = 4,
        d_feat: int = 8,
        default_window: int = 12,
        seed: int = 42,
        device: torch.device | str = "cpu",
    ) -> None:
        self.d_in = d_in
        self.d_out = d_out
        self.d_feat = d_feat
        self.default_window = default_window
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def set_seed(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    # -------------------------------------------------------------------------
    # Suite 1: Abrupt & Gradual Operator Drift (Observable vs Unobservable)
    # -------------------------------------------------------------------------
    def generate_operator_drift(
        self,
        batch_size: int,
        length: int = 64,
        *,
        drift_type: Literal["abrupt", "gradual", "stationary"] = "abrupt",
        observable: bool = True,
        drift_point: int | None = None,
    ) -> Phase4Batch:
        B, L = batch_size, length
        t_star = drift_point or (L // 2)

        x = torch.zeros(B, L, self.d_in, device=self.device)
        y = torch.zeros(B, L, self.d_out, device=self.device)
        mask = torch.zeros(B, L, device=self.device)
        cue = torch.zeros(B, L, device=self.device)
        precisions = torch.ones(B, L, device=self.device)

        for i in range(B):
            W1 = torch.randn(self.d_feat, self.d_out, generator=self.generator, device=self.device) / math.sqrt(self.d_feat)
            W2 = torch.randn(self.d_feat, self.d_out, generator=self.generator, device=self.device) / math.sqrt(self.d_feat)

            if drift_type == "stationary":
                W2 = W1

            # Sequence context
            x_seq = torch.randn(L - 1, self.d_feat, generator=self.generator, device=self.device)
            y_seq = torch.zeros(L - 1, self.d_out, device=self.device)

            for t in range(L - 1):
                if t < t_star or drift_type == "stationary":
                    W_t = W1
                    cue_val = 0.0
                elif drift_type == "abrupt":
                    W_t = W2
                    # Changepoint pulse at t_star
                    cue_val = 1.0 if t == t_star else 0.0
                elif drift_type == "gradual":
                    alpha = min(1.0, (t - t_star) / max(1, L - 1 - t_star))
                    W_t = (1.0 - alpha) * W1 + alpha * W2
                    cue_val = 1.0 / max(1, L - 1 - t_star)  # Constant positive drift rate
                else:
                    W_t = W1
                    cue_val = 0.0

                noise = torch.randn(self.d_out, generator=self.generator, device=self.device) * 0.1
                y_seq[t] = x_seq[t] @ W_t + noise
                if observable:
                    cue[i, t] = cue_val

            x[i, :-1, : self.d_feat] = x_seq
            x[i, :-1, self.d_feat : self.d_feat + self.d_out] = y_seq

            # Query at L-1: tests the post-drift operator W2
            q = torch.randn(self.d_feat, generator=self.generator, device=self.device)
            x[i, -1, : self.d_feat] = q
            x[i, -1, self.d_in - 2] = 1.0  # is_query bit
            if observable and drift_type == "abrupt" and t_star == L - 1:
                cue[i, -1] = 1.0
            x[i, -1, self.d_in - 1] = cue[i, -1]

            y[i, -1] = q @ W2
            mask[i, -1] = 1.0

        return Phase4Batch(
            x=x,
            y=y,
            mask=mask,
            cue=cue if observable else torch.zeros_like(cue),
            true_precisions=precisions,
            corrupted_precisions=precisions,
            metadata={"drift_type": drift_type, "observable": observable, "t_star": t_star},
        )

    # -------------------------------------------------------------------------
    # Suite 2: Heterogeneous Write Precision, Corrupted Labels, Outliers, Nonlinear
    # -------------------------------------------------------------------------
    def generate_heterogeneous_precision(
        self,
        batch_size: int,
        length: int = 48,
        *,
        noise_distribution: Literal["heteroscedastic", "student_t", "outliers", "nonlinear"] = "heteroscedastic",
        corruption_type: Literal["none", "inverted", "uniform", "scrambled"] = "none",
    ) -> Phase4Batch:
        B, L = batch_size, length
        x = torch.zeros(B, L, self.d_in, device=self.device)
        y = torch.zeros(B, L, self.d_out, device=self.device)
        mask = torch.zeros(B, L, device=self.device)
        true_prec = torch.ones(B, L, device=self.device)
        corrupted_prec = torch.ones(B, L, device=self.device)

        for i in range(B):
            W = torch.randn(self.d_feat, self.d_out, generator=self.generator, device=self.device) / math.sqrt(self.d_feat)
            x_seq = torch.randn(L - 1, self.d_feat, generator=self.generator, device=self.device)
            y_seq = torch.zeros(L - 1, self.d_out, device=self.device)

            # Assign heterogeneous noise stds: half the tokens clean (std=0.05), half noisy (std=0.50)
            is_noisy = torch.rand(L - 1, generator=self.generator, device=self.device) < 0.5
            stds = torch.where(is_noisy, torch.tensor(0.50, device=self.device), torch.tensor(0.05, device=self.device))
            p_true = 1.0 / (stds**2)  # 400.0 vs 4.0
            true_prec[i, :-1] = p_true

            # Generate noise
            if noise_distribution == "student_t":
                # Student-t with nu=3
                u1 = torch.randn(L - 1, self.d_out, generator=self.generator, device=self.device)
                u2 = torch.randn(L - 1, 3, generator=self.generator, device=self.device).square().sum(dim=-1, keepdim=True) / 3.0
                noise = (u1 / torch.sqrt(u2.clamp_min(1e-4))) * stds.unsqueeze(-1)
            elif noise_distribution == "outliers":
                noise = torch.randn(L - 1, self.d_out, generator=self.generator, device=self.device) * stds.unsqueeze(-1)
                # 5% extreme outliers
                outlier_mask = torch.rand(L - 1, generator=self.generator, device=self.device) < 0.05
                noise[outlier_mask] = torch.randn(outlier_mask.sum().item(), self.d_out, generator=self.generator, device=self.device) * 5.0
            else:
                noise = torch.randn(L - 1, self.d_out, generator=self.generator, device=self.device) * stds.unsqueeze(-1)

            if noise_distribution == "nonlinear":
                # Misspecified nonlinear term
                U = torch.randn(self.d_feat, self.d_out, generator=self.generator, device=self.device) / math.sqrt(self.d_feat)
                y_seq = x_seq @ W + 0.3 * torch.tanh(x_seq @ U) + noise
            else:
                y_seq = x_seq @ W + noise

            # Apply corruption to evidence labels
            if corruption_type == "inverted":
                corrupted_prec[i, :-1] = 1.0 / p_true  # Inverted: high precision to noisy
            elif corruption_type == "uniform":
                corrupted_prec[i, :-1] = 1.0
            elif corruption_type == "scrambled":
                perm = torch.randperm(L - 1, generator=self.generator, device=self.device)
                corrupted_prec[i, :-1] = p_true[perm]
            else:
                corrupted_prec[i, :-1] = p_true

            x[i, :-1, : self.d_feat] = x_seq
            x[i, :-1, self.d_feat : self.d_feat + self.d_out] = y_seq

            # Query at L-1: test latent linear target Wq
            q = torch.randn(self.d_feat, generator=self.generator, device=self.device)
            x[i, -1, : self.d_feat] = q
            x[i, -1, self.d_in - 2] = 1.0
            true_prec[i, -1] = 100.0
            corrupted_prec[i, -1] = 100.0

            if noise_distribution == "nonlinear":
                y[i, -1] = q @ W + 0.3 * torch.tanh(q @ U)
            else:
                y[i, -1] = q @ W
            mask[i, -1] = 1.0

        return Phase4Batch(
            x=x,
            y=y,
            mask=mask,
            cue=None,
            true_precisions=true_prec,
            corrupted_precisions=corrupted_prec,
            metadata={"noise_distribution": noise_distribution, "corruption_type": corruption_type},
        )

    # -------------------------------------------------------------------------
    # Suite 3: Repeated Updates, Overrides, Many-to-One, State Pollution
    # -------------------------------------------------------------------------
    def generate_repeated_overrides(
        self,
        batch_size: int,
        length: int = 48,
        *,
        override_type: Literal["cache_override", "remote_override", "many_to_one", "state_pollution"] = "cache_override",
        pollution_count: int = 16,
    ) -> Phase4Batch:
        B, L = batch_size, length
        win = self.default_window
        x = torch.zeros(B, L, self.d_in, device=self.device)
        y = torch.zeros(B, L, self.d_out, device=self.device)
        mask = torch.zeros(B, L, device=self.device)
        cue = torch.zeros(B, L, device=self.device)

        for i in range(B):
            if override_type == "cache_override":
                # Key K written at remote pos (t1 < L-1-win) with V1
                # Overwritten at cache pos (t2 >= L-1-win) with V2
                k = F.normalize(torch.randn(self.d_feat, generator=self.generator, device=self.device), dim=-1)
                v1 = torch.randn(self.d_out, generator=self.generator, device=self.device)
                v2 = torch.randn(self.d_out, generator=self.generator, device=self.device)

                t1 = 4
                t2 = L - 3
                x[i, t1, : self.d_feat] = k
                x[i, t1, self.d_feat : self.d_feat + self.d_out] = v1
                x[i, t2, : self.d_feat] = k
                x[i, t2, self.d_feat : self.d_feat + self.d_out] = v2

                # Query K, expected answer is V2
                x[i, -1, : self.d_feat] = k
                x[i, -1, self.d_in - 2] = 1.0
                y[i, -1] = v2

            elif override_type == "remote_override":
                # Both writes are remote (t1 < t2 < L-1-win)
                k = F.normalize(torch.randn(self.d_feat, generator=self.generator, device=self.device), dim=-1)
                v1 = torch.randn(self.d_out, generator=self.generator, device=self.device)
                v2 = torch.randn(self.d_out, generator=self.generator, device=self.device)

                t1 = 4
                t2 = 12
                x[i, t1, : self.d_feat] = k
                x[i, t1, self.d_feat : self.d_feat + self.d_out] = v1
                x[i, t2, : self.d_feat] = k
                x[i, t2, self.d_feat : self.d_feat + self.d_out] = v2
                # Observable override cue at t2
                cue[i, t2] = 1.0

                x[i, -1, : self.d_feat] = k
                x[i, -1, self.d_in - 2] = 1.0
                y[i, -1] = v2

            elif override_type == "many_to_one":
                # Multiple distinct keys K1, K2, K3 map to same value V
                v_shared = torch.randn(self.d_out, generator=self.generator, device=self.device)
                keys = F.normalize(torch.randn(4, self.d_feat, generator=self.generator, device=self.device), dim=-1)
                for j in range(4):
                    x[i, j * 3, : self.d_feat] = keys[j]
                    x[i, j * 3, self.d_feat : self.d_feat + self.d_out] = v_shared

                # Query random key among them
                chosen = torch.randint(0, 4, (1,), generator=self.generator, device=self.device).item()
                x[i, -1, : self.d_feat] = keys[chosen]
                x[i, -1, self.d_in - 2] = 1.0
                y[i, -1] = v_shared

            else:  # state_pollution
                # Target association stored at pos 2
                k_target = F.normalize(torch.randn(self.d_feat, generator=self.generator, device=self.device), dim=-1)
                v_target = torch.randn(self.d_out, generator=self.generator, device=self.device)
                x[i, 2, : self.d_feat] = k_target
                x[i, 2, self.d_feat : self.d_feat + self.d_out] = v_target

                # Insert pollution items
                n_poll = min(pollution_count, L - 4)
                for j in range(n_poll):
                    pos = 3 + j
                    x[i, pos, : self.d_feat] = torch.randn(self.d_feat, generator=self.generator, device=self.device) * 0.5
                    x[i, pos, self.d_feat : self.d_feat + self.d_out] = torch.randn(self.d_out, generator=self.generator, device=self.device) * 0.5

                x[i, -1, : self.d_feat] = k_target
                x[i, -1, self.d_in - 2] = 1.0
                y[i, -1] = v_target

            mask[i, -1] = 1.0

        return Phase4Batch(
            x=x,
            y=y,
            mask=mask,
            cue=cue,
            true_precisions=torch.ones(B, L, device=self.device),
            corrupted_precisions=torch.ones(B, L, device=self.device),
            metadata={"override_type": override_type, "pollution_count": pollution_count},
        )

    # -------------------------------------------------------------------------
    # Suite 4: Pointer Chasing & Multi-Hop Composition (Hops in {1, 2, 4, 8, 16})
    # -------------------------------------------------------------------------
    def generate_pointer_chasing(
        self,
        batch_size: int,
        length: int = 48,
        *,
        hops: int = 4,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Generate a pointer chasing chain: K[0] -> K[1] -> ... -> K[hops] = target.

        Returns hidden sequence, initial query K[0], and target list [K[1], ..., K[hops]].
        """
        B, L = batch_size, length
        hidden = torch.zeros(B, L, self.d_in, device=self.device)

        initial_queries = torch.zeros(B, self.d_feat, device=self.device)
        all_chains = []

        for i in range(B):
            chain = torch.randn(hops + 1, self.d_feat, generator=self.generator, device=self.device)
            chain = F.normalize(chain, dim=-1)
            all_chains.append(chain)

            # Store links: link j is chain[j] -> chain[j+1]
            for j in range(hops):
                pos = 2 + j * 2
                assert pos < L - 1, f"Chain length {hops} exceeds sequence length {L}"
                hidden[i, pos, : self.d_feat] = chain[j]
                hidden[i, pos, self.d_feat : self.d_feat + self.d_out] = chain[j + 1][: self.d_out]

            initial_queries[i] = chain[0]

        chain_targets = [torch.zeros(B, self.d_out, device=self.device) for _ in range(hops)]
        for i in range(B):
            for h in range(hops):
                chain_targets[h][i] = all_chains[i][h + 1][: self.d_out]

        return hidden, initial_queries, chain_targets

    # -------------------------------------------------------------------------
    # Suite 5: Cache/Remote Mixed-Hop Chains in Every Order
    # -------------------------------------------------------------------------
    def generate_mixed_chain(
        self,
        batch_size: int,
        length: int = 64,
        *,
        pattern: list[Literal["C", "R"]] = ["C", "C", "R", "R"],
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Generate mixed cache ('C') and remote ('R') multi-hop chain in exact requested order."""
        B, L = batch_size, length
        win = self.default_window
        hidden = torch.zeros(B, L, self.d_in, device=self.device)
        hops = len(pattern)

        initial_queries = torch.zeros(B, self.d_feat, device=self.device)
        all_chains = []

        for i in range(B):
            chain = F.normalize(torch.randn(hops + 1, self.d_feat, generator=self.generator, device=self.device), dim=-1)
            all_chains.append(chain)

            remote_pos = 2
            cache_pos = L - 1 - win + 2

            for j, loc in enumerate(pattern):
                if loc == "R":
                    pos = remote_pos
                    remote_pos += 3
                else:
                    pos = cache_pos
                    cache_pos += 2
                assert 0 <= pos < L - 1, f"Position {pos} out of range"

                hidden[i, pos, : self.d_feat] = chain[j]
                hidden[i, pos, self.d_feat : self.d_feat + self.d_out] = chain[j + 1][: self.d_out]

            initial_queries[i] = chain[0]

        chain_targets = [torch.zeros(B, self.d_out, device=self.device) for _ in range(hops)]
        for i in range(B):
            for h in range(hops):
                chain_targets[h][i] = all_chains[i][h + 1][: self.d_out]

        return hidden, initial_queries, chain_targets

    # -------------------------------------------------------------------------
    # Suite 6: Capacity Sweeps (Rank, Adversarial Associations, State Bytes)
    # -------------------------------------------------------------------------
    def generate_adversarial_capacity(
        self,
        batch_size: int,
        length: int,
        *,
        num_associations: int = 16,
    ) -> Phase4Batch:
        B, L = batch_size, length
        x = torch.zeros(B, L, self.d_in, device=self.device)
        y = torch.zeros(B, L, self.d_out, device=self.device)
        mask = torch.zeros(B, L, device=self.device)

        for i in range(B):
            keys = F.normalize(torch.randn(num_associations, self.d_feat, generator=self.generator, device=self.device), dim=-1)
            vals = torch.randn(num_associations, self.d_out, generator=self.generator, device=self.device)

            for j in range(min(num_associations, L - 2)):
                x[i, j, : self.d_feat] = keys[j]
                x[i, j, self.d_feat : self.d_feat + self.d_out] = vals[j]

            chosen = torch.randint(0, min(num_associations, L - 2), (1,), generator=self.generator, device=self.device).item()
            x[i, -1, : self.d_feat] = keys[chosen]
            x[i, -1, self.d_in - 2] = 1.0
            y[i, -1] = vals[chosen]
            mask[i, -1] = 1.0

        return Phase4Batch(
            x=x,
            y=y,
            mask=mask,
            cue=None,
            true_precisions=torch.ones(B, L, device=self.device),
            corrupted_precisions=torch.ones(B, L, device=self.device),
            metadata={"num_associations": num_associations},
        )
