"""Synthetic curriculum generator covering 7 task families for Phase 3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class CurriculumBatch:
    """Standard container for a batch of synthetic curriculum sequences."""

    x: Tensor  # [batch, length, d_in]
    y: Tensor  # [batch, length, d_out]
    mask: Tensor  # [batch, length]
    family_id: int
    family_name: str
    metadata: dict[str, Any]


class CurriculumGenerator:
    """Generates synthetic in-context tasks with configurable training and held-out distributions."""

    def __init__(
        self,
        d_in: int = 16,
        d_out: int = 4,
        d_feat: int = 10,
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
    # Task Family 1: Noisy linear and affine in-context regression
    # -------------------------------------------------------------------------
    def generate_task1_noisy_linear(
        self,
        batch_size: int,
        length: int,
        *,
        noise_std: float = 0.15,
    ) -> CurriculumBatch:
        B, L = batch_size, length
        x = torch.zeros(B, L, self.d_in, device=self.device)
        y = torch.zeros(B, L, self.d_out, device=self.device)
        mask = torch.zeros(B, L, device=self.device)

        # Random ground truth affine map per sequence
        W = torch.randn(B, self.d_feat, self.d_out, generator=self.generator, device=self.device) / math.sqrt(self.d_feat)
        b = torch.randn(B, 1, self.d_out, generator=self.generator, device=self.device) * 0.2

        # Context features
        x_ctx = torch.randn(B, L - 1, self.d_feat, generator=self.generator, device=self.device)
        noise = torch.randn(B, L - 1, self.d_out, generator=self.generator, device=self.device) * noise_std
        y_ctx = torch.bmm(x_ctx, W) + b + noise

        # Store context in X
        x[:, :-1, : self.d_feat] = x_ctx
        x[:, :-1, self.d_feat : self.d_feat + self.d_out] = y_ctx
        # Query token at L-1
        x_query = torch.randn(B, 1, self.d_feat, generator=self.generator, device=self.device)
        y_target = torch.bmm(x_query, W) + b  # Denoised true linear target

        x[:, -1:, : self.d_feat] = x_query
        x[:, -1, self.d_in - 2] = 1.0  # is_query bit
        x[:, -1, self.d_in - 1] = 0.0  # cue bit = 0

        y[:, -1:] = y_target
        mask[:, -1] = 1.0

        return CurriculumBatch(
            x=x,
            y=y,
            mask=mask,
            family_id=1,
            family_name="noisy_linear_regression",
            metadata={"noise_std": noise_std},
        )

    # -------------------------------------------------------------------------
    # Task Family 2: Recent exact associative copy
    # -------------------------------------------------------------------------
    def generate_task2_recent_copy(
        self,
        batch_size: int,
        length: int,
        *,
        window: int | None = None,
    ) -> CurriculumBatch:
        B, L = batch_size, length
        win = window or self.default_window
        x = torch.zeros(B, L, self.d_in, device=self.device)
        y = torch.zeros(B, L, self.d_out, device=self.device)
        mask = torch.zeros(B, L, device=self.device)

        # Distribute items recently: age between 1 and win-1
        num_items = min(4, win - 2)
        target_indices = torch.randint(0, num_items, (B,), generator=self.generator, device=self.device)

        for i in range(B):
            keys = torch.randn(num_items, self.d_feat, generator=self.generator, device=self.device)
            keys = F.normalize(keys, dim=-1)
            vals = torch.randn(num_items, self.d_out, generator=self.generator, device=self.device)

            # Place items at positions (L - 1 - num_items) to (L - 2)
            start_pos = L - 1 - num_items
            for j in range(num_items):
                pos = start_pos + j
                x[i, pos, : self.d_feat] = keys[j]
                x[i, pos, self.d_feat : self.d_feat + self.d_out] = vals[j]

            # Query at L-1
            chosen = target_indices[i].item()
            x[i, -1, : self.d_feat] = keys[chosen]
            x[i, -1, self.d_in - 2] = 1.0  # is_query
            x[i, -1, self.d_in - 1] = 0.0  # cue = 0
            y[i, -1] = vals[chosen]
            mask[i, -1] = 1.0

        return CurriculumBatch(
            x=x,
            y=y,
            mask=mask,
            family_id=2,
            family_name="recent_associative_copy",
            metadata={"num_items": num_items, "window": win},
        )

    # -------------------------------------------------------------------------
    # Task Family 3: Remote structured recall within rank
    # -------------------------------------------------------------------------
    def generate_task3_remote_recall(
        self,
        batch_size: int,
        length: int,
        *,
        rank: int = 4,
        window: int | None = None,
    ) -> CurriculumBatch:
        B, L = batch_size, length
        win = window or self.default_window
        x = torch.zeros(B, L, self.d_in, device=self.device)
        y = torch.zeros(B, L, self.d_out, device=self.device)
        mask = torch.zeros(B, L, device=self.device)

        # Keys stored early in sequence (positions 0 to rank-1, where age > win)
        target_indices = torch.randint(0, rank, (B,), generator=self.generator, device=self.device)

        for i in range(B):
            keys = torch.randn(rank, self.d_feat, generator=self.generator, device=self.device)
            # Orthogonalize keys for perfect rank capacity
            q, _ = torch.linalg.qr(keys.T)
            keys = q[:, :rank].T
            vals = torch.randn(rank, self.d_out, generator=self.generator, device=self.device)

            for j in range(rank):
                x[i, j, : self.d_feat] = keys[j]
                x[i, j, self.d_feat : self.d_feat + self.d_out] = vals[j]

            # Query at L-1
            chosen = target_indices[i].item()
            x[i, -1, : self.d_feat] = keys[chosen]
            x[i, -1, self.d_in - 2] = 1.0
            x[i, -1, self.d_in - 1] = 0.0
            y[i, -1] = vals[chosen]
            mask[i, -1] = 1.0

        return CurriculumBatch(
            x=x,
            y=y,
            mask=mask,
            family_id=3,
            family_name="remote_structured_recall",
            metadata={"rank": rank, "window": win},
        )

    # -------------------------------------------------------------------------
    # Task Family 4: Mixed latent-denoising and episodic-exception targets with observable cue
    # -------------------------------------------------------------------------
    def generate_task4_mixed_exception(
        self,
        batch_size: int,
        length: int,
        *,
        window: int | None = None,
        exception_prob: float = 0.5,
    ) -> CurriculumBatch:
        B, L = batch_size, length
        win = window or self.default_window
        x = torch.zeros(B, L, self.d_in, device=self.device)
        y = torch.zeros(B, L, self.d_out, device=self.device)
        mask = torch.zeros(B, L, device=self.device)

        is_exception = torch.rand(B, generator=self.generator, device=self.device) < exception_prob

        W = torch.randn(B, self.d_feat, self.d_out, generator=self.generator, device=self.device) / math.sqrt(self.d_feat)
        b = torch.randn(B, 1, self.d_out, generator=self.generator, device=self.device) * 0.2

        # Context features
        x_ctx = torch.randn(B, L - 1, self.d_feat, generator=self.generator, device=self.device)
        noise = torch.randn(B, L - 1, self.d_out, generator=self.generator, device=self.device) * 0.15
        y_ctx = torch.bmm(x_ctx, W) + b + noise

        # Insert an episodic exception item into the recent window (e.g. at L-3)
        ex_pos = L - 3
        delta = torch.randn(B, self.d_out, generator=self.generator, device=self.device)
        delta = F.normalize(delta, dim=-1) * 2.5  # Large exception offset
        y_exception = y_ctx[:, ex_pos] + delta
        y_ctx[:, ex_pos] = y_exception

        x[:, :-1, : self.d_feat] = x_ctx
        x[:, :-1, self.d_feat : self.d_feat + self.d_out] = y_ctx

        for i in range(B):
            if is_exception[i]:
                # Query is the exception key, observable cue = 1
                x[i, -1, : self.d_feat] = x_ctx[i, ex_pos]
                x[i, -1, self.d_in - 2] = 1.0  # is_query
                x[i, -1, self.d_in - 1] = 1.0  # observable cue = 1.0
                y[i, -1] = y_exception[i]  # Target is the exact memorized exception!
            else:
                # Query is a new latent query, observable cue = 0
                q = torch.randn(1, self.d_feat, generator=self.generator, device=self.device)
                x[i, -1, : self.d_feat] = q
                x[i, -1, self.d_in - 2] = 1.0  # is_query
                x[i, -1, self.d_in - 1] = 0.0  # observable cue = 0.0
                y[i, -1] = torch.mm(q, W[i]) + b[i]  # Target is denoised latent trend
            mask[i, -1] = 1.0

        return CurriculumBatch(
            x=x,
            y=y,
            mask=mask,
            family_id=4,
            family_name="mixed_latent_and_exception",
            metadata={"is_exception": is_exception, "observable_cue": is_exception.float()},
        )

    # -------------------------------------------------------------------------
    # Task Family 5: Selective copy and local token shift
    # -------------------------------------------------------------------------
    def generate_task5_selective_copy(
        self,
        batch_size: int,
        length: int,
        *,
        window: int | None = None,
    ) -> CurriculumBatch:
        B, L = batch_size, length
        win = window or self.default_window
        x = torch.zeros(B, L, self.d_in, device=self.device)
        y = torch.zeros(B, L, self.d_out, device=self.device)
        mask = torch.zeros(B, L, device=self.device)

        # Intersperse signal and distractor tokens in the recent window
        for i in range(B):
            # Target signal item placed at L-4
            sig_pos = L - 4
            sig_val = torch.randn(self.d_out, generator=self.generator, device=self.device)
            sig_key = torch.randn(self.d_feat, generator=self.generator, device=self.device)
            sig_key = F.normalize(sig_key, dim=-1)

            x[i, sig_pos, : self.d_feat] = sig_key
            x[i, sig_pos, self.d_feat : self.d_feat + self.d_out] = sig_val
            x[i, sig_pos, self.d_feat - 1] = 1.0  # Signal indicator bit

            # Distractors around it have indicator bit 0
            for pos in range(L - 1):
                if pos != sig_pos:
                    x[i, pos, : self.d_feat] = torch.randn(self.d_feat, generator=self.generator, device=self.device) * 0.1
                    x[i, pos, self.d_feat - 1] = 0.0

            # Query asks for the signal token
            x[i, -1, : self.d_feat] = sig_key
            x[i, -1, self.d_in - 2] = 1.0  # is_query
            x[i, -1, self.d_in - 1] = 0.0
            y[i, -1] = sig_val
            mask[i, -1] = 1.0

        return CurriculumBatch(
            x=x,
            y=y,
            mask=mask,
            family_id=5,
            family_name="selective_copy_and_shift",
            metadata={},
        )

    # -------------------------------------------------------------------------
    # Task Family 6: Cache-boundary recall at ages w-1, w, w+1, w+2
    # -------------------------------------------------------------------------
    def generate_task6_cache_boundary(
        self,
        batch_size: int,
        length: int,
        *,
        age: int = 12,
        window: int | None = None,
    ) -> CurriculumBatch:
        B, L = batch_size, length
        win = window or self.default_window
        x = torch.zeros(B, L, self.d_in, device=self.device)
        y = torch.zeros(B, L, self.d_out, device=self.device)
        mask = torch.zeros(B, L, device=self.device)

        # Place the probed item at exact age `age` relative to query at L-1
        item_pos = L - 1 - age
        assert 0 <= item_pos < L - 1, f"Invalid age {age} for length {L}"

        for i in range(B):
            k = torch.randn(self.d_feat, generator=self.generator, device=self.device)
            k = F.normalize(k, dim=-1)
            v = torch.randn(self.d_out, generator=self.generator, device=self.device)

            x[i, item_pos, : self.d_feat] = k
            x[i, item_pos, self.d_feat : self.d_feat + self.d_out] = v

            # Fill other positions with low-level background noise
            for pos in range(L - 1):
                if pos != item_pos:
                    x[i, pos, : self.d_feat] = torch.randn(self.d_feat, generator=self.generator, device=self.device) * 0.1

            # Query at L-1
            x[i, -1, : self.d_feat] = k
            x[i, -1, self.d_in - 2] = 1.0
            x[i, -1, self.d_in - 1] = 0.0
            y[i, -1] = v
            mask[i, -1] = 1.0

        return CurriculumBatch(
            x=x,
            y=y,
            mask=mask,
            family_id=6,
            family_name="cache_boundary_recall",
            metadata={"age": age, "window": win},
        )

    # -------------------------------------------------------------------------
    # Task Family 7: Over-capacity, conflicting-write, and no-relevant-context negatives
    # -------------------------------------------------------------------------
    def generate_task7_negatives(
        self,
        batch_size: int,
        length: int,
        *,
        mode: Literal["over_capacity", "conflicting_write", "no_context"] | None = None,
        window: int | None = None,
    ) -> CurriculumBatch:
        B, L = batch_size, length
        win = window or self.default_window
        x = torch.zeros(B, L, self.d_in, device=self.device)
        y = torch.zeros(B, L, self.d_out, device=self.device)
        mask = torch.zeros(B, L, device=self.device)

        modes = ["over_capacity", "conflicting_write", "no_context"]

        for i in range(B):
            sample_mode = mode or modes[i % 3]

            if sample_mode == "over_capacity":
                # Write 2 * d_feat items
                num_items = min(20, L - 2)
                keys = torch.randn(num_items, self.d_feat, generator=self.generator, device=self.device)
                vals = torch.randn(num_items, self.d_out, generator=self.generator, device=self.device)
                for j in range(num_items):
                    x[i, j, : self.d_feat] = keys[j]
                    x[i, j, self.d_feat : self.d_feat + self.d_out] = vals[j]
                # Query random item
                chosen = torch.randint(0, num_items, (1,), generator=self.generator, device=self.device).item()
                x[i, -1, : self.d_feat] = keys[chosen]
                x[i, -1, self.d_in - 2] = 1.0
                y[i, -1] = vals[chosen]

            elif sample_mode == "conflicting_write":
                # Write key K with value V1 early, then overwrite K with value V2 late
                k = F.normalize(torch.randn(self.d_feat, generator=self.generator, device=self.device), dim=-1)
                v1 = torch.randn(self.d_out, generator=self.generator, device=self.device)
                v2 = torch.randn(self.d_out, generator=self.generator, device=self.device)

                pos1 = 2
                pos2 = L - 3
                x[i, pos1, : self.d_feat] = k
                x[i, pos1, self.d_feat : self.d_feat + self.d_out] = v1

                x[i, pos2, : self.d_feat] = k
                x[i, pos2, self.d_feat : self.d_feat + self.d_out] = v2

                # Query key K; target is latest write V2
                x[i, -1, : self.d_feat] = k
                x[i, -1, self.d_in - 2] = 1.0
                y[i, -1] = v2

            else:  # no_context
                # Sequence contains random unrelated keys
                for j in range(L - 1):
                    x[i, j, : self.d_feat] = torch.randn(self.d_feat, generator=self.generator, device=self.device)
                    x[i, j, self.d_feat : self.d_feat + self.d_out] = torch.randn(self.d_out, generator=self.generator, device=self.device)

                # Query is an orthogonal key; target is 0 (neutral default)
                q = torch.randn(self.d_feat, generator=self.generator, device=self.device)
                x[i, -1, : self.d_feat] = q
                x[i, -1, self.d_in - 2] = 1.0
                y[i, -1] = torch.zeros(self.d_out, device=self.device)

            mask[i, -1] = 1.0

        return CurriculumBatch(
            x=x,
            y=y,
            mask=mask,
            family_id=7,
            family_name="overcapacity_conflicting_no_context_negatives",
            metadata={"mode": mode},
        )

    # -------------------------------------------------------------------------
    # Master batch generation
    # -------------------------------------------------------------------------
    def generate_random_batch(
        self,
        batch_size: int,
        length: int = 32,
        *,
        window: int | None = None,
    ) -> CurriculumBatch:
        """Randomly sample one task family per batch or uniform mix."""
        win = window or self.default_window
        family = torch.randint(1, 8, (1,), generator=self.generator, device=self.device).item()
        if family == 1:
            return self.generate_task1_noisy_linear(batch_size, length)
        elif family == 2:
            return self.generate_task2_recent_copy(batch_size, length, window=win)
        elif family == 3:
            return self.generate_task3_remote_recall(batch_size, length, window=win)
        elif family == 4:
            return self.generate_task4_mixed_exception(batch_size, length, window=win)
        elif family == 5:
            return self.generate_task5_selective_copy(batch_size, length, window=win)
        elif family == 6:
            # Random age among boundary ages
            age = win + torch.randint(-1, 3, (1,), generator=self.generator, device=self.device).item()
            return self.generate_task6_cache_boundary(batch_size, length, age=age, window=win)
        else:
            return self.generate_task7_negatives(batch_size, length, window=win)

    def generate_balanced_batch(
        self,
        batch_size_per_task: int = 8,
        length: int = 32,
        *,
        window: int | None = None,
    ) -> CurriculumBatch:
        """Generate an equal mixture across all 7 task families."""
        win = window or self.default_window
        batches = [
            self.generate_task1_noisy_linear(batch_size_per_task, length),
            self.generate_task2_recent_copy(batch_size_per_task, length, window=win),
            self.generate_task3_remote_recall(batch_size_per_task, length, window=win),
            self.generate_task4_mixed_exception(batch_size_per_task, length, window=win),
            self.generate_task5_selective_copy(batch_size_per_task, length, window=win),
            self.generate_task6_cache_boundary(batch_size_per_task, length, age=win, window=win),
            self.generate_task7_negatives(batch_size_per_task, length, window=win),
        ]
        x = torch.cat([b.x for b in batches], dim=0)
        y = torch.cat([b.y for b in batches], dim=0)
        mask = torch.cat([b.mask for b in batches], dim=0)
        return CurriculumBatch(
            x=x,
            y=y,
            mask=mask,
            family_id=0,
            family_name="balanced_curriculum_mixture",
            metadata={"batch_size_per_task": batch_size_per_task, "batches": batches},
        )
