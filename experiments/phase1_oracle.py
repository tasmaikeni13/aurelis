#!/usr/bin/env python3
"""Pinned Phase 1 mathematical oracle, calibration, and pathology gate.

The experiment deliberately mixes independent full-history reconstruction,
the production streaming path, direct deterministic certificates, conditional
Monte Carlo, dtype/conditioning probes, and measured state/profile metadata.
Every tolerance is loaded from the immutable config before any random row is
generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from aurelis import (
    aurelis_read,
    consume,
    historical_oracle,
    initial_state,
    occurrence_partition,
    read,
)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "phase1_oracle.json"
RESULTS = REPO / "results" / "phase1"
RAW = RESULTS / "raw"
PLOTS = REPO / "plots" / "phase1"
OUTPUT_FIELDS = ("remote", "full_residual", "bayes", "episodic")
DIAGNOSTIC_FIELDS = ("h", "V_R", "V_H", "K_RH", "g_raw", "g_B", "g_E")


def git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, text=True, capture_output=True, check=False
    )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_value(value: Any) -> Any:
    """Strict-JSON conversion that keeps nonfinite observations visible."""

    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "+inf" if value > 0 else "-inf"
    if isinstance(value, Path):
        return str(value)
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_value(row), sort_keys=True, allow_nan=False) + "\n")


def unit_rows(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.linalg.vector_norm(tensor, dim=-1, keepdim=True).clamp_min(1e-30)


def make_keys(
    kind: str, count: int, d_key: int, generator: torch.Generator
) -> torch.Tensor:
    if count == 0:
        return torch.empty(0, d_key, dtype=torch.float64)
    random = torch.randn(count, d_key, generator=generator, dtype=torch.float64)
    if kind == "random":
        return unit_rows(random)
    if kind == "orthogonal":
        blocks = []
        remaining = count
        while remaining:
            q, _ = torch.linalg.qr(
                torch.randn(d_key, d_key, generator=generator, dtype=torch.float64)
            )
            take = min(remaining, d_key)
            blocks.append(q[:take])
            remaining -= take
        return torch.cat(blocks)
    base = unit_rows(torch.randn(1, d_key, generator=generator, dtype=torch.float64))[0]
    if kind == "correlated":
        return unit_rows(base + 5e-2 * random)
    if kind == "duplicate":
        return base.expand(count, d_key).clone()
    if kind == "near_collision":
        return unit_rows(base + 1e-8 * random)
    if kind == "rank_deficient":
        rank = max(1, d_key // 3)
        left = torch.randn(count, rank, generator=generator, dtype=torch.float64)
        right = torch.randn(rank, d_key, generator=generator, dtype=torch.float64)
        return unit_rows(left @ right)
    if kind == "zero":
        return torch.zeros(count, d_key, dtype=torch.float64)
    if kind == "small_norm":
        return 1e-12 * unit_rows(random)
    raise ValueError(f"unknown key pathology {kind}")


def max_output_error(actual: Any, expected: Any) -> float:
    errors: list[float] = []
    for field in OUTPUT_FIELDS:
        errors.append(
            float(
                torch.max(torch.abs(getattr(actual, field) - getattr(expected, field)))
                .detach()
                .cpu()
            )
        )
    for field in DIAGNOSTIC_FIELDS:
        left = getattr(actual.diagnostics, field)
        right = getattr(expected.diagnostics, field)
        finite = torch.isfinite(left) & torch.isfinite(right)
        if bool(finite.any()):
            errors.append(
                float(torch.max(torch.abs(left[finite] - right[finite])).detach().cpu())
            )
        if not torch.equal(torch.isfinite(left), torch.isfinite(right)):
            return math.inf
    return max(errors, default=0.0)


def exact_tolerance(config: dict[str, Any], kappa: float, scale: float, n: int = 1) -> float:
    tolerance = config["preregistered_tolerances"]
    return float(
        tolerance["fp64_identity_absolute_floor"]
        + tolerance["fp64_identity_epsilon_multiplier"]
        * np.finfo(np.float64).eps
        * max(1.0, kappa)
        * max(1.0, scale)
        * max(1, n)
    )


def stream_tolerance(config: dict[str, Any], kappa: float, scale: float, n: int) -> float:
    tolerance = config["preregistered_tolerances"]
    return float(
        tolerance["streaming_oracle_absolute_floor"]
        + tolerance["streaming_oracle_epsilon_multiplier"]
        * np.finfo(np.float64).eps
        * max(1.0, kappa)
        * max(1.0, scale)
        * max(1, n)
    )


def run_stream_case(
    *,
    config: dict[str, Any],
    generator: torch.Generator,
    case: str,
    d_key: int,
    d_value: int,
    window: int,
    n_remote: int,
    pathology: str = "random",
) -> dict[str, Any]:
    length = n_remote + window
    keys_2d = make_keys(pathology, length, d_key, generator)
    operator = torch.randn(d_value, d_key, generator=generator, dtype=torch.float64) / math.sqrt(d_key)
    values_2d = keys_2d @ operator.mT + 0.03 * torch.randn(
        length, d_value, generator=generator, dtype=torch.float64
    )
    evidence_1d = torch.logspace(-2, 2, max(length, 1), dtype=torch.float64)[:length]
    queries_2d = unit_rows(
        torch.randn(length, d_key, generator=generator, dtype=torch.float64)
    )
    keys = keys_2d.view(1, 1, length, d_key)
    values = values_2d.view(1, 1, length, d_value)
    evidence = evidence_1d.view(1, 1, length)
    prior = 0.3
    state = initial_state(1, 1, d_key, d_value, window, prior=prior)
    maximum_error = 0.0
    maximum_ratio = 0.0
    partition_failures = 0
    invalid_rows = 0
    for prefix in range(1, length + 1):
        state = consume(
            state,
            keys[:, :, prefix - 1],
            values[:, :, prefix - 1],
            evidence[:, :, prefix - 1],
            occurrence_id=prefix - 1,
        )
        query = queries_2d[prefix - 1].view(1, 1, d_key)
        actual = read(state, query, temperature=2.0)
        expected = historical_oracle(
            keys[:, :, :prefix],
            values[:, :, :prefix],
            evidence[:, :, :prefix],
            query,
            window=window,
            prior=prior,
            temperature=2.0,
        )
        error = max_output_error(actual, expected)
        kappa = float(torch.linalg.cond(state.precision)[0, 0])
        scale = max(
            1.0,
            float(torch.max(torch.abs(state.precision))),
            float(torch.max(torch.abs(state.cross))) if state.cross.numel() else 1.0,
        )
        valid = math.isfinite(kappa) and kappa * np.finfo(np.float64).eps <= config[
            "preregistered_tolerances"
        ]["maximum_valid_kappa_times_epsilon"]
        tolerance = stream_tolerance(config, kappa, scale, prefix)
        if valid:
            maximum_ratio = max(maximum_ratio, error / tolerance)
        else:
            invalid_rows += 1
        maximum_error = max(maximum_error, error)
        remote, recent = occurrence_partition(state)
        if remote + recent != tuple(range(prefix)) or set(remote).intersection(recent):
            partition_failures += 1
    return {
        "suite": "streaming_prefix",
        "case": case,
        "d_key": d_key,
        "d_value": d_value,
        "window": window,
        "n_remote_final": n_remote,
        "length": length,
        "pathology": pathology,
        "prefixes_checked": length,
        "maximum_absolute_error": maximum_error,
        "maximum_tolerance_ratio_valid_rows": maximum_ratio,
        "invalid_conditioning_rows": invalid_rows,
        "partition_failures": partition_failures,
        "status": "PASS" if maximum_ratio <= 1.0 and partition_failures == 0 else "FAIL",
    }


def streaming_sweep(config: dict[str, Any], generator: torch.Generator) -> list[dict[str, Any]]:
    sweep = config["sweeps"]
    rows: list[dict[str, Any]] = []
    for d_key in sweep["d_key"]:
        for d_value in sweep["d_value"]:
            for window in sweep["window"]:
                rows.append(
                    run_stream_case(
                        config=config,
                        generator=generator,
                        case="shape",
                        d_key=d_key,
                        d_value=d_value,
                        window=window,
                        n_remote=3,
                    )
                )
    for d_key in sweep["d_key"]:
        loads = {
            "below": max(1, d_key // 2),
            "at": d_key,
            "above": 2 * d_key,
        }
        for label, load in loads.items():
            rows.append(
                run_stream_case(
                    config=config,
                    generator=generator,
                    case=f"remote_load_{label}",
                    d_key=d_key,
                    d_value=3,
                    window=2,
                    n_remote=load,
                )
            )
    return rows


def deterministic_sweep(config: dict[str, Any], generator: torch.Generator) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tolerances = config["preregistered_tolerances"]
    pathologies = config["sweeps"]["key_pathologies"]
    priors = config["sweeps"]["prior_precision"]
    for pathology_index, pathology in enumerate(pathologies):
        for prior_index, prior in enumerate(priors):
            d_key, d_value, n_remote, window = 8, 3, 16, 8
            remote_keys = make_keys(pathology, n_remote, d_key, generator)
            local_keys = make_keys(pathology, window, d_key, generator)
            truth = torch.randn(d_value, d_key, generator=generator, dtype=torch.float64) / math.sqrt(d_key)
            evidence = torch.logspace(-4, 4, n_remote, dtype=torch.float64)
            local_evidence = (
                torch.ones(window, dtype=torch.float64)
                if (pathology_index + prior_index) % 2 == 0
                else torch.logspace(-4, 4, window, dtype=torch.float64)
            )
            residuals = 0.02 * torch.randn(window, d_value, generator=generator, dtype=torch.float64)
            remote_values = remote_keys @ truth.mT
            local_values = local_keys @ truth.mT + residuals
            precision_2d = prior * torch.eye(d_key, dtype=torch.float64) + torch.einsum(
                "n,ni,nj->ij", evidence, remote_keys, remote_keys
            )
            cross_2d = torch.einsum("n,nv,nd->vd", evidence, remote_values, remote_keys)
            query_1d = unit_rows(torch.randn(1, d_key, generator=generator, dtype=torch.float64))[0]
            temperature = config["sweeps"]["attention_temperature"][(pathology_index + prior_index) % 8]
            output = aurelis_read(
                precision_2d.view(1, 1, d_key, d_key),
                cross_2d.view(1, 1, d_value, d_key),
                local_keys.view(1, 1, window, d_key),
                local_values.view(1, 1, window, d_value),
                local_evidence.view(1, 1, window),
                query_1d.view(1, 1, d_key),
                temperature=temperature,
            )
            diagnostics = output.diagnostics
            weights = diagnostics.attention[0, 0]
            kbar = diagnostics.kbar[0, 0]
            M = torch.linalg.solve(precision_2d, cross_2d.mT).mT
            delta_bar = weights @ residuals
            residual_identity_rhs = delta_bar + (M - truth) @ (query_1d - kbar)
            residual_identity_error = float(
                torch.max(torch.abs(output.full_residual[0, 0] - truth @ query_1d - residual_identity_rhs))
            )
            gate = (-0.25, 0.0, 0.37, 1.0, 1.25)[(pathology_index + prior_index) % 5]
            vbar = diagnostics.vbar[0, 0]
            gated = M @ query_1d + gate * (vbar - M @ kbar)
            general_rhs = (M - truth) @ query_1d + gate * (
                delta_bar - (M - truth) @ kbar
            )
            general_error = float(torch.max(torch.abs(gated - truth @ query_1d - general_rhs)))
            reproduced = truth @ query_1d + (weights @ (local_keys @ truth.mT) - truth @ kbar)
            reproduction_error = float(torch.max(torch.abs(reproduced - truth @ query_1d)))
            V_R = float(diagnostics.V_R)
            V_H = float(diagnostics.V_H)
            K_RH = float(diagnostics.K_RH)
            denominator = V_R + V_H - 2.0 * K_RH
            gate_form_covariance = (V_R - K_RH) / denominator
            gate_form_reduced = float(
                torch.dot(query_1d, torch.linalg.solve(precision_2d, kbar))
                / (
                    float(diagnostics.h)
                    + torch.dot(kbar, torch.linalg.solve(precision_2d, kbar))
                )
            )
            grid = torch.linspace(0.0, 1.0, 10001, dtype=torch.float64)
            grid_variance = (
                (1 - grid).square() * V_R
                + grid.square() * V_H
                + 2 * grid * (1 - grid) * K_RH
            )
            g_B = float(diagnostics.g_B)
            routed_variance = (
                (1 - g_B) ** 2 * V_R + g_B**2 * V_H + 2 * g_B * (1 - g_B) * K_RH
            )
            kappa = float(torch.linalg.cond(precision_2d))
            scale = max(1.0, float(torch.max(torch.abs(precision_2d))), float(torch.max(torch.abs(cross_2d))))
            valid = math.isfinite(kappa) and kappa * np.finfo(np.float64).eps <= tolerances[
                "maximum_valid_kappa_times_epsilon"
            ]
            identity_tolerance = exact_tolerance(config, kappa, scale, n_remote)
            tiny = np.finfo(np.float64).tiny
            denominator_condition = (
                abs(V_R) + abs(V_H) + 2.0 * abs(K_RH)
            ) / max(abs(denominator), tiny)
            numerator_result = abs(V_R - K_RH)
            numerator_condition = (
                (abs(V_R) + abs(K_RH)) / numerator_result
                if numerator_result > 0
                else math.inf
            )
            gate_reduction_condition = max(
                kappa, denominator_condition, numerator_condition
            )
            valid_gate_reduction = (
                math.isfinite(gate_reduction_condition)
                and gate_reduction_condition * np.finfo(np.float64).eps
                <= tolerances["maximum_valid_kappa_times_epsilon"]
            )
            gate_tolerance = exact_tolerance(
                config,
                gate_reduction_condition,
                max(1.0, abs(gate_form_covariance), abs(gate_form_reduced)),
            )
            variance_tolerance = exact_tolerance(
                config, kappa, max(1.0, abs(V_R), abs(V_H), abs(K_RH))
            )

            gram = torch.einsum("n,ni,nj->ij", evidence, remote_keys, remote_keys)
            lambda_min = float(torch.linalg.eigvalsh(gram).amin())
            ridge_domain = lambda_min > 100 * np.finfo(np.float64).eps * max(1.0, float(torch.linalg.norm(gram, 2)))
            ridge_local_values = local_keys @ truth.mT
            ridge_output = aurelis_read(
                precision_2d.view(1, 1, d_key, d_key),
                cross_2d.view(1, 1, d_value, d_key),
                local_keys.view(1, 1, window, d_key),
                ridge_local_values.view(1, 1, window, d_value),
                local_evidence.view(1, 1, window),
                query_1d.view(1, 1, d_key),
                temperature=temperature,
            )
            ridge_error = float(torch.linalg.vector_norm(ridge_output.full_residual[0, 0] - truth @ query_1d))
            ridge_r = query_1d - ridge_output.diagnostics.kbar[0, 0]
            ridge_bound = (
                prior
                * float(torch.linalg.matrix_norm(truth, ord=2))
                / (lambda_min + prior)
                * float(torch.linalg.vector_norm(ridge_r))
                if ridge_domain
                else math.nan
            )
            gate_regret = routed_variance - float(grid_variance.min())
            noninferiority = routed_variance - min(V_R, V_H)
            passed = (
                (not valid or max(residual_identity_error, general_error, reproduction_error) <= identity_tolerance)
                and (
                    not valid_gate_reduction
                    or abs(gate_form_covariance - gate_form_reduced) <= gate_tolerance
                )
                and gate_regret
                <= max(tolerances["dense_gate_variance_slack"], variance_tolerance)
                and noninferiority
                <= max(tolerances["dense_gate_variance_slack"], variance_tolerance)
                and (not ridge_domain or ridge_error <= ridge_bound + tolerances["finite_ridge_bound_slack"])
            )
            rows.append(
                {
                    "suite": "deterministic_identity",
                    "pathology": pathology,
                    "prior": prior,
                    "temperature": temperature,
                    "evidence_model": "homoscedastic" if torch.all(local_evidence == 1) else "heteroscedastic",
                    "condition_number": kappa,
                    "valid_conditioned_identity_domain": valid,
                    "identity_tolerance": identity_tolerance,
                    "residual_identity_error": residual_identity_error,
                    "general_gate_identity_error": general_error,
                    "linear_reproduction_error": reproduction_error,
                    "gate_form_equivalence_error": abs(gate_form_covariance - gate_form_reduced),
                    "gate_reduction_condition_number": gate_reduction_condition,
                    "valid_gate_covariance_reduction_domain": valid_gate_reduction,
                    "gate_formula_tolerance": gate_tolerance,
                    "g_raw": gate_form_reduced,
                    "g_B": g_B,
                    "dense_grid_regret": gate_regret,
                    "endpoint_noninferiority_slack": noninferiority,
                    "variance_comparison_tolerance": variance_tolerance,
                    "ridge_domain_positive_definite_gram": ridge_domain,
                    "ridge_lambda_min": lambda_min,
                    "ridge_error": ridge_error,
                    "ridge_bound": ridge_bound,
                    "status": "PASS" if passed else "FAIL",
                }
            )
    return rows


def clipping_and_recall(config: dict[str, Any], generator: torch.Generator) -> dict[str, Any]:
    clipping_rows = []
    for label, query_value in (("below_zero", -1.0), ("interior", 0.5), ("above_one", 2.0)):
        precision = torch.ones(1, 1, 1, 1, dtype=torch.float64)
        cross = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
        keys = torch.ones(1, 1, 1, 1, dtype=torch.float64)
        values = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
        evidence = torch.full((1, 1, 1), 10.0, dtype=torch.float64)
        query = torch.tensor([[[query_value]]], dtype=torch.float64)
        output = aurelis_read(precision, cross, keys, values, evidence, query)
        raw = float(output.diagnostics.g_raw)
        expected_class = "below_zero" if raw < 0 else "above_one" if raw > 1 else "interior"
        clipping_rows.append(
            {
                "requested_class": label,
                "observed_class": expected_class,
                "g_raw": raw,
                "g_B": float(output.diagnostics.g_B),
                "status": "PASS" if expected_class == label else "FAIL",
            }
        )

    d_key, d_value, window = 8, 3, 8
    keys = unit_rows(torch.randn(window, d_key, generator=generator, dtype=torch.float64))
    query = keys[0].clone()
    truth = torch.randn(d_value, d_key, generator=generator, dtype=torch.float64)
    values = keys @ truth.mT
    exception = unit_rows(torch.randn(1, d_value, generator=generator, dtype=torch.float64))[0]
    values[0] += 2.0 * exception
    precision = torch.eye(d_key, dtype=torch.float64).view(1, 1, d_key, d_key)
    cross = truth.view(1, 1, d_value, d_key)
    convergence_rows = []
    for temperature in config["sweeps"]["attention_temperature"]:
        output = aurelis_read(
            precision,
            cross,
            keys.view(1, 1, window, d_key),
            values.view(1, 1, window, d_value),
            torch.ones(1, 1, window, dtype=torch.float64),
            query.view(1, 1, d_key),
            temperature=temperature,
            episodic_responsibility=1.0,
        )
        convergence_rows.append(
            {
                "temperature": temperature,
                "selected_mass": float(output.diagnostics.attention[0, 0, 0]),
                "numerically_saturated": bool(output.diagnostics.attention[0, 0, 0] == 1),
                "full_residual_copy_error": float(torch.linalg.vector_norm(output.full_residual[0, 0] - values[0])),
                "episodic_copy_error": float(torch.linalg.vector_norm(output.episodic[0, 0] - values[0])),
            }
        )

    one_key = aurelis_read(
        precision,
        cross,
        query.view(1, 1, 1, d_key),
        values[0].view(1, 1, 1, d_value),
        torch.ones(1, 1, 1, dtype=torch.float64),
        query.view(1, 1, d_key),
        episodic_responsibility=1.0,
    )
    exact_error = float(torch.linalg.vector_norm(one_key.full_residual[0, 0] - values[0]))
    return {
        "clipping": clipping_rows,
        "finite_temperature_convergence": convergence_rows,
        "hard_one_hot_realized_by_singleton_cache": {
            "error": exact_error,
            "status": "PASS"
            if exact_error <= config["preregistered_tolerances"]["exact_one_hot_absolute"]
            else "FAIL",
        },
    }


@dataclass
class OnlineMoments:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0

    def add(self, values: np.ndarray) -> None:
        flattened = np.asarray(values, dtype=np.float64).ravel()
        self.count += flattened.size
        self.total += float(flattened.sum(dtype=np.float64))
        self.total_square += float(np.square(flattened).sum(dtype=np.float64))

    @property
    def mean(self) -> float:
        return self.total / self.count

    @property
    def standard_error(self) -> float:
        if self.count < 2:
            return math.inf
        variance = max(0.0, (self.total_square - self.total**2 / self.count) / (self.count - 1))
        return math.sqrt(variance / self.count)


def conditional_case(
    *,
    config: dict[str, Any],
    rng: np.random.Generator,
    name: str,
    precision: np.ndarray,
    cache_keys: np.ndarray,
    evidence: np.ndarray,
    query: np.ndarray,
    temperature: float,
) -> dict[str, Any]:
    trials = int(config["monte_carlo_trials"])
    d_key = query.size
    d_value = 3
    mean = np.zeros((d_value, d_key), dtype=np.float64)
    pilot_values = cache_keys @ mean.T
    output = aurelis_read(
        torch.from_numpy(precision).view(1, 1, d_key, d_key),
        torch.from_numpy(mean @ precision).view(1, 1, d_value, d_key),
        torch.from_numpy(cache_keys).view(1, 1, len(cache_keys), d_key),
        torch.from_numpy(pilot_values).view(1, 1, len(cache_keys), d_value),
        torch.from_numpy(evidence).view(1, 1, len(cache_keys)),
        torch.from_numpy(query).view(1, 1, d_key),
        temperature=temperature,
    )
    weights = output.diagnostics.attention[0, 0].numpy()
    kbar = output.diagnostics.kbar[0, 0].numpy()
    V_R = float(output.diagnostics.V_R)
    V_H = float(output.diagnostics.V_H)
    K_RH = float(output.diagnostics.K_RH)
    gate = float(output.diagnostics.g_B)
    g_independent = V_R / (V_R + V_H)
    V_gate = (1 - gate) ** 2 * V_R + gate**2 * V_H + 2 * gate * (1 - gate) * K_RH
    V_independent = (
        (1 - g_independent) ** 2 * V_R
        + g_independent**2 * V_H
        + 2 * g_independent * (1 - g_independent) * K_RH
    )
    predicted = {
        "V_R": V_R,
        "V_H": V_H,
        "K_RH": K_RH,
        "V_g": V_gate,
        "V_independence_router": V_independent,
    }
    moments = {key: OnlineMoments() for key in predicted}
    differences = {"gate_minus_remote": OnlineMoments(), "gate_minus_residual": OnlineMoments(), "independence_minus_gate": OnlineMoments()}
    posterior_chol = np.linalg.cholesky(np.linalg.inv(precision))
    chunk = 2500
    generated = 0
    while generated < trials:
        take = min(chunk, trials - generated)
        delta = rng.normal(size=(take, d_value, d_key)) @ posterior_chol.T
        local_noise = rng.normal(size=(take, d_value, len(cache_keys))) / np.sqrt(evidence)[None, None, :]
        weighted_noise = np.einsum("w,tvw->tv", weights, local_noise)
        e_remote = -np.einsum("tvd,d->tv", delta, query)
        e_residual = weighted_noise - np.einsum("tvd,d->tv", delta, query - kbar)
        e_gate = (1 - gate) * e_remote + gate * e_residual
        e_independent = (1 - g_independent) * e_remote + g_independent * e_residual
        values = {
            "V_R": e_remote**2,
            "V_H": e_residual**2,
            "K_RH": e_remote * e_residual,
            "V_g": e_gate**2,
            "V_independence_router": e_independent**2,
        }
        for key, sample in values.items():
            moments[key].add(sample)
        differences["gate_minus_remote"].add(values["V_g"] - values["V_R"])
        differences["gate_minus_residual"].add(values["V_g"] - values["V_H"])
        differences["independence_minus_gate"].add(values["V_independence_router"] - values["V_g"])
        generated += take

    z = float(config["normal_critical_value"])
    floor = float(config["preregistered_tolerances"]["monte_carlo_numerical_floor"])
    estimates = {}
    all_inside = True
    for key, expected in predicted.items():
        estimate = moments[key].mean
        half_width = z * moments[key].standard_error + floor
        inside = abs(estimate - expected) <= half_width
        all_inside &= inside
        estimates[key] = {
            "predicted": expected,
            "empirical": estimate,
            "standard_error": moments[key].standard_error,
            "interval_half_width_99_percent": half_width,
            "inside_interval": inside,
        }
    paired = {}
    endpoint_noninferior = True
    for key, moment in differences.items():
        upper = moment.mean + z * moment.standard_error + floor
        paired[key] = {"mean": moment.mean, "standard_error": moment.standard_error, "upper_99_percent": upper}
        if key.startswith("gate_minus"):
            endpoint_noninferior &= upper <= floor * 2
    omission = differences["independence_minus_gate"]
    omission_z = omission.mean / max(omission.standard_error, np.finfo(float).tiny)
    return {
        "name": name,
        "trials": trials,
        "evidence_min": float(evidence.min()),
        "evidence_max": float(evidence.max()),
        "g_raw": float(output.diagnostics.g_raw),
        "g_B": gate,
        "g_independence": g_independent,
        "estimates": estimates,
        "paired_risk_differences": paired,
        "covariance_omission_z_score": omission_z,
        "all_formula_estimates_inside_preregistered_99_percent_intervals": all_inside,
        "routed_endpoint_noninferiority_with_mc_error": endpoint_noninferior,
        "status": "PASS" if all_inside and endpoint_noninferior else "FAIL",
    }


def monte_carlo_sweep(config: dict[str, Any], rng: np.random.Generator) -> list[dict[str, Any]]:
    d_key, window = 6, 6
    base_precision = np.diag(np.geomspace(0.5, 4.0, d_key)).astype(np.float64)
    random_keys = rng.normal(size=(window, d_key))
    random_keys /= np.linalg.norm(random_keys, axis=1, keepdims=True)
    cases = [
        dict(
            name="homoscedastic_diffuse",
            precision=base_precision,
            cache_keys=random_keys,
            evidence=np.ones(window),
            query=rng.normal(size=d_key),
            temperature=0.0,
        ),
        dict(
            name="heteroscedastic_four_orders",
            precision=base_precision,
            cache_keys=random_keys,
            evidence=np.geomspace(1e-2, 1e2, window),
            query=rng.normal(size=d_key),
            temperature=4.0,
        ),
        dict(
            name="clipped_below_zero",
            precision=np.eye(d_key),
            cache_keys=np.tile(-np.eye(d_key)[0], (window, 1)),
            evidence=np.full(window, 60.0),
            query=np.eye(d_key)[0],
            temperature=1.0,
        ),
        dict(
            name="clipped_above_one",
            precision=np.eye(d_key),
            cache_keys=np.tile(0.4 * np.eye(d_key)[0], (window, 1)),
            evidence=np.full(window, 1e4),
            query=np.eye(d_key)[0],
            temperature=1.0,
        ),
        dict(
            name="constructed_covariance_omission",
            precision=np.eye(d_key),
            cache_keys=np.tile(1.5 * np.eye(d_key)[0], (window, 1)),
            evidence=np.full(window, 10.0 / window),
            query=np.eye(d_key)[0],
            temperature=1.0,
        ),
    ]
    return [conditional_case(config=config, rng=rng, **case) for case in cases]


def target_conflict_and_misspecification(config: dict[str, Any]) -> dict[str, Any]:
    precision = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    truth = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    cross = truth.clone()
    key = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    query = torch.ones(1, 1, 1, dtype=torch.float64)
    latent = torch.tensor([[[2.0]]], dtype=torch.float64)
    observed = torch.tensor([[[[7.0]]]], dtype=torch.float64)
    evidence = torch.full((1, 1, 1), 100.0, dtype=torch.float64)
    output = aurelis_read(
        precision,
        cross,
        key,
        observed,
        evidence,
        query,
        episodic_responsibility=1.0,
    )
    latent_bayes = float(torch.mean((output.bayes - latent) ** 2))
    latent_episodic = float(torch.mean((output.episodic - latent) ** 2))
    copy_bayes = float(torch.mean((output.bayes - observed.squeeze(-2)) ** 2))
    copy_episodic = float(torch.mean((output.episodic - observed.squeeze(-2)) ** 2))
    regret = latent_bayes - float(torch.mean((output.remote - latent) ** 2))
    return {
        "g_B": float(output.diagnostics.g_B),
        "latent_target": {
            "bayes_mse": latent_bayes,
            "episodic_mse": latent_episodic,
            "preferred": "bayes",
        },
        "episodic_copy_target": {
            "bayes_mse": copy_bayes,
            "episodic_mse": copy_episodic,
            "preferred": "episodic",
        },
        "misspecified_local_exception_counterexample": {
            "remote_mse": float(torch.mean((output.remote - latent) ** 2)),
            "routed_mse": latent_bayes,
            "regret_vs_best_endpoint": regret,
            "retained": True,
        },
        "status": "PASS"
        if latent_bayes < latent_episodic
        and copy_episodic < copy_bayes
        and regret >= config["preregistered_tolerances"]["misspecification_minimum_regret"]
        else "FAIL",
    }


def handoff_boundary(config: dict[str, Any], generator: torch.Generator) -> dict[str, Any]:
    d_key, d_value, window, length = 4, 2, 2, 4
    keys = unit_rows(torch.randn(length, d_key, generator=generator, dtype=torch.float64))
    values = torch.randn(length, d_value, generator=generator, dtype=torch.float64)
    values[0] += 3.0
    evidence = torch.logspace(-1, 1, length, dtype=torch.float64)
    query_base = keys[0].clone()
    state = initial_state(1, 1, d_key, d_value, window, prior=0.5)
    boundaries = []
    for prefix in range(1, length + 1):
        state = consume(
            state,
            keys[prefix - 1].view(1, 1, d_key),
            values[prefix - 1].view(1, 1, d_value),
            evidence[prefix - 1].view(1, 1),
        )
        if prefix not in (2, 3, 4):
            continue
        streaming = read(state, query_base.view(1, 1, d_key), temperature=8.0)
        key_variable = keys[:prefix].clone().view(1, 1, prefix, d_key).requires_grad_(True)
        value_variable = values[:prefix].clone().view(1, 1, prefix, d_value).requires_grad_(True)
        query_variable = query_base.clone().view(1, 1, d_key).requires_grad_(True)
        oracle = historical_oracle(
            key_variable,
            value_variable,
            evidence[:prefix].view(1, 1, prefix),
            query_variable,
            window=window,
            prior=0.5,
            temperature=8.0,
        )
        loss = oracle.bayes.sum()
        gradient_query, gradient_keys, gradient_values = torch.autograd.grad(
            loss, (query_variable, key_variable, value_variable)
        )
        remote, recent = occurrence_partition(state)
        boundaries.append(
            {
                "position": {2: "immediately_before", 3: "at_handoff", 4: "immediately_after"}[prefix],
                "prefix": prefix,
                "remote_ids": remote,
                "recent_ids": recent,
                "tracked_occurrence_count": int(0 in remote) + int(0 in recent),
                "streaming_oracle_error": max_output_error(streaming, oracle),
                "output": oracle.bayes.detach().flatten().tolist(),
                "gradient_query": gradient_query.detach().flatten().tolist(),
                "gradient_tracked_key": gradient_keys.detach()[0, 0, 0].tolist(),
                "gradient_tracked_value": gradient_values.detach()[0, 0, 0].tolist(),
                "all_gradients_finite": bool(
                    torch.isfinite(gradient_query).all()
                    and torch.isfinite(gradient_keys).all()
                    and torch.isfinite(gradient_values).all()
                ),
            }
        )
    discontinuities = []
    for left, right in zip(boundaries, boundaries[1:]):
        discontinuities.append(
            {
                "transition": f"{left['position']}->{right['position']}",
                "output_l2_jump": float(np.linalg.norm(np.asarray(right["output"]) - np.asarray(left["output"]))),
                "query_gradient_l2_jump": float(np.linalg.norm(np.asarray(right["gradient_query"]) - np.asarray(left["gradient_query"]))),
                "tracked_value_gradient_l2_jump": float(np.linalg.norm(np.asarray(right["gradient_tracked_value"]) - np.asarray(left["gradient_tracked_value"]))),
            }
        )
    passed = all(
        row["tracked_occurrence_count"] == 1
        and row["all_gradients_finite"]
        and row["streaming_oracle_error"] <= config["preregistered_tolerances"]["streaming_oracle_absolute_floor"]
        for row in boundaries
    )
    return {"boundaries": boundaries, "discontinuities": discontinuities, "status": "PASS" if passed else "FAIL"}


def quantize(tensor: torch.Tensor, storage_dtype: torch.dtype) -> torch.Tensor:
    return tensor.to(storage_dtype).to(torch.float32 if storage_dtype == torch.bfloat16 else storage_dtype)


def dtype_conditioning_sweep(config: dict[str, Any], generator: torch.Generator) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    d_key, d_value, n_remote, window = 8, 3, 24, 8
    base = unit_rows(torch.randn(1, d_key, generator=generator, dtype=torch.float64))[0]
    native_bfloat16 = False
    native_bfloat16_reason = ""
    bfloat16_probe_device = torch.device(
        "cuda" if torch.cuda.is_available() and torch.version.hip else "cpu"
    )
    try:
        torch.linalg.cholesky(
            torch.eye(2, dtype=torch.bfloat16, device=bfloat16_probe_device)
        )
        if bfloat16_probe_device.type == "cuda":
            torch.cuda.synchronize()
        native_bfloat16 = True
    except (RuntimeError, NotImplementedError) as error:
        native_bfloat16_reason = str(error).splitlines()[0]
    dtype_specs = [
        ("float64", torch.float64, torch.float64),
        ("float32", torch.float32, torch.float32),
        ("bfloat16_storage_float32_compute", torch.bfloat16, torch.float32),
    ]
    for prior in config["sweeps"]["prior_precision"]:
        raw_keys = unit_rows(base + 1e-3 * torch.randn(n_remote + window, d_key, generator=generator, dtype=torch.float64))
        raw_values = torch.randn(n_remote + window, d_value, generator=generator, dtype=torch.float64)
        raw_evidence = torch.logspace(-4, 4, n_remote + window, dtype=torch.float64)
        raw_query = unit_rows(base + 1e-3 * torch.randn(1, d_key, generator=generator, dtype=torch.float64))[0]
        for label, storage_dtype, compute_dtype in dtype_specs:
            keys = quantize(raw_keys, storage_dtype)
            values = quantize(raw_values, storage_dtype)
            evidence = quantize(raw_evidence, storage_dtype)
            query = quantize(raw_query, storage_dtype)
            prior_quantized = float(torch.tensor(prior, dtype=storage_dtype).to(compute_dtype))
            remote_keys = keys[:n_remote].to(compute_dtype)
            remote_values = values[:n_remote].to(compute_dtype)
            remote_evidence = evidence[:n_remote].to(compute_dtype)
            precision = prior_quantized * torch.eye(d_key, dtype=compute_dtype) + torch.einsum(
                "n,ni,nj->ij", remote_evidence, remote_keys, remote_keys
            )
            cross = torch.einsum("n,nv,nd->vd", remote_evidence, remote_values, remote_keys)
            cholesky_ok = True
            error_text = ""
            try:
                output = aurelis_read(
                    precision.view(1, 1, d_key, d_key),
                    cross.view(1, 1, d_value, d_key),
                    keys[n_remote:].to(compute_dtype).view(1, 1, window, d_key),
                    values[n_remote:].to(compute_dtype).view(1, 1, window, d_value),
                    evidence[n_remote:].to(compute_dtype).view(1, 1, window),
                    query.to(compute_dtype).view(1, 1, d_key),
                    temperature=4.0,
                )
            except RuntimeError as error:
                cholesky_ok = False
                error_text = str(error).splitlines()[0]
                output = None
            oracle_precision = precision.to(torch.float64)
            oracle_cross = cross.to(torch.float64)
            oracle = None
            try:
                oracle = aurelis_read(
                    oracle_precision.view(1, 1, d_key, d_key),
                    oracle_cross.view(1, 1, d_value, d_key),
                    keys[n_remote:].to(torch.float64).view(1, 1, window, d_key),
                    values[n_remote:].to(torch.float64).view(1, 1, window, d_value),
                    evidence[n_remote:].to(torch.float64).view(1, 1, window),
                    query.to(torch.float64).view(1, 1, d_key),
                    temperature=4.0,
                )
            except RuntimeError:
                pass
            kappa = float(torch.linalg.cond(oracle_precision)) if torch.isfinite(oracle_precision).all() else math.inf
            forward_error = max_output_error(output, oracle) if output is not None and oracle is not None else math.inf
            solve_residual = (
                max(float(output.diagnostics.solve_residual_q), float(output.diagnostics.solve_residual_kbar))
                if output is not None
                else math.inf
            )
            finite = bool(output is not None and torch.isfinite(output.bayes).all())
            rows.append(
                {
                    "storage_dtype": label,
                    "compute_dtype": str(compute_dtype).removeprefix("torch."),
                    "prior": prior,
                    "condition_number": kappa,
                    "cholesky_status": "success" if cholesky_ok else "failure",
                    "cholesky_error": error_text,
                    "solve_residual": solve_residual,
                    "forward_error_vs_fp64_identically_quantized_inputs": forward_error,
                    "finite": finite,
                    "domain": "valid" if cholesky_ok and oracle is not None else "numerically_unresolved",
                }
            )

    invalid_rows = []
    invalid_cases = [
        ("singular_zero_prior", torch.zeros(d_key, d_key), torch.ones(window)),
        ("nan_precision", torch.full((d_key, d_key), float("nan")), torch.ones(window)),
        ("infinite_precision", torch.full((d_key, d_key), float("inf")), torch.ones(window)),
        ("negative_evidence", torch.eye(d_key), -torch.ones(window)),
    ]
    for name, precision, evidence in invalid_cases:
        status = "unexpected_success"
        finite = False
        message = ""
        try:
            output = aurelis_read(
                precision.to(torch.float64).view(1, 1, d_key, d_key),
                torch.zeros(1, 1, d_value, d_key, dtype=torch.float64),
                torch.zeros(1, 1, window, d_key, dtype=torch.float64),
                torch.zeros(1, 1, window, d_value, dtype=torch.float64),
                evidence.to(torch.float64).view(1, 1, window),
                torch.zeros(1, 1, d_key, dtype=torch.float64),
            )
            finite = bool(torch.isfinite(output.bayes).all())
            status = "returned_nonfinite" if not finite else "unexpected_finite"
        except (RuntimeError, ValueError) as error:
            status = "rejected"
            message = str(error).splitlines()[0]
        invalid_rows.append({"case": name, "expected_domain": "invalid", "observed": status, "finite": finite, "message": message})

    valid_fp32 = [row for row in rows if row["storage_dtype"] == "float32" and row["domain"] == "valid"]
    ordered = sorted(valid_fp32, key=lambda row: row["condition_number"])
    split = max(1, len(ordered) // 3)
    low_error = float(np.median([row["forward_error_vs_fp64_identically_quantized_inputs"] for row in ordered[:split]]))
    high_error = float(np.median([row["forward_error_vs_fp64_identically_quantized_inputs"] for row in ordered[-split:]]))
    max_fp32 = max(row["forward_error_vs_fp64_identically_quantized_inputs"] for row in valid_fp32)
    bfloat_rows = [row for row in rows if row["storage_dtype"].startswith("bfloat16") and row["domain"] == "valid"]
    max_bfloat = max(row["forward_error_vs_fp64_identically_quantized_inputs"] for row in bfloat_rows)
    passed = (
        high_error >= low_error
        and max_fp32 >= config["preregistered_tolerances"]["fp32_minimum_observable_forward_error"]
        and max_bfloat >= config["preregistered_tolerances"]["bfloat16_minimum_observable_forward_error"]
        and all(row["observed"] in ("rejected", "returned_nonfinite") for row in invalid_rows)
    )
    return {
        "native_bfloat16_cholesky_supported": native_bfloat16,
        "native_bfloat16_cholesky_probe_device": str(bfloat16_probe_device),
        "native_bfloat16_exclusion_reason": native_bfloat16_reason,
        "identical_quantized_input_policy": "Each reduced row is compared with fp64 after the same storage quantization; bfloat16 storage is promoted to fp32 because native Cholesky is unsupported on the active probe device.",
        "rows": rows,
        "invalid_domain_rows": invalid_rows,
        "condition_tracking": {"low_condition_fp32_median_error": low_error, "high_condition_fp32_median_error": high_error},
        "maximum_fp32_forward_error": max_fp32,
        "maximum_bfloat16_storage_forward_error": max_bfloat,
        "status": "PASS" if passed else "FAIL",
    }


def tensor_storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
    storages: dict[int, int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def cost_sweep(config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for d_key in config["sweeps"]["d_key"]:
        for d_value in config["sweeps"]["d_value"]:
            for window in config["sweeps"]["window"]:
                state = initial_state(1, 1, d_key, d_value, window)
                tensors = (
                    state.precision,
                    state.cross,
                    state.factor,
                    state.cache_keys,
                    state.cache_values,
                    state.cache_evidence,
                    state.cache_ids,
                )
                observed = tensor_storage_bytes(tensors)
                theoretical_minimum = 8 * (
                    d_key**2 + d_value * d_key + window * (d_key + d_value + 1)
                )
                implementation_analytic = 8 * (
                    2 * d_key**2 + d_value * d_key + window * (d_key + d_value + 1)
                ) + 8 * window
                rows.append(
                    {
                        "d_key": d_key,
                        "d_value": d_value,
                        "window": window,
                        "theoretical_minimum_dense_bytes_excluding_ids": theoretical_minimum,
                        "implementation_analytic_bytes": implementation_analytic,
                        "observed_unique_tensor_storage_bytes": observed,
                        "difference": observed - implementation_analytic,
                    }
                )

    d_key, d_value, window = 8, 3, 8
    precision = torch.eye(d_key, dtype=torch.float64).view(1, 1, d_key, d_key)
    cross = torch.randn(1, 1, d_value, d_key, dtype=torch.float64)
    keys = torch.randn(1, 1, window, d_key, dtype=torch.float64)
    values = torch.randn(1, 1, window, d_value, dtype=torch.float64)
    evidence = torch.ones(1, 1, window, dtype=torch.float64)
    query = torch.randn(1, 1, d_key, dtype=torch.float64)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profile:
        aurelis_read(precision, cross, keys, values, evidence, query)
    events = profile.key_averages()
    observed_calls = int(sum(event.count for event in events))
    profiled_flops = int(sum(event.flops for event in events))
    analytic_flops = int(
        window * (2 * d_key - 1)
        + window
        + 3 * window
        + d_key * (2 * window - 1)
        + d_value * (2 * window - 1)
        + (3 * d_key**3) // 3
        + 6 * d_key**2
        + 2 * d_value * (2 * d_key - 1)
        + 5 * (2 * d_key - 1)
        + 6 * d_value
    )
    operation_record = {
        "shape": {"d_key": d_key, "d_value": d_value, "window": window},
        "analytic_arithmetic_flops_including_dense_cholesky_and_three_solves": analytic_flops,
        "torch_profiler_supported_flops": profiled_flops,
        "torch_profiler_operator_invocations": observed_calls,
        "profiler_scope_note": "PyTorch reports FLOPs for supported kernels only; Cholesky and triangular solve FLOPs are absent, so invocation count and supported-FLOP coverage are retained separately.",
        "supported_flop_fraction_of_analytic": profiled_flops / analytic_flops,
    }
    return {
        "state_rows": rows,
        "operation_record": operation_record,
        "status": "PASS" if all(row["difference"] == 0 for row in rows) and observed_calls > 0 and profiled_flops > 0 else "FAIL",
    }


def plots(monte_carlo: list[dict[str, Any]], conditioning: dict[str, Any]) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    labels = [row["name"] for row in monte_carlo]
    predicted = [row["estimates"]["V_g"]["predicted"] for row in monte_carlo]
    empirical = [row["estimates"]["V_g"]["empirical"] for row in monte_carlo]
    position = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(position - 0.18, predicted, 0.36, label="predicted")
    axis.bar(position + 0.18, empirical, 0.36, label="empirical")
    axis.set_xticks(position, labels, rotation=20, ha="right")
    axis.set_ylabel("conditional per-coordinate MSE")
    axis.set_title("Phase 1 conditional calibration")
    axis.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "conditional_calibration.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for label in ("float32", "bfloat16_storage_float32_compute"):
        rows = [row for row in conditioning["rows"] if row["storage_dtype"] == label and row["domain"] == "valid"]
        axis.loglog(
            [row["condition_number"] for row in rows],
            [max(row["forward_error_vs_fp64_identically_quantized_inputs"], 1e-20) for row in rows],
            marker="o",
            label=label,
        )
    axis.set_xlabel("condition number")
    axis.set_ylabel("maximum forward error vs fp64")
    axis.set_title("Reduced precision on identical quantized inputs")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "conditioning_precision.png", dpi=180)
    plt.close(fig)


def report(summary: dict[str, Any]) -> str:
    aggregate = summary["aggregate"]
    covariance = next(row for row in summary["monte_carlo"] if row["name"] == "constructed_covariance_omission")
    return f"""# AURELIS Phase 1 mathematical-oracle report

Generated by the pinned config and experiment with seed `{summary['seed']}`.
All numerical/statistical tolerances were loaded before data generation.

## Gate summary

| Evidence | Result |
|---|---:|
| Streaming/full-history prefixes checked | {aggregate['streaming_prefixes_checked']} |
| Maximum valid-row streaming tolerance ratio | {aggregate['maximum_streaming_tolerance_ratio']:.6g} |
| Deterministic identity rows | {aggregate['deterministic_rows']} |
| Maximum valid-row exact identity tolerance ratio | {aggregate['maximum_identity_tolerance_ratio']:.6g} |
| Conditional Monte Carlo regimes / trials each | {len(summary['monte_carlo'])} / {summary['config']['monte_carlo_trials']} |
| Covariance-omitting router paired regret z-score | {covariance['covariance_omission_z_score']:.3f} |
| fp32 maximum error against identical-quantized fp64 | {summary['conditioning']['maximum_fp32_forward_error']:.3e} |
| bfloat16-storage maximum error against identical-quantized fp64 | {summary['conditioning']['maximum_bfloat16_storage_forward_error']:.3e} |
| Overall experiment | **{summary['status']}** |

## Interpretation boundaries

Finite-temperature softmax is recorded as convergence, even when fp64
underflow numerically saturates the selected mass to one. Exact cached recall
is certified separately with an exactly one-hot singleton cache. The matched
conditional Gaussian rows support the covariance-aware gate; the retained
misspecified exception shows that its endpoint guarantee does not transfer to
an incorrect target/model. Native bfloat16 Cholesky support is explicitly
probed; where absent, bfloat16 storage inputs are promoted to fp32 compute and
compared with fp64 on the same quantized inputs.

State-byte rows compare the manuscript's minimum dense state with the actual
Phase 0 implementation, which deliberately keeps both `P` and its Cholesky
factor. The operation record separates analytic arithmetic from profiler-
supported FLOPs because the profiler does not assign FLOPs to dense Cholesky
or triangular solve kernels.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    torch_generator = torch.Generator().manual_seed(seed)
    numpy_generator = np.random.default_rng(seed)
    torch.set_default_dtype(torch.float64)
    RAW.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    streaming = streaming_sweep(config, torch_generator)
    deterministic = deterministic_sweep(config, torch_generator)
    route_recall = clipping_and_recall(config, torch_generator)
    monte_carlo = monte_carlo_sweep(config, numpy_generator)
    targets = target_conflict_and_misspecification(config)
    handoff = handoff_boundary(config, torch_generator)
    conditioning = dtype_conditioning_sweep(config, torch_generator)
    costs = cost_sweep(config)
    plots(monte_carlo, conditioning)

    identity_ratios = []
    gate_formula_ratios = []
    for row in deterministic:
        if row["valid_conditioned_identity_domain"]:
            identity_ratios.append(
                max(
                    row["residual_identity_error"],
                    row["general_gate_identity_error"],
                    row["linear_reproduction_error"],
                )
                / row["identity_tolerance"]
            )
        if row["valid_gate_covariance_reduction_domain"]:
            gate_formula_ratios.append(
                row["gate_form_equivalence_error"] / row["gate_formula_tolerance"]
            )
    covariance_case = next(row for row in monte_carlo if row["name"] == "constructed_covariance_omission")
    checks = {
        "streaming": all(row["status"] == "PASS" for row in streaming),
        "deterministic": all(row["status"] == "PASS" for row in deterministic),
        "clipping": all(row["status"] == "PASS" for row in route_recall["clipping"]),
        "one_hot": route_recall["hard_one_hot_realized_by_singleton_cache"]["status"] == "PASS",
        "monte_carlo": all(row["status"] == "PASS" for row in monte_carlo),
        "covariance_omission_measurable": covariance_case["covariance_omission_z_score"]
        >= config["preregistered_tolerances"]["covariance_omission_minimum_z_score"],
        "target_conflict_and_misspecification": targets["status"] == "PASS",
        "handoff": handoff["status"] == "PASS",
        "conditioning": conditioning["status"] == "PASS",
        "costs": costs["status"] == "PASS",
    }
    summary = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "seed": seed,
        "config": config,
        "config_path": str(args.config.resolve().relative_to(REPO)),
        "config_sha256": sha256(args.config),
        "generated_utc": datetime.now(UTC).isoformat(),
        "wall_time_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "torch_hip": torch.version.hip,
            "commit": git(["rev-parse", "HEAD"]),
            "dirty_paths": len(git(["status", "--short"]).splitlines()),
            "pid": os.getpid(),
        },
        "aggregate": {
            "streaming_cases": len(streaming),
            "streaming_prefixes_checked": sum(row["prefixes_checked"] for row in streaming),
            "maximum_streaming_tolerance_ratio": max(row["maximum_tolerance_ratio_valid_rows"] for row in streaming),
            "partition_failures": sum(row["partition_failures"] for row in streaming),
            "deterministic_rows": len(deterministic),
            "maximum_identity_tolerance_ratio": max(identity_ratios, default=0.0),
            "maximum_gate_formula_tolerance_ratio": max(gate_formula_ratios, default=0.0),
            "invalid_gate_cancellation_rows_retained": sum(
                not row["valid_gate_covariance_reduction_domain"] for row in deterministic
            ),
            "invalid_conditioning_rows_retained": sum(row["invalid_conditioning_rows"] for row in streaming) + len(conditioning["invalid_domain_rows"]),
        },
        "streaming": streaming,
        "deterministic": deterministic,
        "routing_and_recall": route_recall,
        "monte_carlo": monte_carlo,
        "target_conflict": targets,
        "handoff": handoff,
        "conditioning": conditioning,
        "costs": costs,
    }
    dump_jsonl(RAW / "streaming_rows.jsonl", streaming)
    dump_jsonl(RAW / "deterministic_rows.jsonl", deterministic)
    dump_jsonl(RAW / "conditioning_rows.jsonl", conditioning["rows"] + conditioning["invalid_domain_rows"])
    dump_json(RESULTS / "metrics.json", summary)
    (RESULTS / "report.md").write_text(report(summary), encoding="utf-8")
    print(json.dumps(json_value({"status": summary["status"], "checks": checks, "aggregate": summary["aggregate"], "wall_time_seconds": summary["wall_time_seconds"]}), indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
