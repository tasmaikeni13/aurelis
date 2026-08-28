#!/usr/bin/env python3
"""Reproducible numerical analysis for the AURELIS hybrid head.

The script is deliberately NumPy-only.  It tests algebraic identities, the
closed-form uncertainty router, conditional Bayesian calibration, exact linear
reproduction, local exception recall, and a small bias/variance sweep.  It is a
theory oracle, not a training implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "analysis" / "results"
DEFAULT_PLOTS = ROOT / "analysis" / "plots"


def unit_rows(x: Array) -> Array:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def softmax(logits: Array) -> Array:
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


@dataclass(frozen=True)
class RemoteState:
    precision: Array
    cross: Array


@dataclass(frozen=True)
class HeadResult:
    remote: Array
    local: Array
    residual: Array
    routed: Array
    weights: Array
    key_mean: Array
    residual_query: Array
    local_noise: float
    remote_variance: float
    residual_variance: float
    covariance: float
    gate_raw: float
    gate: float
    routed_variance: float


def remote_state(
    keys: Array,
    values: Array,
    evidence: Array,
    prior_precision: float,
) -> RemoteState:
    """Return P=alpha I+sum beta kk^T and C=sum beta vk^T."""

    d_key = keys.shape[1]
    weighted_keys = evidence[:, None] * keys
    precision = prior_precision * np.eye(d_key) + keys.T @ weighted_keys
    cross = values.T @ weighted_keys
    return RemoteState(precision=precision, cross=cross)


def solve_precision(state: RemoteState, query: Array) -> Array:
    return np.linalg.solve(state.precision, query)


def memory_read(state: RemoteState, query: Array) -> Array:
    return state.cross @ solve_precision(state, query)


def aurelis_head(
    state: RemoteState,
    cache_keys: Array,
    cache_values: Array,
    cache_evidence: Array,
    query: Array,
    temperature: float,
    *,
    gate_override: float | None = None,
) -> HeadResult:
    """Evaluate one AURELIS head and its scalar uncertainty certificate.

    The formula assumes independent output rows with common noise scale.  All
    returned variances omit that common multiplicative scale.
    """

    weights = softmax(temperature * (cache_keys @ query))
    key_mean = weights @ cache_keys
    local = weights @ cache_values
    residual_query = query - key_mean
    remote = memory_read(state, query)
    memory_key_mean = memory_read(state, key_mean)
    residual = remote + (local - memory_key_mean)

    p_query = solve_precision(state, query)
    p_residual = solve_precision(state, residual_query)
    p_key_mean = solve_precision(state, key_mean)
    local_noise = float(np.sum(weights * weights / cache_evidence))
    remote_variance = float(query @ p_query)
    residual_variance = float(local_noise + residual_query @ p_residual)
    covariance = float(query @ p_residual)
    denominator = float(local_noise + key_mean @ p_key_mean)
    if denominator <= 0.0:
        raise AssertionError("the routing denominator must be positive")
    gate_raw = float((remote_variance - covariance) / denominator)
    gate = float(np.clip(gate_raw, 0.0, 1.0))
    if gate_override is not None:
        gate = float(gate_override)
    routed = remote + gate * (local - memory_key_mean)
    routed_variance = float(
        (1.0 - gate) ** 2 * remote_variance
        + gate**2 * residual_variance
        + 2.0 * gate * (1.0 - gate) * covariance
    )
    return HeadResult(
        remote=remote,
        local=local,
        residual=residual,
        routed=routed,
        weights=weights,
        key_mean=key_mean,
        residual_query=residual_query,
        local_noise=local_noise,
        remote_variance=remote_variance,
        residual_variance=residual_variance,
        covariance=covariance,
        gate_raw=gate_raw,
        gate=gate,
        routed_variance=routed_variance,
    )


def algebra_certificates(rng: np.random.Generator, trials: int = 256) -> dict[str, float]:
    max_decomposition_error = 0.0
    max_gate_stationarity_error = 0.0
    max_grid_regret = 0.0
    max_noninferiority_slack = 0.0
    max_equivalent_gate_error = 0.0
    for _ in range(trials):
        d_key, d_value, window = 7, 5, 9
        remote_keys = unit_rows(rng.normal(size=(24, d_key)))
        remote_values = rng.normal(size=(24, d_value))
        remote_evidence = np.exp(rng.normal(scale=0.35, size=24))
        state = remote_state(remote_keys, remote_values, remote_evidence, 0.7)
        cache_keys = unit_rows(rng.normal(size=(window, d_key)))
        cache_values = rng.normal(size=(window, d_value))
        cache_evidence = np.exp(rng.normal(scale=0.35, size=window))
        query = unit_rows(rng.normal(size=(1, d_key)))[0]
        result = aurelis_head(
            state, cache_keys, cache_values, cache_evidence, query, 3.0
        )

        operator = rng.normal(size=(d_value, d_key))
        local_residuals = cache_values - cache_keys @ operator.T
        memory_matrix = state.cross @ np.linalg.inv(state.precision)
        lhs = result.residual - operator @ query
        rhs = result.weights @ local_residuals + (
            memory_matrix - operator
        ) @ result.residual_query
        max_decomposition_error = max(
            max_decomposition_error, float(np.max(np.abs(lhs - rhs)))
        )

        denominator = (
            result.remote_variance
            + result.residual_variance
            - 2.0 * result.covariance
        )
        gate_from_quadratic = (
            result.remote_variance - result.covariance
        ) / denominator
        max_equivalent_gate_error = max(
            max_equivalent_gate_error,
            abs(gate_from_quadratic - result.gate_raw),
        )
        if 0.0 < result.gate_raw < 1.0:
            derivative = (
                2.0 * result.gate * result.residual_variance
                - 2.0 * (1.0 - result.gate) * result.remote_variance
                + 2.0
                * (1.0 - 2.0 * result.gate)
                * result.covariance
            )
            max_gate_stationarity_error = max(
                max_gate_stationarity_error, abs(derivative)
            )
        grid = np.linspace(0.0, 1.0, 10001)
        grid_variance = (
            (1.0 - grid) ** 2 * result.remote_variance
            + grid**2 * result.residual_variance
            + 2.0 * grid * (1.0 - grid) * result.covariance
        )
        max_grid_regret = max(
            max_grid_regret,
            result.routed_variance - float(np.min(grid_variance)),
        )
        max_noninferiority_slack = max(
            max_noninferiority_slack,
            result.routed_variance
            - min(result.remote_variance, result.residual_variance),
        )
    return {
        "trials": float(trials),
        "max_residual_decomposition_abs_error": max_decomposition_error,
        "max_gate_formula_equivalence_abs_error": max_equivalent_gate_error,
        "max_interior_gate_stationarity_abs_error": max_gate_stationarity_error,
        "max_variance_regret_vs_dense_grid": max_grid_regret,
        "max_noninferiority_slack": max_noninferiority_slack,
    }


def linear_reproduction(rng: np.random.Generator) -> dict[str, float]:
    d_key, d_value, window = 12, 7, 16
    operator = rng.normal(size=(d_value, d_key)) / math.sqrt(d_key)
    center = unit_rows(rng.normal(size=(1, d_key)))[0]
    cache_keys = unit_rows(center + 0.35 * rng.normal(size=(window, d_key)))
    cache_values = cache_keys @ operator.T
    query = unit_rows(center + 0.55 * rng.normal(size=(1, d_key)))[0]
    weights = softmax(2.5 * (cache_keys @ query))
    key_mean = weights @ cache_keys
    local = weights @ cache_values
    corrected = local + operator @ (query - key_mean)
    target = operator @ query
    return {
        "local_attention_l2_error": float(np.linalg.norm(local - target)),
        "full_residual_l2_error": float(np.linalg.norm(corrected - target)),
        "first_moment_mismatch_l2": float(np.linalg.norm(query - key_mean)),
    }


def exception_recall(rng: np.random.Generator) -> list[dict[str, Any]]:
    d_key, d_value, n_remote, window = 16, 8, 128, 16
    operator = rng.normal(size=(d_value, d_key)) / math.sqrt(d_key)
    remote_keys = unit_rows(rng.normal(size=(n_remote, d_key)))
    remote_values = remote_keys @ operator.T
    state = remote_state(
        remote_keys, remote_values, np.ones(n_remote), prior_precision=1e-8
    )
    cache_keys = unit_rows(rng.normal(size=(window, d_key)))
    cache_values = cache_keys @ operator.T
    exception = rng.normal(size=d_value)
    exception /= np.linalg.norm(exception)
    cache_values[0] += 2.5 * exception
    query = cache_keys[0].copy()
    target = cache_values[0]
    rows: list[dict[str, Any]] = []
    for temperature in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        result = aurelis_head(
            state,
            cache_keys,
            cache_values,
            np.ones(window),
            query,
            temperature,
        )
        rows.append(
            {
                "temperature": temperature,
                "target_attention_weight": float(result.weights[0]),
                "gate": result.gate,
                "remote_error": float(np.linalg.norm(result.remote - target)),
                "local_error": float(np.linalg.norm(result.local - target)),
                "full_residual_error": float(
                    np.linalg.norm(result.residual - target)
                ),
                "bayes_routed_error": float(np.linalg.norm(result.routed - target)),
            }
        )

    one_hot = np.zeros(window)
    one_hot[0] = 1.0
    key_mean = one_hot @ cache_keys
    local = one_hot @ cache_values
    exact = local + memory_read(state, query - key_mean)
    rows.append(
        {
            "temperature": "hard_one_hot",
            "target_attention_weight": 1.0,
            "gate": 1.0,
            "remote_error": float(np.linalg.norm(memory_read(state, query) - target)),
            "local_error": float(np.linalg.norm(local - target)),
            "full_residual_error": float(np.linalg.norm(exact - target)),
            "bayes_routed_error": float(np.linalg.norm(exact - target)),
        }
    )
    return rows


def conditional_calibration(
    rng: np.random.Generator, trials: int = 50000
) -> dict[str, float]:
    """Monte Carlo the exact conditional Gaussian formulas.

    We condition on a fixed remote posterior N(M, P^-1), then sample its latent
    operator and disjoint local observation noise.  This isolates the theorem
    from finite-sample posterior-estimation noise.
    """

    d_key, d_value, window = 10, 3, 12
    basis = rng.normal(size=(d_key, d_key))
    precision = basis.T @ basis + 1.5 * np.eye(d_key)
    posterior_mean = rng.normal(size=(d_value, d_key)) / math.sqrt(d_key)
    state = RemoteState(precision=precision, cross=posterior_mean @ precision)
    cache_keys = unit_rows(rng.normal(size=(window, d_key)))
    evidence = np.exp(rng.normal(scale=0.5, size=window))
    query = unit_rows(rng.normal(size=(1, d_key)))[0]
    pilot_values = cache_keys @ posterior_mean.T
    result = aurelis_head(
        state, cache_keys, pilot_values, evidence, query, temperature=3.5
    )

    covariance = np.linalg.inv(precision)
    chol = np.linalg.cholesky(covariance)
    operator_noise = rng.normal(size=(trials, d_value, d_key)) @ chol.T
    operators = posterior_mean[None, :, :] + operator_noise
    local_noise = rng.normal(size=(trials, d_value, window)) / np.sqrt(
        evidence[None, None, :]
    )
    local_values = np.einsum("tvd,wd->tvw", operators, cache_keys) + local_noise
    targets = np.einsum("tvd,d->tv", operators, query)
    local_means = np.einsum("w,tvw->tv", result.weights, local_values)
    remote_predictions = np.broadcast_to(result.remote, targets.shape)
    memory_key_mean = memory_read(state, result.key_mean)
    residual_predictions = remote_predictions + local_means - memory_key_mean
    routed_predictions = remote_predictions + result.gate * (
        local_means - memory_key_mean
    )

    remote_mse = float(np.mean((remote_predictions - targets) ** 2))
    residual_mse = float(np.mean((residual_predictions - targets) ** 2))
    routed_mse = float(np.mean((routed_predictions - targets) ** 2))
    return {
        "trials": float(trials),
        "gate_raw": result.gate_raw,
        "gate": result.gate,
        "predicted_remote_variance": result.remote_variance,
        "empirical_remote_mse": remote_mse,
        "remote_relative_calibration_error": abs(
            remote_mse / result.remote_variance - 1.0
        ),
        "predicted_full_residual_variance": result.residual_variance,
        "empirical_full_residual_mse": residual_mse,
        "residual_relative_calibration_error": abs(
            residual_mse / result.residual_variance - 1.0
        ),
        "predicted_routed_variance": result.routed_variance,
        "empirical_routed_mse": routed_mse,
        "routed_relative_calibration_error": abs(
            routed_mse / result.routed_variance - 1.0
        ),
        "predicted_gain_vs_best_endpoint": min(
            result.remote_variance, result.residual_variance
        )
        - result.routed_variance,
        "empirical_gain_vs_best_endpoint": min(remote_mse, residual_mse)
        - routed_mse,
    }


def bias_variance_sweep(rng: np.random.Generator) -> list[dict[str, float]]:
    d_key, d_value, window = 16, 6, 16
    prior_precision = 1.0
    noise_scale = 0.25
    repeats = 120
    rows: list[dict[str, float]] = []
    for n_remote in (8, 16, 32, 64, 128, 256):
        accum: dict[str, list[float]] = {
            "remote": [],
            "local": [],
            "residual": [],
            "routed": [],
            "gate": [],
            "predicted_routed": [],
        }
        for _ in range(repeats):
            operator = rng.normal(size=(d_value, d_key)) / math.sqrt(
                prior_precision
            )
            remote_keys = unit_rows(rng.normal(size=(n_remote, d_key)))
            remote_values = remote_keys @ operator.T + noise_scale * rng.normal(
                size=(n_remote, d_value)
            )
            beta = np.full(n_remote, noise_scale**-2)
            state = remote_state(
                remote_keys, remote_values, beta, prior_precision
            )
            cache_keys = unit_rows(rng.normal(size=(window, d_key)))
            cache_values = cache_keys @ operator.T + noise_scale * rng.normal(
                size=(window, d_value)
            )
            query = unit_rows(
                cache_keys[0] + 0.35 * rng.normal(size=(1, d_key))
            )[0]
            result = aurelis_head(
                state,
                cache_keys,
                cache_values,
                np.full(window, noise_scale**-2),
                query,
                temperature=6.0,
            )
            target = operator @ query
            for name, prediction in (
                ("remote", result.remote),
                ("local", result.local),
                ("residual", result.residual),
                ("routed", result.routed),
            ):
                accum[name].append(float(np.mean((prediction - target) ** 2)))
            accum["gate"].append(result.gate)
            # P already contains beta=sigma^-2, so its inverse and the local
            # sum(a_i^2 / beta_i) are expressed in observation units.  No
            # second multiplication by sigma^2 belongs here.
            accum["predicted_routed"].append(result.routed_variance)
        rows.append(
            {
                "n_remote": float(n_remote),
                "repeats": float(repeats),
                "remote_mse": float(np.mean(accum["remote"])),
                "local_mse": float(np.mean(accum["local"])),
                "full_residual_mse": float(np.mean(accum["residual"])),
                "routed_mse": float(np.mean(accum["routed"])),
                "mean_gate": float(np.mean(accum["gate"])),
                "predicted_routed_mse": float(
                    np.mean(accum["predicted_routed"])
                ),
            }
        )
    return rows


def conditioning_sweep(rng: np.random.Generator) -> list[dict[str, float]]:
    d_key, d_value, n_remote = 12, 5, 48
    base = unit_rows(rng.normal(size=(1, d_key)))[0]
    keys = unit_rows(base + 1e-3 * rng.normal(size=(n_remote, d_key)))
    values = rng.normal(size=(n_remote, d_value))
    query = unit_rows(base + 1e-3 * rng.normal(size=(1, d_key)))[0]
    rows: list[dict[str, float]] = []
    for prior in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0):
        state = remote_state(keys, values, np.ones(n_remote), prior)
        solved = solve_precision(state, query)
        inverse_solved = np.linalg.inv(state.precision) @ query
        relative_disagreement = float(
            np.linalg.norm(solved - inverse_solved)
            / max(np.linalg.norm(solved), np.finfo(float).tiny)
        )
        rows.append(
            {
                "prior_precision": prior,
                "condition_number": float(np.linalg.cond(state.precision)),
                "solve_inverse_relative_disagreement": relative_disagreement,
                "finite": float(
                    np.isfinite(solved).all() and np.isfinite(inverse_solved).all()
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def create_plots(
    plots_dir: Path,
    exception_rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, float]],
    calibration: dict[str, float],
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    finite_exception = [
        row
        for row in exception_rows
        if isinstance(row["temperature"], (int, float))
        and math.isfinite(row["temperature"])
    ]
    temperatures = [row["temperature"] for row in finite_exception]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for field, label in (
        ("remote_error", "remote solver"),
        ("local_error", "local attention"),
        ("full_residual_error", "full residual"),
        ("bayes_routed_error", "uncertainty-routed"),
    ):
        ax.plot(
            temperatures,
            [row[field] for row in finite_exception],
            marker="o",
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("attention temperature")
    ax.set_ylabel("L2 error on cached exception")
    ax.set_title("Episodic exception recall")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "exception_recall.png", dpi=180)
    plt.close(fig)

    n_remote = [row["n_remote"] for row in sweep_rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for field, label in (
        ("remote_mse", "remote solver"),
        ("local_mse", "local attention"),
        ("full_residual_mse", "full residual"),
        ("routed_mse", "uncertainty-routed"),
    ):
        ax.plot(n_remote, [row[field] for row in sweep_rows], marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("remote associations")
    ax.set_ylabel("mean squared error")
    ax.set_title("Bias/variance tradeoff under the matched Gaussian model")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "bias_variance_sweep.png", dpi=180)
    plt.close(fig)

    labels = ["remote", "full residual", "routed"]
    predicted = [
        calibration["predicted_remote_variance"],
        calibration["predicted_full_residual_variance"],
        calibration["predicted_routed_variance"],
    ]
    empirical = [
        calibration["empirical_remote_mse"],
        calibration["empirical_full_residual_mse"],
        calibration["empirical_routed_mse"],
    ]
    positions = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(positions - 0.18, predicted, width=0.36, label="predicted")
    ax.bar(positions + 0.18, empirical, width=0.36, label="Monte Carlo")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("conditional per-coordinate MSE")
    ax.set_title("Uncertainty calibration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "uncertainty_calibration.png", dpi=180)
    plt.close(fig)


def markdown_report(summary: dict[str, Any]) -> str:
    algebra = summary["algebra"]
    reproduction = summary["linear_reproduction"]
    calibration = summary["conditional_calibration"]
    last_exception = summary["exception_recall"][-1]
    sweep = summary["bias_variance_sweep"]
    return f"""# AURELIS numerical analysis

Generated deterministically by `analysis/aurelis_numerical.py` with seed
`{summary['seed']}`.  These experiments test the mathematical mechanism; they
are not language-model or accelerator benchmarks.

## Gates

| Check | Result | Gate |
|---|---:|---:|
| Residual decomposition maximum absolute error | {algebra['max_residual_decomposition_abs_error']:.3e} | < 1e-10 |
| Gate formula equivalence maximum absolute error | {algebra['max_gate_formula_equivalence_abs_error']:.3e} | < 1e-10 |
| Routed variance regret against dense grid | {algebra['max_variance_regret_vs_dense_grid']:.3e} | < 1e-8 |
| Routed variance non-inferiority slack | {algebra['max_noninferiority_slack']:.3e} | <= 1e-12 |
| Linear reproduction error | {reproduction['full_residual_l2_error']:.3e} | < 1e-12 |
| Hard cached-hit residual error | {last_exception['full_residual_error']:.3e} | < 1e-12 |
| Remote uncertainty relative calibration error | {calibration['remote_relative_calibration_error']:.3%} | < 3% |
| Full-residual uncertainty relative calibration error | {calibration['residual_relative_calibration_error']:.3%} | < 3% |
| Routed uncertainty relative calibration error | {calibration['routed_relative_calibration_error']:.3%} | < 3% |

## Principal observations

Arbitrary softmax weights incur L2 error
`{reproduction['local_attention_l2_error']:.6f}` on an exactly linear map,
whereas first-moment residual correction reduces the error to
`{reproduction['full_residual_l2_error']:.3e}`.  With a one-hot cached hit, the
same correction returns the exceptional stored value with error
`{last_exception['full_residual_error']:.3e}`.

In the conditional Gaussian experiment ({int(calibration['trials'])} trials),
the analytic router used gate `{calibration['gate']:.6f}`.  Its predicted and
empirical per-coordinate MSE were
`{calibration['predicted_routed_variance']:.6f}` and
`{calibration['empirical_routed_mse']:.6f}`.  The empirical improvement over
the better endpoint was `{calibration['empirical_gain_vs_best_endpoint']:.6f}`.

Across the finite-sample sweep, routed MSE moved from
`{sweep[0]['routed_mse']:.6f}` at {int(sweep[0]['n_remote'])} remote writes to
`{sweep[-1]['routed_mse']:.6f}` at {int(sweep[-1]['n_remote'])} remote writes.
The full tables are stored beside this report; conditioning failures are
retained rather than filtered.

## Scope

The calibration certificate assumes a linear-Gaussian latent operator,
disjoint remote/local observations, fixed attention weights conditional on
keys, and exact floating-point solves up to measured numerical error.  It is
not a claim that learned features satisfy those assumptions, nor that the
Bayes gate is optimal for exact episodic-copy targets.  The hard-hit result is
a separate deterministic theorem with gate one.
"""


def validate(summary: dict[str, Any]) -> None:
    algebra = summary["algebra"]
    calibration = summary["conditional_calibration"]
    reproduction = summary["linear_reproduction"]
    hard_hit = summary["exception_recall"][-1]
    assert algebra["max_residual_decomposition_abs_error"] < 1e-10
    assert algebra["max_gate_formula_equivalence_abs_error"] < 1e-10
    assert algebra["max_variance_regret_vs_dense_grid"] < 1e-8
    assert algebra["max_noninferiority_slack"] <= 1e-12
    assert reproduction["full_residual_l2_error"] < 1e-12
    assert hard_hit["full_residual_error"] < 1e-12
    assert calibration["remote_relative_calibration_error"] < 0.03
    assert calibration["residual_relative_calibration_error"] < 0.03
    assert calibration["routed_relative_calibration_error"] < 0.03
    assert calibration["predicted_gain_vs_best_endpoint"] >= -1e-12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    summary: dict[str, Any] = {
        "architecture": "AURELIS",
        "seed": args.seed,
        "numpy_version": np.__version__,
        "algebra": algebra_certificates(rng),
        "linear_reproduction": linear_reproduction(rng),
        "exception_recall": exception_recall(rng),
        "conditional_calibration": conditional_calibration(rng),
        "bias_variance_sweep": bias_variance_sweep(rng),
        "conditioning_sweep": conditioning_sweep(rng),
    }
    validate(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "exception_recall.csv", summary["exception_recall"])
    write_csv(
        args.output_dir / "bias_variance_sweep.csv",
        summary["bias_variance_sweep"],
    )
    write_csv(
        args.output_dir / "conditioning_sweep.csv", summary["conditioning_sweep"]
    )
    create_plots(
        args.plots_dir,
        summary["exception_recall"],
        summary["bias_variance_sweep"],
        summary["conditional_calibration"],
    )
    (args.output_dir / "NUMERICAL_REPORT.md").write_text(
        markdown_report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
