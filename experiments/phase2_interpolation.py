#!/usr/bin/env python3
"""Run the complete finite-epsilon interpolation and capacity sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from csm import FP64GaussMarkovMemory, make_keys, make_values
from csm.synthetic import dataset_seed


def git_output(arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def associations(dimension: int, load: float) -> int:
    return max(1, int(round(dimension * load)))


def finite(value: float) -> float | str:
    return value if math.isfinite(value) else "inf"


def percentile(values: Iterable[float], q: float) -> float:
    return float(np.quantile(np.asarray(list(values), dtype=np.float64), q))


def run_sweep(
    config: dict[str, Any], device: torch.device
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    minimum_epsilon = min(config["epsilons"])
    maximum_epsilon = max(config["epsilons"])
    direction: dict[tuple[Any, ...], dict[float, float]] = defaultdict(dict)
    independent_limit: list[float] = []
    breakdown_limit: list[float] = []
    theorem_ratios: list[float] = []

    for d_key in config["key_dimensions"]:
        for load in config["loads"]:
            count = associations(d_key, load)
            for seed in config["seeds"]:
                for key_regime_index, key_regime in enumerate(config["key_regimes"]):
                    key_seed = dataset_seed(seed, d_key, count, key_regime_index)
                    key_generator = torch.Generator(device=device).manual_seed(key_seed)
                    keys = make_keys(
                        key_regime,
                        count,
                        d_key,
                        generator=key_generator,
                        device=device,
                        correlation=config["correlation"],
                        near_collinear_noise=config["near_collinear_noise"],
                    )
                    singular_values = torch.linalg.svdvals(keys)
                    rank_tolerance = (
                        max(keys.shape)
                        * torch.finfo(torch.float64).eps
                        * singular_values.max()
                    )
                    numerical_rank = int((singular_values > rank_tolerance).sum().item())
                    full_row_rank = numerical_rank == count and count <= d_key
                    minimum_gram_eigenvalue = (
                        float(singular_values.min().square().item())
                        if full_row_rank
                        else 0.0
                    )
                    maximum_gram_eigenvalue = float(
                        singular_values.max().square().item()
                    )
                    gram_condition = (
                        maximum_gram_eigenvalue / minimum_gram_eigenvalue
                        if full_row_rank
                        else float("inf")
                    )

                    value_sets = []
                    for value_regime_index, value_regime in enumerate(
                        config["value_regimes"]
                    ):
                        d_value = (
                            count
                            if value_regime == "one_hot"
                            and count < config["value_dimension"]
                            else config["value_dimension"]
                        )
                        value_seed = dataset_seed(
                            seed,
                            d_key,
                            count,
                            key_regime_index,
                            value_regime_index,
                        )
                        value_generator = torch.Generator(device=device).manual_seed(
                            value_seed
                        )
                        values = make_values(
                            value_regime,
                            count,
                            d_value,
                            generator=value_generator,
                            device=device,
                        )
                        value_sets.append((value_regime, values))

                    for epsilon in config["epsilons"]:
                        system_condition = (
                            maximum_gram_eigenvalue + epsilon
                        ) / (
                            (minimum_gram_eigenvalue + epsilon)
                            if full_row_rank and count == d_key
                            else epsilon
                        )
                        # For K < d, A has d-K eigenvalues equal to epsilon.
                        # For K > d, singular_values already spans all d directions.
                        if count > d_key and numerical_rank == d_key:
                            system_condition = (
                                maximum_gram_eigenvalue + epsilon
                            ) / (float(singular_values.min().square().item()) + epsilon)

                        for value_regime, values in value_sets:
                            memory = FP64GaussMarkovMemory(
                                d_key, values.shape[1], epsilon=epsilon
                            )
                            state = memory.undiscounted_state(keys, values)
                            reads, confidences = memory.read_many_with_confidence(
                                state, keys
                            )
                            differences = reads - values
                            error_norms = torch.linalg.vector_norm(differences, dim=1)
                            target_norms = torch.linalg.vector_norm(values, dim=1)
                            relative_errors = error_norms / target_norms.clamp_min(1e-300)
                            frobenius_error = float(
                                torch.linalg.vector_norm(differences).item()
                            )
                            value_frobenius = float(
                                torch.linalg.vector_norm(values).item()
                            )
                            relative_frobenius = frobenius_error / max(
                                value_frobenius, 1e-300
                            )
                            theorem_bound = float("nan")
                            theorem_ratio = float("nan")
                            if full_row_rank:
                                theorem_bound = (
                                    epsilon
                                    / (minimum_gram_eigenvalue + epsilon)
                                    * value_frobenius
                                )
                                theorem_ratio = frobenius_error / max(
                                    theorem_bound, 1e-300
                                )
                                theorem_ratios.append(theorem_ratio)

                            row = {
                                "d_key": d_key,
                                "d_value": values.shape[1],
                                "associations": count,
                                "load": load,
                                "seed": seed,
                                "key_regime": key_regime,
                                "value_regime": value_regime,
                                "epsilon": epsilon,
                                "numerical_rank": numerical_rank,
                                "full_row_rank": full_row_rank,
                                "minimum_gram_eigenvalue": minimum_gram_eigenvalue,
                                "gram_condition_number": finite(gram_condition),
                                "system_condition_number": finite(system_condition),
                                "mean_relative_error": float(relative_errors.mean().item()),
                                "median_relative_error": float(relative_errors.median().item()),
                                "p90_relative_error": float(
                                    torch.quantile(relative_errors, 0.9).item()
                                ),
                                "p99_relative_error": float(
                                    torch.quantile(relative_errors, 0.99).item()
                                ),
                                "maximum_relative_error": float(relative_errors.max().item()),
                                "relative_frobenius_error": relative_frobenius,
                                "exact_recall_rate": float(
                                    (
                                        relative_errors
                                        <= config["exact_relative_tolerance"]
                                    )
                                    .double()
                                    .mean()
                                    .item()
                                ),
                                "mean_c_q": float(confidences.mean().item()),
                                "median_c_q": float(confidences.median().item()),
                                "maximum_c_q": float(confidences.max().item()),
                                "theorem_frobenius_bound": theorem_bound,
                                "actual_to_theorem_bound": theorem_ratio,
                            }
                            rows.append(row)

                            case = (
                                d_key,
                                count,
                                seed,
                                key_regime,
                                value_regime,
                            )
                            direction[case][epsilon] = relative_frobenius
                            if epsilon == minimum_epsilon:
                                if (
                                    key_regime
                                    in ("orthogonal", "random_gaussian", "correlated")
                                    and full_row_rank
                                ):
                                    independent_limit.append(relative_frobenius)
                                if key_regime == "duplicate" or count > d_key:
                                    breakdown_limit.append(relative_frobenius)

    direction_checks = [
        values[minimum_epsilon] <= values[maximum_epsilon] * (1.0 + 1e-12)
        for values in direction.values()
    ]
    summary = {
        "row_count": len(rows),
        "dataset_count": len(direction),
        "independent_limit_median_relative_error": percentile(
            independent_limit, 0.5
        ),
        "independent_limit_p99_relative_error": percentile(independent_limit, 0.99),
        "breakdown_limit_median_relative_error": percentile(breakdown_limit, 0.5),
        "breakdown_to_independent_median_ratio": percentile(
            breakdown_limit, 0.5
        )
        / max(percentile(independent_limit, 0.5), 1e-300),
        "finite_epsilon_direction_fraction": sum(direction_checks)
        / len(direction_checks),
        "maximum_actual_to_theorem_bound": max(theorem_ratios),
        "independent_limit_case_count": len(independent_limit),
        "breakdown_limit_case_count": len(breakdown_limit),
        "theorem_case_count": len(theorem_ratios),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_quantiles(
    rows: list[dict[str, Any]], group_name: str, x_name: str
) -> dict[tuple[str, float], tuple[float, float, float]]:
    groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        groups[(str(row[group_name]), float(row[x_name]))].append(
            float(row["relative_frobenius_error"])
        )
    return {
        key: (
            percentile(values, 0.1),
            percentile(values, 0.5),
            percentile(values, 0.9),
        )
        for key, values in groups.items()
    }


def make_plots(
    rows: list[dict[str, Any]], config: dict[str, Any], directory: Path
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    regimes = config["key_regimes"]

    quantiles = grouped_quantiles(rows, "key_regime", "epsilon")
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for regime in regimes:
        x = np.asarray(config["epsilons"])
        triples = [quantiles[(regime, float(value))] for value in x]
        low, median, high = np.asarray(triples).T
        axis.loglog(x, median, marker="o", label=regime)
        axis.fill_between(x, low, high, alpha=0.12)
    axis.set(
        xlabel="epsilon",
        ylabel="relative Frobenius recall error",
        title="Finite-epsilon error (median; 10–90% band)",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "error_vs_epsilon.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    minimum_epsilon = min(config["epsilons"])
    low_epsilon = [row for row in rows if row["epsilon"] == minimum_epsilon]
    quantiles = grouped_quantiles(low_epsilon, "key_regime", "load")
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for regime in regimes:
        x = np.asarray(config["loads"])
        triples = [quantiles[(regime, float(value))] for value in x]
        low, median, high = np.asarray(triples).T
        axis.semilogy(x, median, marker="o", label=regime)
        axis.fill_between(x, low, high, alpha=0.12)
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1, label="K=d_key")
    axis.set(
        xlabel="load K / d_key",
        ylabel="relative Frobenius recall error",
        title=f"Capacity at epsilon={minimum_epsilon:g} (median; 10–90% band)",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "error_vs_load.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    eligible = [
        row
        for row in rows
        if row["full_row_rank"] and row["minimum_gram_eigenvalue"] > 0
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for regime in regimes:
        subset = [row for row in eligible if row["key_regime"] == regime]
        if subset:
            axis.loglog(
                [row["minimum_gram_eigenvalue"] for row in subset],
                [row["relative_frobenius_error"] for row in subset],
                ".",
                alpha=0.22,
                markersize=3,
                label=regime,
            )
    axis.set(
        xlabel="minimum eigenvalue of key Gram matrix",
        ylabel="relative Frobenius recall error",
        title="Conditioning controls regularization error",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "error_vs_min_gram_eigenvalue.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for regime in regimes:
        subset = [row for row in rows if row["key_regime"] == regime]
        axis.loglog(
            np.maximum([row["mean_c_q"] for row in subset], 1e-16),
            np.maximum(
                [row["relative_frobenius_error"] for row in subset], 1e-16
            ),
            ".",
            alpha=0.16,
            markersize=3,
            label=regime,
        )
    axis.set(
        xlabel="mean c(q) (posterior variance; lower is surer)",
        ylabel="relative Frobenius recall error",
        title="Reported uncertainty statistic versus actual recall error",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "confidence_vs_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    bins = np.asarray(config["conditioning_bins_log10"], dtype=np.float64)
    heat = np.full((len(config["loads"]), len(bins) - 1), np.nan)
    for load_index, load in enumerate(config["loads"]):
        for bin_index in range(len(bins) - 1):
            values = []
            for row in rows:
                condition = row["system_condition_number"]
                if condition == "inf" or row["load"] != load:
                    continue
                log_condition = math.log10(max(float(condition), 1.0))
                if bins[bin_index] <= log_condition < bins[bin_index + 1]:
                    values.append(max(float(row["relative_frobenius_error"]), 1e-16))
            if values:
                heat[load_index, bin_index] = math.log10(percentile(values, 0.5))
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    image = axis.imshow(heat, aspect="auto", origin="lower", cmap="viridis")
    axis.set_xticks(np.arange(len(bins) - 1))
    axis.set_xticklabels(
        [f"{bins[i]:g}–{bins[i + 1]:g}" for i in range(len(bins) - 1)],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(np.arange(len(config["loads"])))
    axis.set_yticklabels(config["loads"])
    axis.set(
        xlabel="log10 condition-number bin",
        ylabel="load K / d_key",
        title="Median log10 recall error by load and conditioning",
    )
    fig.colorbar(image, ax=axis, label="median log10 relative error")
    fig.tight_layout()
    path = directory / "load_conditioning_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def gate_summary(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["gate"]
    checks = {
        "independent_limit": summary[
            "independent_limit_median_relative_error"
        ]
        <= thresholds["independent_limit_median_relative_error"],
        "finite_epsilon_direction": summary["finite_epsilon_direction_fraction"]
        >= thresholds["finite_epsilon_direction_fraction"],
        "finite_epsilon_theorem_bound": summary[
            "maximum_actual_to_theorem_bound"
        ]
        <= thresholds["theorem_bound_ratio"],
        "dependent_overcapacity_breakdown": summary[
            "breakdown_to_independent_median_ratio"
        ]
        >= thresholds["breakdown_error_ratio"],
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": thresholds}


def render_report(record: dict[str, Any]) -> str:
    summary = record["summary"]
    gate = record["gate"]
    status = "PASS" if gate["passed"] else "FAIL"
    checks = "\n".join(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in gate["checks"].items()
    )
    plots = "\n".join(
        f"- [`{Path(path).name}`](../{path})" for path in record["plots"]
    )
    return f"""# Phase 2 interpolation report

## Gate decision: {status}

This sweep retains all {summary['row_count']:,} aggregate rows from {summary['dataset_count']:,} deterministic key/value datasets. It covers five key dimensions, eight loads, seven logarithmic epsilon values, three seeds, five key regimes, and three value regimes. No seed or difficult regime was removed.

{checks}

| Measurement | Observed | Gate threshold |
|---|---:|---:|
| independent under-capacity median relative error at minimum epsilon | {summary['independent_limit_median_relative_error']:.6e} | <= {gate['thresholds']['independent_limit_median_relative_error']:.1e} |
| independent under-capacity p99 relative error at minimum epsilon | {summary['independent_limit_p99_relative_error']:.6e} | diagnostic |
| cases with error no worse at minimum than maximum epsilon | {summary['finite_epsilon_direction_fraction']:.3%} | >= {gate['thresholds']['finite_epsilon_direction_fraction']:.1%} |
| maximum actual / valid theorem bound | {summary['maximum_actual_to_theorem_bound']:.6f} | <= {gate['thresholds']['theorem_bound_ratio']:.6f} |
| dependent/over-capacity to independent median error ratio | {summary['breakdown_to_independent_median_ratio']:.3e} | >= {gate['thresholds']['breakdown_error_ratio']:.1f} |

## Mathematical interpretation

For a row-key matrix `K` with full row rank and value matrix `V`, stored-key recall is `K(K^T K + epsilon I)^(-1)K^T V`. The exact push-through identity makes its error `-epsilon (K K^T + epsilon I)^(-1)V`, hence

`||error||_F <= epsilon / (lambda_min(K K^T) + epsilon) ||V||_F`.

The automated bound check covers {summary['theorem_case_count']:,} full-row-rank cases and is deliberately not applied to duplicate or over-capacity cases. The observed limit and finite-epsilon direction support the theorem in its stated domain. Conditioning controls how small epsilon must become before that limit is numerically visible.

Dependent keys and `K > d_key` cannot interpolate arbitrary values: the read is a regularized projection through a rank-limited key geometry. Their nonzero low-epsilon error is therefore a structural failure of arbitrary association recall, not evidence against the scoped full-row-rank theorem. Near-collinear keys are mathematically independent in most generated cases but expose the predicted finite-precision and regularization sensitivity.

`c(q) = q^T(S + epsilon I)^(-1)q` is plotted as a posterior-variance statistic; lower means surer. It is not asserted to be a calibrated per-item error probability.

## Plots

{plots}

## Reproducibility

- git source checkpoint: `{record['git_commit']}`
- working tree dirty at experiment start: `{record['working_tree_dirty']}`
- config: [`configs/phase2_interpolation.json`](../configs/phase2_interpolation.json)
- complete seeds: `{record['config']['seeds']}`
- hardware: `{record['hardware']}`
- software: Python `{record['software']['python']}`, PyTorch `{record['software']['torch']}`, HIP `{record['software']['hip']}`, NumPy `{record['software']['numpy']}`
- wall-clock time: `{record['wall_clock_seconds']:.3f}` seconds
- peak allocated VRAM: `{record['peak_vram_bytes'] / 2**30:.6f}` GiB
- all aggregate rows: [`phase2/interpolation_sweep.csv`](phase2/interpolation_sweep.csv)
- machine-readable summary and resolved config: [`phase2_metrics.json`](phase2_metrics.json)

## Scoped conclusion

{record['interpretation']}

This phase tests exact synthetic association geometry only. It does not establish learned-memory, language-model, wall-clock, or broad architectural superiority claims.
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase2_interpolation.json")
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--plots", type=Path, default=Path("plots/phase2"))
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    source_dirty = bool(git_output(["status", "--porcelain"]))
    config = json.loads(args.config.read_text())
    device = select_device(config["device"])
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows, summary = run_sweep(config, device)
    plots = make_plots(rows, config, args.plots)
    gate = gate_summary(summary, config)
    synchronize(device)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "working_tree_dirty": source_dirty,
        "config_path": str(args.config),
        "config": config,
        "hardware": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "numpy": np.__version__,
        },
        "wall_clock_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "summary": summary,
        "plots": plots,
        "gate": gate,
        "interpretation": (
            "Within the stated full-row-rank domain, recall approaches exact interpolation as epsilon decreases and respects the finite-epsilon bound. Dependent and over-capacity geometries show the expected measurable rank-limited breakdown."
            if gate["passed"]
            else "At least one Phase 2 criterion failed. The raw record is retained; the affected mathematical or empirical claim must be narrowed or corrected before relying on it."
        ),
    }
    args.results.mkdir(parents=True, exist_ok=True)
    write_csv(args.results / "phase2" / "interpolation_sweep.csv", rows)
    (args.results / "phase2_metrics.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    report = render_report(record)
    (args.results / "phase2_interpolation_report.md").write_text(report)
    print(report)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
