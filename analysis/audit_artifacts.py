#!/usr/bin/env python3
"""Audit the current theory-only AURELIS deliverables."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    paper = ROOT / "aurelis.md"
    require(paper.is_file(), "standalone paper is missing")
    require(
        not (ROOT / "conjugate-state-machines.md").exists(),
        "superseded paper still exists",
    )
    paper_text = paper.read_text(encoding="utf-8")
    for required in (
        "# AURELIS:",
        "## 2. Related work and novelty boundary",
        "**Theorem 5.2",
        "**Theorem 6.1",
        "## 8. Formal verification",
        "## 9. Numerical analysis",
        "## 10. Failure analysis",
        "*End of paper.*",
    ):
        require(required in paper_text, f"paper section missing: {required}")
    for transition_phrase in (
        "formerly known as",
        "used to be",
        "we renamed",
        "previous paper",
        "old architecture",
    ):
        require(
            transition_phrase not in paper_text.lower(),
            f"paper contains migration narrative: {transition_phrase}",
        )

    numbered = sorted(path.name for path in (ROOT / "phases").glob("phase*.md"))
    expected = [f"phase{index}.md" for index in range(9)]
    require(numbered == expected, f"numbered phases differ: {numbered}")
    protocol = (ROOT / "phases" / "AUTONOMY_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    for requirement in (
        "Mandatory failure-repair loop",
        "Research before patching",
        "Formalize the repair",
        "Run all inherited gates",
        "Iterate",
    ):
        require(requirement in protocol, f"autonomy protocol missing {requirement}")
    for phase in numbered:
        content = (ROOT / "phases" / phase).read_text(encoding="utf-8")
        require("AUTONOMY_PROTOCOL.md" in content, f"{phase} omits shared protocol")
        require("PASS" in content, f"{phase} has no explicit pass gate")

    summary_path = ROOT / "analysis" / "results" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary["architecture"] == "AURELIS", "wrong numerical architecture")
    require(summary["seed"] == 20260828, "wrong primary seed")
    require(
        summary["algebra"]["max_residual_decomposition_abs_error"] < 1e-10,
        "residual identity gate failed",
    )
    require(
        summary["linear_reproduction"]["full_residual_l2_error"] < 1e-12,
        "linear reproduction gate failed",
    )
    require(
        summary["conditional_calibration"][
            "routed_relative_calibration_error"
        ]
        < 0.03,
        "routed calibration gate failed",
    )
    for artifact in (
        "analysis/results/NUMERICAL_REPORT.md",
        "analysis/results/exception_recall.csv",
        "analysis/results/bias_variance_sweep.csv",
        "analysis/results/conditioning_sweep.csv",
        "analysis/plots/exception_recall.png",
        "analysis/plots/bias_variance_sweep.png",
        "analysis/plots/uncertainty_calibration.png",
        "research/LITERATURE_REVIEW.md",
        "lean/PROOF_COVERAGE.md",
    ):
        require((ROOT / artifact).is_file(), f"artifact missing: {artifact}")

    lean_files = [ROOT / "lean" / "Aurelis.lean", *(ROOT / "lean/Aurelis").glob("*.lean")]
    require(len(lean_files) >= 6, "formalization modules are missing")
    forbidden = re.compile(r"\b(sorry|admit|axiom)\b")
    for lean_file in lean_files:
        text = lean_file.read_text(encoding="utf-8")
        require(not forbidden.search(text), f"proof placeholder in {lean_file}")

    report = """# AURELIS theory artifact audit

PASS

- Standalone paper present; superseded paper absent; no migration narrative.
- Literature/novelty review present.
- Primary numerical seed and algebra/calibration gates pass.
- Raw numerical tables and all three generated figures are present.
- Lean source has the expected modules and no proof placeholders/project axioms.
- Exactly nine numbered phases (`phase0.md` through `phase8.md`) exist.
- Every numbered phase imports the shared self-correction protocol and defines PASS gates.

This audit covers the requested theory, numerical, formal, paper, and phase-prompt
deliverables. It does not claim that Phase 0 implementation or later empirical
phases have executed.
"""
    output = ROOT / "analysis" / "results" / "THEORY_AUDIT.md"
    output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
