# AURELIS

Solve-Free Recurrent Memory with Residual-Certified Retrieval.

AURELIS couples a solve-free gated delta recurrent state with local window attention and an optional exact archive governed by deterministic residual error bounds.

## Operating Contracts

AURELIS defines two explicit operating modes:

1. **Bounded Mode:** Fixed recurrent state $S \in \mathbb{R}^{d_v \times d_k}$ and recent attention cache of window $w$. Delivers fast transported estimates $r(q) = \bar{v}_L + S_t(q - \bar{k}_L)$ in $O(d_v d_k + w(d_k + d_v))$ without dense key-space matrix solves.
2. **Archive Mode:** Same recurrent working state coupled with paged storage of evicted observations. Selects exact observations and completes unread mass using the recurrent predictor, stopping when the deterministic error certificate $\le \epsilon$ or returning an explicit budget exhaustion signal. Full reads recover full softmax in real arithmetic.

## Repository Navigation

- **[Specification & Mathematics](aurelis.md):** Architectural design, mathematical equations (1)–(15), error bounds, and proofs.
- **[Primary Literature & Prior Art](research/LITERATURE_REVIEW.md):** Survey of hybrid architectures, recurrent memories, and novelty boundaries.
- **[Claim Registry](CLAIMS.md):** Formal theorem mapping and empirical hypothesis registry.
- **[Research Plan](RESEARCH_PLAN.md):** Phased falsification and evaluation roadmap.
- **[Implementation Contract](phases/IMPLEMENTATION_CONTRACT.md):** Module contracts and equation-to-code mapping.
- **[Research Phases](phases/README.md):** Detailed step-by-step phase execution contracts.
- **[Formal Core](lean/README.md):** Lean 4 / mathlib machine-checked proofs and [coverage ledger](lean/PROOF_COVERAGE.md).

## Formal Verification

AURELIS includes machine-checked proofs in Lean 4 (mathlib 4.19.0) under `lean/`:

```bash
cd lean
lake build
```

The formal core verifies:
- Deterministic finite-state recall capacity lower bounds (Eq. 1)
- Gated delta update query-linearity and unit-key write reproduction (Eq. 2)
- Rank-one perturbation energy identity and contraction stability (Eq. 5)
- Mass-consistent archive completion normalization and error identity (Eq. 8, 9)
- Deterministic residual error certificates under interval envelopes (Eq. 10)
- Coordinate key-box and page exponential envelope intervals (Eq. 11, 12)
- Causal cache handoff partitioning
