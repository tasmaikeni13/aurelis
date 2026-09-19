# AURELIS Phase 4: Reproducible Counterexamples Outside Useful Regime

| Case | Failure Mechanism | Mathematical Cause | Empirical Consequence |
|---|---|---|---|
| **Counterexample 1: Incompressible Associations (Capacity Lower Bound)** | Random Gaussian keys and values with independent coordinates | Recurrent state S cannot linearly compress uncorrelated vectors (Thm 1.1). Both AURELIS and per-page center read all pages. AURELIS incurs extra S-compute FLOPs without saving fetches. | `Recurrent predictor provides 0% page reduction.` |
| **Counterexample 2: Metadata-Dominant Contexts (Overhead Exceeds Dense Attention)** | Short context lengths (t <= 24) or tiny page sizes (p = 2) | The FLOP cost of scanning page envelopes, calculating bounds, and sorting unread nodes exceeds 100% of dense FlashAttention/GEMM FLOPs. | `Total FLOPs for certified archive read exceeds dense causal attention baseline.` |
| **Counterexample 3: Abrupt Drift Before Adaptation** | Generating matrix switches from W1 to W2 halfway through context | Queries targeting pre-drift tokens suffer larger residuals under S (which decayed and adapted to W2) than static page envelopes. | `Recurrent prediction increases residual bound b_j(r) over static page center c_j.` |
| **Counterexample 4: Loose Page Envelopes** | Coordinate bounding boxes inflated by 2.0x | Score intervals widen, causing mass uncertainty eta to dominate denominator floor. Certified stopping cannot fire, forcing full reads. | `100% full read fallback rate under loose bounding boxes.` |
