# AURELIS: Solve-Free Recurrent Memory with Residual-Certified Retrieval

Research specification.

**Status:** Architecture specification and mathematical analysis, with machine-checked Lean 4 formal proofs. Implementation and evaluation follow [the phase contracts](phases/README.md).

## Abstract

Attention trades growing storage and reads for query-dependent access to past
observations. Finite-state recurrence trades that access for bounded working
memory. Neither a better solver nor a learned gate removes this tradeoff.
We propose AURELIS, a solve-free recurrent predictor coupled to local
attention and an optional exact archive. The recurrent branch uses an existing
gated delta update; it is not presented as a new SSM. Local attention supplies
a residual-transport prediction. In archive mode, selected exact observations
and a prediction for unread observations share a normalization. We derive a
deterministic error bound accounting for both unread value residuals and
uncertainty in their softmax mass. Retrieval continues until the bound meets a
tolerance or the service reports that its budget is insufficient. Reading all
observations recovers full softmax on the same query, keys, and values in real
arithmetic.

The research hypothesis is that recurrence reduces the exact data needed at
a specified error and latency budget. This is not an empirical finding or an
established novelty claim. We provide a finite-state recall lower bound,
perturbation stability conditions, a read algorithm, training objectives,
cost accounting, and falsification gates. Lean checks the finite-state bound,
delta transition energy, vector completion identity, and conditional norm
certificate. Page construction, floating-point interval kernels, model quality,
and serving correctness remain implementation obligations.

## 1. The problem that can actually be solved

The goal is a measurable quality–latency–storage improvement on specified
workloads, with explicit behavior when assumptions fail. There is no credible
universal “last hybrid you will ever need.” AURELIS eliminates per-read dense
matrix inversions, provides recoverable exact evidence when needed, and bounds
observable approximation error.

Three meanings of “exact” must remain separate:

1. A cache retains particular encoded observations.
2. A read equals full softmax over those observations in real arithmetic,
   or meets a declared numerical tolerance in code.
3. A model emits the desired fact or string.

Neither the first nor the second implies the third. Finite-temperature softmax
usually mixes several values. Matching one head does not certify a generated
answer.

### 1.1 Finite-state impossibility

Fix n distinct addresses and an alphabet of m possible values. A deterministic
encoder answering every address exactly for every assignment must map the
m^n assignments injectively to its state space. Otherwise two assignments
share a state but differ at a queried address. Hence

$$
|\mathcal S|\ge m^n,\qquad b\ge n\log_2m                 \tag{1}
$$

for a state with at most 2^b possibilities. Capacity.lean proves injectivity
and the cardinality inequality; the bit interpretation is an analytic
corollary. This is a worst-case finite-state deterministic statement, not a
bound for every structured or probabilistic task. Unlimited-precision real
numbers are not physically bounded memory.

Archiving raw tokens and recomputing activations can replace KV storage, but
token storage, model version, and replay work still count. A confidence score
cannot restore discarded arbitrary information.

### 1.2 Architectural principles

AURELIS enforces three core architectural invariants:

1. **Solve-Free Linear Complexity:** The recurrent state update and bounded read must not perform matrix inversions, factorizations, or key-space solves. Update and bounded read complexities scale as $O(d_v d_k + w(d_k + d_v))$.
2. **Disjoint Causal Partitioning:** Active window cache tokens and evicted recurrent tokens are strictly partitioned. Evicted observations update the recurrent state exactly once.
3. **Mass-Consistent Residual Certification:** When exact archive retrieval is engaged, unread history is represented through predicted mass with deterministic error certificates, guaranteeing recovery of full attention when all observations are fetched.

## 2. Prior art and novelty boundary

Hybrids are already useful: [Nemotron-H](https://arxiv.org/abs/2504.03624) and
[Kimi Linear](https://arxiv.org/abs/2510.26692) report substantial hybrid-model
results. Their gains are conditional on their workloads and kernels and do
not transfer to AURELIS.

[Mamba-2/SSD](https://arxiv.org/abs/2405.21060) connects structured recurrence
to efficient matrix computation. [Gated DeltaNet](https://arxiv.org/abs/2412.06464)
supplies our update and a chunkwise training precedent.
[Based](https://arxiv.org/abs/2402.18668),
[Infini-attention](https://arxiv.org/abs/2404.07143), and
[Native Hybrid Attention](https://arxiv.org/abs/2510.07019) already combine
compressed memory with exact token access. The June 2026 study
[Rethinking the Role of Efficient Attention](https://arxiv.org/abs/2606.15378)
finds that full attention carries much long-range retrieval in its studied
hybrids. This motivates testing branch contributions, not dismissing recurrence.

[Quest](https://arxiv.org/abs/2406.10774) uses query-dependent page bounds;
[RetrievalAttention](https://arxiv.org/abs/2409.10516) uses indexed host memory;
[NSA](https://arxiv.org/abs/2502.11089) and
[MoBA](https://arxiv.org/abs/2502.13189) make sparse computation an architectural
and kernel concern. [ResKV](https://arxiv.org/abs/2607.29591) reconstructs omitted
numerator and denominator contributions. These are close precedents.

[Runtime-Certified Bounded-Error Quantized Attention](https://arxiv.org/abs/2605.20868)
already proposes local bounds with exact-data fallback. The July 2026
[uncertainty-gated block selector](https://arxiv.org/abs/2607.07724) adapts a
sparse budget using cutoff uncertainty. We do **not** claim to invent hybrids,
delta updates, residual caches, paging, adaptive retrieval, error certificates,
or exact fallback.

The candidate contribution is the coupling of a delayed recurrent transport
predictor, mass-consistent unread-history completion, and a value-sensitive
deterministic stopping rule, trained and evaluated by the cost of a certified
read. Priority is unestablished. Scientific value requires beating
recurrence-free completion and certified sparse attention at matched total
resources. Combining ingredients alone is insufficient.
The [literature ledger](research/LITERATURE_REVIEW.md) records the comparisons.

## 3. State, causality, and two operating contracts

Condition on causal projected associations (k_i,v_i), 1≤i≤t, where k_i,q are
in R^{d_k} and v_i in R^{d_v}. Queries and keys use a common coordinate space
for transport. The reference score is s_i=κ qᵀk_i with κ>0. Positional
transformations or biases, if added, must be represented in both the reference
and every bound. The initial specification uses this dot-product score.

The recent set is L_t={max(1,t-w+1),…,t}, w≥1; remote indices are
R_t={1,…,max(0,t-w)}. State consists of S_t in R^{d_v×d_k}, a ring of w
observations and their write gates, and optionally a paged archive of remote
observations. Occurrence IDs, not content equality, define membership.

At a causal boundary: construct the association and gates, append to the ring,
hand the newly evicted observation to recurrence once, append it to the
archive if enabled, then read. Store each write gate with its observation at
the original position. Reading an archive page never repeats a recurrent
write. Padding, document resets, packed examples, and branches respect these
boundaries.

| Contract | Persistent state | Output obligation |
|---|---|---|
| bounded | Fixed matrix and recent ring | Fast transported estimate; no general full-history softmax guarantee. |
| archive | Same working state plus remote KV and index | Valid local error bound, full-read result, or explicit budget failure. |

The archive is a redundant exact backup of observations compressed into S.
Only recent/remote observations and selected/unread attention sets are
disjoint. Neither a state summary nor a previously fetched page is inserted
as a second exact copy in the softmax denominator.

## 4. Solve-free recurrent transport

For an evicted association (k,v,α,β), use

$$
\widetilde S=\alpha S,\qquad
S^+=\widetilde S+\beta(v-\widetilde S k)k^\top,\qquad
0\le\alpha,\beta\le1,\quad \|k\|_2\le1.                \tag{2}
$$

Normalize keys with a nonzero norm floor, allowing zero keys. This changes
expressivity and belongs in all comparisons. Start with S_0=0. No precision
matrix, inverse, Cholesky factor, or posterior variance appears. Equation (2)
provides an efficient solve-free gated delta update.

Local softmax gives k̄_L=Σa_i k_i and v̄_L=Σa_i v_i. The bounded read and
archive's initial completion predictor are

$$
r(q)=\bar v_L+S_t(q-\bar k_L).                        \tag{3}
$$

This uses one matrix-vector read and local attention. An independently learned
transport gate is an ablation, not Bayesian uncertainty; the base specification
has no such gate. For any linear truth W,

$$
r(q)-Wq=(\bar v_L-W\bar k_L)+(S_t-W)(q-\bar k_L).      \tag{4}
$$

Exact S_t=W and linearly consistent local values imply r(q)=Wq.
A one-hot local hit with q=k_j gives r(q)=v_j. These are conditional identities,
not assertions that training produces W or that finite softmax is one-hot.
ResidualCorrection.lean proves them for arbitrary linear maps. They apply
to the bounded predictor; archive completion has a different output.

### 4.1 Perturbation stability

Two states receiving the same input and gates have difference
D⁺=αD(I-βkkᵀ). For each row error x,

$$
\|x-\beta k(k^\top x)\|^2
=\|x\|^2-\beta(2-\beta\|k\|^2)(k^\top x)^2.           \tag{5}
$$

Thus β≥0 and β||k||²≤2 give a nonexpansive row transition and
||D⁺||_F≤α||D||_F. DeltaMemory.lean checks the row identity and contraction;
summing rows gives the matrix corollary analytically. This does not prove
bounded driven state, gradients through gates, or bounded accumulated rounding.
If α≤a<1, ||v||≤V, and ||k||≤1, a geometric-series argument gives

$$
\|S_n\|_F\le a^n\|S_0\|_F+V(1-a^n)/(1-a).            \tag{6}
$$

At a=1 this supplies no uniform-in-time guarantee. Test zero, repeated, and
nearly parallel keys, nonfinite states, and long streams at production dtype.

### 4.2 Training computation

Let A_i=α_i(I-β_i k_i k_iᵀ) and B_i=β_i v_i k_iᵀ. Then
S_i=S_{i-1}A_i+B_i. Composing updates gives
(A_1A_2,B_1A_2+B_2). Associativity permits chunk algorithms; it does not
make arbitrary dense matrix scans cheap. Scalar affine scan composition alone
does not prove this matrix algorithm.

Use a structured delta chunk algorithm with chunk checkpoints and backward
recomputation. “Solve-free” means no per-token key-space matrix inversion or
factorization. Some chunk algorithms use small unit-triangular operations; count their costs.
A Python loop, dense d_k³ transition products, or all-prefix state tensors
are not acceptable production training paths.

## 5. Mass-consistent archive completion

Archive mode targets ordinary full softmax on **the same current Q/K/V**,
not equation (3) or an independently executed Transformer:

$$
y_*=\frac{\sum_{i\le t}e^{s_i}v_i}{\sum_{i\le t}e^{s_i}}. \tag{7}
$$

Let A include all recent occurrences and fetched remote pages; O is its exact
complement. Write

$$
Z_A=\sum_{i\in A}e^{s_i}>0,\quad N_A=\sum_{i\in A}e^{s_i}v_i,\quad
Z_O=\sum_{i\in O}e^{s_i},\quad N_O=\sum_{i\in O}e^{s_i}v_i.
$$

Maintain certified 0≤L_O≤Z_O≤U_O and choose Ẑ_O=(L_O+U_O)/2.
Freeze r=r(q) during refinement and compute

$$
\widehat y_A=\frac{N_A+\widehat Z_O r}{Z_A+\widehat Z_O}. \tag{8}
$$

This is normalized completion. The coefficient of r is a mass estimate, not a
confidence probability. If O is empty, Ẑ_O=0 and the output is full attention,
independent of S. Bounded mode emits (3) directly; do not invent mass estimates
without summaries. Empty histories use a start token or a separate zero-output
convention; certificate statements require Z_A>0.

## 6. Residual and normalizer certificate

Let R_O=N_O-Z_O r and η=(U_O-L_O)/2. Multiplying (8) by its denominator gives

$$
(Z_A+Z_O)(y_*-\widehat y_A)
=R_O+(Z_O-\widehat Z_O)(r-\widehat y_A).               \tag{9}
$$

If ||R_O||≤B_O, triangle inequality and the mass interval give

$$
\boxed{\|y_*-\widehat y_A\|_2\le
\mathcal E_A=
\frac{B_O+\eta\|r-\widehat y_A\|_2}{Z_A+L_O}.}        \tag{10}
$$

The denominator is positive. Omitting the normalizer term is invalid even
when the predictor is accurate on average. CertifiedRead.lean proves the
vector identity and conditional norm bound over real normed spaces.

### 6.1 Computable page envelopes

A sealed page j stores count n_j, coordinatewise bounds k_j⁻≤k_i≤k_j⁺,
a value center c_j (initially the arithmetic mean), and outward radius
ρ_j≥max_i||v_i-c_j||. Update a partial page's summaries on every append or
keep it exact until sealed. Define

$$
\ell_j=\kappa\sum_d\min(q_d k^-_{jd},q_d k^+_{jd}),\qquad
u_j=\kappa\sum_d\max(q_d k^-_{jd},q_d k^+_{jd}),       \tag{11}
$$
$$
L_j=n_j e^{\ell_j},\quad U_j=n_j e^{u_j},\qquad
b_j(r)=U_j(\|c_j-r\|+\rho_j).                         \tag{12}
$$

Every key lies in its box, so L_j≤Z_j≤U_j. Triangle inequality yields
||Σ_{i∈j}e^{s_i}(v_i-r)||≤b_j. Sum unread-page bounds for L_O,U_O,B_O.
The exponential interval and weighted residual norm step are Lean-checked;
the multidimensional box constructor and its tensor implementation are not.

A hierarchy may store enclosing boxes/balls and subtree counts. Its frontier
must disjointly cover every unread occurrence. Intersect child and parent
bounds when helpful. Tighter mass bounds do not by themselves imply a
monotonically decreasing final certificate because the estimated output changes.
A flat page scan is the correctness reference. ANN candidates may order reads;
an ANN miss cannot exclude a subtree from the bound.

Summaries describe immutable raw values, not residuals under a stale state.
Changing S changes r; recompute ||c_j-r|| or a conservative subtree bound.
A stored v_i-S_write k_i cannot be treated as v_i-S_t k_i without correction.

### 6.2 Algorithm and budget semantics

For each query in archive mode:

1. Compute r, exact recent sums, and a covering archive frontier.
2. Evaluate (8) and an outward-rounded (10).
3. If the bound plus arithmetic error is at most absolute tolerance ε, return
   output, bound, certified status, bytes read, and pages read.
4. Otherwise refine a node or fetch an unread leaf. Initial priority is
   [b_j+((U_j-L_j)/2)||r-ŷ_A||]/estimated_service_cost_j.
   This heuristic has no optimality or monotonic-improvement guarantee.
5. Update selected sums and the disjoint cover; recompute the bound. Retain
   the best prior candidate if a new bound increases.
6. A full read returns full attention with its numerical allowance. An earlier
   resource limit returns budget_exhausted and the best bound. A permissive
   service may return an explicitly uncertified estimate; a strict one fails
   or escalates. Timeout is never interpreted as certification.

Fetching each remaining leaf once terminates at full attention for finite
history if resources permit. Group read rounds into fixed shape buckets but
validate each row/head. Missing pages, invalid summaries, corrupt IDs, archive
failures, or cancellation invalidate the requested guarantee.

### 6.3 Why recurrence may help, and how to refute that claim

If unread values are close to r, their residual is small even when unread
attention mass is large. A low-mass page with a large exceptional value can
matter, motivating value-sensitive refinement.

The bound is conservative. If all values and r equal c, B_O=0 and
r=ŷ_A=c, so no archive values need be fetched. Heterogeneous values and loose
boxes can instead force full reads. A recurrent predictor can worsen the bound.

Compare r to zero, the local value barycenter, and a summary-based predictor
under identical indexing and cost accounting. Also compare per-page completion
p_j=c_j:

$$
\widehat y=\frac{N_A+\sum_j\widehat Z_jp_j}{Z_A+\sum_j\widehat Z_j},\qquad
\|y_*-\widehat y\|\le
\frac{\sum_j[B_j+\eta_j\|p_j-\widehat y\|]}{Z_A+\sum_jL_j},              \tag{13}
$$

where B_j≥||N_j-Z_jp_j|| and η_j bounds mass-estimation error.
For midpoint estimates and p_j=c_j, B_j=U_jρ_j suffices. Summing the numerator
identity proves (13) analytically; it is not presently Lean-checked. If this
cheaper predictor wins, retire the recurrent-completion novelty claim.
Recurrence is not entitled to a performance advantage by definition.

### 6.4 Floating point and the scope of safety

These equations use real arithmetic. Use a common log-sum-exp shift for exact
and bounded contributions, rescaling all sums/intervals when the shift changes.
Underflow must not round a positive upper bound to zero. Overflow, NaNs, or
invalid intervals trigger failure or full-read fallback. Outward rounding is
needed for endpoints, dot products, exp, norms, reductions, and division.
Empirical fp64 agreement does not certify a BF16/FP32 implementation.

Declare the reference encoding. FP16 originals are exact relative to those
encoded values, not hypothetical pre-quantization reals. Additional compression
needs key/value perturbation bounds or retained originals. If arithmetic adds
at most δ_num, the stop rule is E_A+δ_num≤ε. Arbitrary epsilon padding is not
a proof. The floating-point certificate remains unimplemented.

The bound is per head/layer/current Q/K/V. Earlier approximation changes later
queries and cached keys. Full attention on the current archive does not recover
an independently run dense model's whole trajectory. With separately justified
layer Lipschitz constants L_l and local errors δ_l one can propagate
E_{l+1}≤L_l E_l+δ_l; without those premises no global logit bound follows.
Trajectory-level fallback may require replay from an exact checkpoint.
This is not a certificate of factual correctness, security, or answer safety.

## 7. Training specification

Train the streaming recurrence, delayed writes, and actual read. Initially
use separate declared bounded and archive configurations; do not silently
switch the contract on an untrained checkpoint. Randomize archive read budgets
in training. For sampled queries/selected sets, compute a full-attention
teacher on the same Q/K/V. A proposed loss is

$$
\mathcal L=\mathcal L_{LM}
+\lambda_r\|r-\operatorname{stopgrad}(N_O/Z_O)\|^2
+\lambda_c\|\widehat y-y_*\|^2
+\lambda_b\widetilde{\mathcal E}_A
+\lambda_{io}\widehat{\mathrm{bytes}}.                \tag{14}
$$

Skip the unread-target term when O is empty. The differentiable bound surrogate
is never a runtime certificate. Calibrate the cost surrogate on measured
hardware. Normalize/weight losses so collapsed values or reduced output scale
cannot win while harming LM quality. Count teacher and extra backward work.
Because r does not see the final unread set, its target is a distribution over
subset budgets; this mismatch is a weakness to test.

Start with fixed selection or stop-gradient discrete routing. Any
straight-through estimator is explicitly biased. Dense teachers on every query
reintroduce quadratic training: sample them and disclose the fraction.
Compare equal training FLOPs/wall time as well as tokens. Auxiliary regression
cannot substitute for real LM training and held-out evaluation.

## 8. Systems costs and deployment obligations

Per head, bounded persistent state is O(d_v d_k+w(d_k+d_v)); mixing work per
token is O(d_v d_k+w(d_k+d_v)), excluding projections/MLPs. Eliminating the
precision matrix and per-token factorization is a design-level reduction,
not a measured speedup. State traffic and small operations can remain costly.

For page size B, p≈t/B pages, m visited summaries, and f fetched pages,
archive read work is approximately

$$
O(d_vd_k+w(d_k+d_v)+m(d_k+d_v)+fB(d_k+d_v)).           \tag{15}
$$

Selection, synchronization, sorting, and queueing add costs. Flat summary
scanning has m=Θ(t/B), not constant cost. A hierarchy has no worst-case
sublinear guarantee. Raw archive storage is Θ(t(d_k+d_v)); leaf metadata is
Θ((t/B)(d_k+d_v)). Keeping every summary on device makes HBM grow. A bounded
hot-index cache needs host traversal/transfers, which must be measured.
Count prefix checkpoints, speculative branches, allocator padding, page tables,
transfer buffers, and archive writes too. Full-read fallback can use bounded
staging buffers but remains linear in history for each query. Invoking it at
every prefill position restores quadratic total work.

[FlashAttention-3](https://arxiv.org/abs/2407.08608) makes optimized dense
attention a serious baseline. [GQA](https://arxiv.org/abs/2305.13245) reduces KV
heads; [PagedAttention](https://arxiv.org/abs/2309.06180) addresses allocation and
sharing. Use comparable KV head sharing and report both state and total
device/host memory. Naive dense code is not an adequate industry baseline.

The [vLLM hybrid cache design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
documents differing cache layouts/lifetimes; its
[hybrid disaggregated serving work](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg)
addresses transfers of recurrent and attention state. An AURELIS prefix
checkpoint includes S, the ring and gates, archive length, index version, and
position. Rejecting speculative tokens restores that tuple, not just KV length.
Use immutable shared pages and copy-on-write mutable state.

Validate populated-cache prefill/decode equivalence, continuous batching,
resets, branching, cancellation, prefix hits, rollback, and archive recovery.
Report time to first token, inter-token latency, throughput at a fixed SLO,
p50/p95/p99 latency under load, memory by tier, and full-read/budget-failure
rates. A favorable mean kernel time with bad fallback p99 fails deployment.

## 9. Falsifiable research program

The research program is specified across [Phases 0–9](phases/README.md),
defining implementation and evidence gates. Compare modern dense GQA attention;
an optimized recurrent and layerwise hybrid baseline; the same delta/local
branches with simple fusion; sparse/archive attention without recurrence; and
per-page completion (13).

Sweep context, batch/concurrency, dimensions, window/page sizes, tolerance,
bytes fetched, and archive locality. Use trained models, paired seeds,
held-out data, confidence intervals, and measured cache tensors. Test old
random associations, diffuse attention, value outliers, repeated keys, late
disambiguation, multi-hop recall, drift, and boundary handoff. One needle is
insufficient evidence of general memory.

The main hypothesis survives only if recurrence reduces measured service cost
at fixed accepted error and task quality against the strongest nonrecurrent
completion, and the resulting design gives a useful Pareto improvement against
dense/hybrid baselines. Preregister tolerances/margins before evaluation.
If summary scans, loose bounds, or fallback rates consume the gain, preserve
the negative result and stop scaling that design.

## 10. Formal coverage and limitations

[The proof ledger](lean/PROOF_COVERAGE.md) is authoritative. Machine-checked
modules cover Capacity, DeltaMemory, CertifiedRead, and PageEnvelope, alongside
handoff and transport lemmas.

This contribution is a derivation, formal core, and implementation specification.
It does not solve lossless compression of unlimited arbitrary history, prove
that certificates will be tight on learned values, establish novelty, or
demonstrate an industrial speedup. Those boundaries determine the research
gates; a Lean build does not satisfy them.
