# Literature ledger and research decision

Primary-source search and inspection: 2026-09-16. This is a targeted review,
not an exhaustive priority search. No publication after this date is relied on.
Search families included attention/SSM recall capacity, gated delta kernels,
hybrid serving, sparse KV retrieval, residual completion, runtime-certified
attention, and adaptive block selection. Paper abstract/full-text pages and
official serving documentation were inspected; reported results were not
independently reproduced.

## Sources and design consequences

| Primary source | What it contributes | Consequence for AURELIS |
|---|---|---|
| [Based, 2402.18668](https://arxiv.org/abs/2402.18668) | Linear/local attention and recall–throughput tradeoffs | Fixed memory is a capacity choice, not unlimited exact recall |
| [Mamba-2 / SSD, 2405.21060](https://arxiv.org/abs/2405.21060) | Structured recurrence and hardware-efficient computation | Use structured kernels; generic associative algebra alone is insufficient |
| [Gated DeltaNet, 2412.06464v3](https://arxiv.org/abs/2412.06464v3) | Decay plus delta writes and chunkwise parallelism | Borrow the known recurrence explicitly; no novelty claim for equation (2) |
| [Infini-attention, 2404.07143](https://arxiv.org/abs/2404.07143) | Compressive memory with local attention | Same-block compressed/exact memory is prior art |
| [NHA, 2510.07019v3](https://arxiv.org/abs/2510.07019v3) | Recurrent slots and recent tokens in a shared attention operation | “Native same-layer hybrid” is not a distinctive claim |
| [Nemotron-H, 2504.03624](https://arxiv.org/abs/2504.03624) | Substantial trained hybrid family and inference comparisons | Useful hybrids exist; compare real optimized implementations |
| [Kimi Linear, 2510.26692v2](https://arxiv.org/abs/2510.26692v2) | KDA/MLA hybrid, specialized kernels, released model/serving work | Quality and systems design must be evaluated together |
| [Rethinking efficient attention, 2606.15378v1](https://arxiv.org/abs/2606.15378v1) | Retrieval and optimization roles in studied hybrids | Ablate recurrence's contribution instead of attributing full-attention retrieval to it |
| [MesaNet, 2506.05233](https://arxiv.org/abs/2506.05233) | Locally optimal regression memory with efficient solve techniques | Least-squares memory is prior art; retaining a solver needs measured justification |
| [Quest, 2406.10774](https://arxiv.org/abs/2406.10774) | Query-aware page selection using key minima/maxima | Borrow page boxes; include all unread mass in any certificate |
| [RetrievalAttention, 2409.10516](https://arxiv.org/abs/2409.10516) | Attention-aware indexed CPU KV retrieval | Host storage/index/transfer are real costs; ANN recall is not certification |
| [NSA, 2502.11089v2](https://arxiv.org/abs/2502.11089v2) | Trainable hardware-conscious sparse attention | Block sparsity must be reflected in training and actual kernels |
| [MoBA, 2502.13189](https://arxiv.org/abs/2502.13189) | Learned block routing and sparse/full transition | Sparse fallback and adaptive attention are established directions |
| [ResKV, 2607.29591v1](https://arxiv.org/abs/2607.29591v1) | Restores omitted numerator/denominator statistics within a cache budget | Mass-consistent completion itself is not new |
| [Runtime-certified quantized attention, 2605.20868v1](https://arxiv.org/abs/2605.20868v1) | Local attention-error bounds with retained-original fallback | Runtime certification and exact fallback are prior art, including limited scope |
| [Uncertainty-gated block selection, 2607.07724v1](https://arxiv.org/abs/2607.07724v1) | Expands sparse budgets using cutoff uncertainty | Budget adaptation alone is not a contribution; distinguish a deterministic value-sensitive bound |
| [Vashista Sparse Attention, 2602.13804v1](https://arxiv.org/abs/2602.13804v1) | Conditional sparse concentration under support-gap premises | Do not transfer conditional constant-support conclusions to arbitrary queries |
| [Stream, 2510.19875v2](https://arxiv.org/abs/2510.19875v2) | Hierarchical sparse tracing/pruning for long-context analysis | Hierarchical selection is prior art; its stated scaling does not prove ours |
| [FlashAttention-3, 2407.08608](https://arxiv.org/abs/2407.08608) | Hardware-aware exact attention | Baseline must use an appropriate efficient dense kernel |
| [GQA, 2305.13245](https://arxiv.org/abs/2305.13245) | Fewer shared KV heads | Match head sharing and dtype; full MHA cache comparisons alone are weak |
| [PagedAttention, 2309.06180](https://arxiv.org/abs/2309.06180) | KV allocation, paging and sharing for serving | Include fragmentation, prefix reuse, scheduling, and concurrency |
| [vLLM hybrid cache design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/) | Different group state sizes/lifetimes and prefix rules | Snapshot and rollback the complete hybrid state; pin a commit during implementation |
| [vLLM hybrid disaggregated serving, 2026-04-21](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) | Recurrent and attention state transfers across serving stages | Transfers/layouts are deployment work, not consequences of recurrence math |

Official documentation is mutable and was accessed on the review date; capture
the exact serving-source commit in phase 0. Search results sometimes carried
older alternate titles for arXiv 2510.19875; this ledger uses the verified
current primary-page title, not the search snippet's title.

## Bottlenecks and decisions

**Attention:** exact full reads retain history and consume KV bandwidth.
Efficient kernels, GQA, paging and compression improve constants and use of
memory; they do not make arbitrary history disappear. Sparse selection can
drop useful evidence, especially with diffuse attention or unusual values.

**Recurrence:** state interference, limited capacity, forgetting, trained
representation mismatch, and finite precision affect recall. Efficient decode
does not automatically yield efficient training or prefix reuse. Our stable
delta transition proof is a fixed-input perturbation statement, not a theorem
of semantic memory.

**Hybrids:** kernels/layouts, recurrence checkpointing, KV lifetime differences,
prefix reuse, speculative rollback, batching, transfer, and long-tail retrieval
all matter. A successful architecture can still be badly implemented.

**AURELIS:** dense key-space matrix factorizations and quadratic masked-local
prefill are avoided by adopting solve-free gated delta updates and true local
window attention; whether the recurrent predictor preserves useful quality
remains the empirical question to evaluate.

## Closest comparisons for novelty

| Proposed component | Already established | What must be distinguished experimentally |
|---|---|---|
| Delta state plus recent exact tokens | Gated DeltaNet, Based, Infini-attention, NHA | Delayed transport plus completion must beat simple fusion |
| Missing-mass numerator/denominator completion | ResKV | Current recurrent predictor and certified refinement must add value |
| Query-aware paged exact reads | Quest, RetrievalAttention | Value-sensitive selection must improve actual service cost at matched error |
| Runtime error bound with full fallback | Certified quantized attention | The recurrent residual/normalizer bound must enable a useful operating region |
| Adaptive sparse budget | Uncertainty-gated selection, sparse routing work | A bound-driven stop must beat fixed/heuristic budgets including overhead |

The new synthesis is a **candidate**, not a proved priority claim. The paper's
error identity follows elementary normalized-estimator algebra; formalizing it
does not make that algebra a new invention. Required evidence is the benefit of
the specific integrated design, especially against per-page center completion.
If recurrence-free completion is better, publish that result or redesign the
hypothesis; do not preserve recurrence solely to keep the project name.
