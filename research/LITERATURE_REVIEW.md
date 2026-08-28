# AURELIS literature review and design decision

_Research freeze: 2026-08-28._

## Question

How should an exact local-attention mechanism and a bounded recurrent
least-squares memory be combined so the result is more than alternating two
existing layer types?

## Closest architecture families

| Work | Mechanism relevant here | Boundary relative to AURELIS |
|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) | Global softmax key-value retrieval | Exact cache grows with context; no bounded remote posterior |
| [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258) (Ba et al., 2016) | Fast associative weights as an alternative view of attention | Hebbian-style fast weights; no delayed handoff or posterior router |
| [Transformers are RNNs](https://arxiv.org/abs/2006.16236) (Katharopoulos et al., 2020) | Recurrent kernel/linear attention | Does not invert the key covariance and has no exact local cache |
| [Based](https://arxiv.org/abs/2402.18668) (Arora et al., 2024/2025) | Global linear attention plus small sliding-window attention; establishes a state/recall tradeoff | Combines primitives architecturally; AURELIS instead couples them through residual values and posterior covariance |
| [RecurrentGemma / Griffin](https://arxiv.org/abs/2404.07839) (Botev et al., 2024) | Linear recurrence plus local attention with fixed decode state | Layer/block hybrid; remote state is not the exact ridge posterior used here |
| [Samba](https://arxiv.org/abs/2406.07522) (Ren et al., 2024) | Layerwise Mamba plus sliding-window attention | Motivates recent/remote complementarity; no within-head residual correction |
| [Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/abs/2406.06484) (Yang et al., 2024) | DeltaNet and hybrids with sliding-window or global attention | Online first-order update and layer interleaving rather than cumulative optimal ridge plus analytic fusion |
| [Gated Delta Networks](https://arxiv.org/abs/2412.06464) (Yang et al., 2024/ICLR 2025) | Data-dependent decay plus delta-rule memory; hybrid variants | Strong empirical comparator; different state objective and no conditional uncertainty certificate |
| [Test-time regression](https://arxiv.org/abs/2501.12352) (Wang, Shi, and Fox, 2025) | Unifies attention and recurrent memories as regression; derives higher-order softmax attention | Most important conceptual precursor. AURELIS uses a remote global slope as a control variate for local constant attention rather than solving a fresh local polynomial regression per query |
| [MesaNet](https://arxiv.org/abs/2506.05233) (von Oswald et al., 2025/ICLR 2026) | Cumulative least-squares state solved near-optimally with chunkwise conjugate gradient | Closest remote branch. AURELIS adds disjoint delayed handoff, exact local cache, residual coupling, and a closed-form covariance router |
| [A Systematic Analysis of Hybrid Linear Attention](https://arxiv.org/abs/2507.06457) (Wang et al., 2025) | Large controlled study of linear/full-attention ratios | Shows that the recurrent component and ratio matter; it does not test AURELIS's within-head estimator |
| [Native Hybrid Attention](https://arxiv.org/abs/2510.07019) (Du et al., 2025) | Recurrent slots and short-term tokens enter one softmax in a uniform layer | Closest same-layer hybrid; AURELIS instead keeps a solver posterior and cache disjoint and fuses predictions through residual uncertainty |
| [Kimi Linear](https://arxiv.org/abs/2510.26692) (Kimi Team, 2025) | Layerwise Kimi Delta Attention plus Multi-Head Latent Attention | Large-scale evidence for hybrid linear/full attention; different layerwise mechanism |
| [Rethinking the Role of Efficient Attention in Hybrid Architectures](https://arxiv.org/abs/2606.15378) (Qiao et al., 2026) | Reports that full-attention layers carry much long-range retrieval in studied hybrids | A warning against assuming a bounded remote state has solved arbitrary retrieval; motivates explicit adversarial recall gates |
| [Efficient Attention via Control Variates](https://arxiv.org/abs/2302.04542) (Zheng et al., 2023) | Uses control variates to reduce random-feature approximation error | Related estimator language, but targets approximation of global softmax rather than recent residuals over a remote posterior |

## Design selected

The architecture is named **AURELIS**: **Attention with Uncertainty-Routed
Residuals over an Episodic–Long-range Inference State**.

At boundary `t`, the last `w` associations remain in a softmax cache. Older
associations, and only older associations, enter a Bayesian ridge state

`P = alpha I + sum beta k k^T`, `C = sum beta v k^T`, `M = C P^-1`.

For attention weights `a`, key barycenter `kbar`, and value barycenter `vbar`,
the head emits

`y_g(q) = M q + g (vbar - M kbar)`.

The full-residual endpoint (`g=1`) is

`vbar + M(q-kbar)`.

It exactly reproduces a linear map when `M` is that map, regardless of the
softmax smoothing weights, and it exactly returns a one-hot cached exception.
Under a disjoint linear-Gaussian model, the conditional variance is a convex
quadratic in `g`; its minimizer is available in closed form and is clipped to
`[0,1]`. This produces a query-dependent, parameter-free Bayes gate. An
episodic responsibility can override the Bayes gate when the target is the
observed cached value rather than the latent denoised operator output.

## What is and is not a novelty claim

The broad idea “recurrent memory plus local/full attention” is established and
is not claimed as new. Least-squares sequence memory, test-time regression,
local polynomial attention, and control-variate attention also predate this
work. The proposed research object is the specific conjunction of:

1. occurrence-level delayed handoff, so the two estimators use disjoint data;
2. the residual formula `M q + g (vbar - M kbar)` inside one head;
3. the posterior covariance calculation including cross-covariance between
   the two endpoints;
4. the resulting clipped closed-form gate; and
5. an explicit separation between latent denoising and episodic-copy targets.

This repository provides derivations, numerical evidence, and partial formal
verification for that conjunction. It does not claim priority beyond the
survey above, benchmark superiority, or large-scale language-model evidence.

## Falsification priorities

- Learned keys may not admit a stable linear remote operator.
- A learned episodic gate may fail to distinguish “copy this observation” from
  “denoise the latent relation.”
- The `d x d` solve can erase the throughput benefit at practical head sizes.
- Delayed handoff can create optimization discontinuities at the cache edge.
- Fixed remote state cannot recall unbounded adversarial associations; the
  lower bound remains intact.
- Posterior variance is calibrated only under the declared model and must be
  treated as a routing feature under misspecification.
