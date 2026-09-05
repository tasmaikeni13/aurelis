# Phase 6 — Language-Model Viability & Comparative Publication Gate

Start only after Phase 5 PASS. Read all prior evidence and
`phases/AUTONOMY_PROTOCOL.md`. This phase establishes natural language modeling viability
and rigorous comparative benchmarks for publication across three distinct architectural
candidates at both 125M and 350M parameter scales. Execute the failure-repair loop until PASS.

## Three Publication Candidate Architectures

For peer-reviewed publication and definitive architectural comparison, implement and
evaluate three matched candidates:

1. **AURELIS (Candidate 1)**:
   - Same-head dual-memory architecture combining exact causal sliding-window attention
     ($w \in \{64, 128\}$) with a delayed Bayesian ridge regression state.
   - Computes local barycenters $\bar{k}(q)$ and $\bar{v}(q)$, remote linear mapping $Mq = CP^{-1}q$,
     and uncertainty-routed innovation residuals $[ \bar{v} - M\bar{k} ]$.
   - Evaluates both **AURELIS-B** (exact Bayes gate $g_B$ derived with cross-covariance $K_{RH}$)
     and **AURELIS-E** (with calibrated episodic override $g_E = \max(g_B, e_t)$).
   - Inference decode state: strictly constant $O(d_k^2 + d_v d_k + w(d_k + d_v))$ per head.

2. **Modern Causal Transformer (Candidate 2 — Pure Attention Baseline)**:
   - Standard modern causal decoder-only transformer (LLaMA/Mistral style).
   - Rotary Position Embeddings (RoPE), Pre-RMSNorm, causal multi-head self-attention,
     and SwiGLU feedforward MLP ($d_{\text{ffn}} = \frac{8}{3} d_{\text{model}}$).
   - Standard $O(L)$ growing KV-cache at inference and $O(L^2)$ training complexity.

3. **Strong SSM + Attention Hybrid (Candidate 3 — State-of-the-Art Hybrid Baseline)**:
   - High-performance interleaved State Space Model (SSM) + Causal Multi-Head Attention hybrid
     (following modern Samba / Jamba / RecurrentGemma literature).
   - Alternates selective state-space recurrent blocks (Mamba-2 style input-dependent selection
     with 1D causal depthwise convolution and SiLU gating) with causal multi-head attention blocks.
   - Pre-RMSNorm and SwiGLU MLP feedforward networks.
   - Represents the strongest published competitive hybrid paradigm.

## Dual Model Capacity Targets: 125M and 350M

Preregister and implement both parameter scales to validate small-scale viability and
medium-scale scaling readiness:

- **125M Scale**:
  - $d_{\text{model}} = 768$, $\text{heads} = 12$, $\text{layers} = 12$, $d_k = 64$, $d_v = 64$.
  - Vocabulary: $50,257$ (GPT-2 / FineWeb-Edu standard).
  - Context length: $2048$ tokens.
- **350M Scale**:
  - $d_{\text{model}} = 1024$, $\text{heads} = 16$, $\text{layers} = 24$, $d_k = 64$, $d_v = 64$.
  - Vocabulary: $50,257$ (GPT-2 / FineWeb-Edu standard).
  - Context length: $2048$ tokens.

Parameters across all three candidates must be calibrated within $\pm 3\%$ at each scale.

## Implementations & ROCm/HIP Optimizations

Provide complete, self-contained modular implementations in `src/aurelis/models/`:

- `config.py`: Standardized model configuration dataclasses for 125M and 350M scales.
- `transformer.py`: Causal Transformer with RoPE, Pre-RMSNorm, SwiGLU, and KV cache.
- `hybrid_ssm.py`: Strong SSM + Attention Hybrid with selective scan and attention blocks.
- `aurelis_lm.py`: Full AURELIS Language Model with sliding cache, delayed Bayesian state,
  innovation residual routing, Pre-RMSNorm, SwiGLU, and constant-state decoding.
- `hip_kernels.py`: Accelerated HIP / ROCm kernels and fused operators targeting AMD Instinct
  MI300X (`gfx942`) for recurrent state updates, fast associative scans, and fused gating,
  with transparent PyTorch eager/Triton fallback.

## Diagnostic & Natural Language Benchmark Suite

Evaluate all three candidates at both scales across:

1. **Multi-Query Associative Recall (MQAR)**: Key-value retrieval under distractor loads.
2. **Cache-Boundary & Recent Copy**: Exact copy within local cache and across eviction boundary.
3. **Episodic Exception vs Latent Denoising**: Memorized exception recovery vs structured relation.
4. **Induction & Selective Copy**: Long-distance prefix pattern completion.
5. **Multi-Hop Pointer Chains**: Mixed recent/remote pointer chasing.
6. **Passkey Retrieval / Needle-In-A-Haystack**: Needle retrieval at extended contexts (up to 4096).
7. **FineWeb-Edu Language Modeling**: Validation perplexity and loss convergence.
8. **Systems Profiling (AMD Instinct MI300X)**:
   - Prefill throughput (tokens/second) across sequence lengths $\{512, 1024, 2048, 4096\}$.
   - Per-token decode latency (ms/token).
   - Peak VRAM allocation during training and inference.
   - Active decoding memory footprint ($O(1)$ constant state for AURELIS vs $O(L)$ for Transformer).

## PASS Gates

- All three architectures (AURELIS, Transformer, SSM + Attention Hybrid) are fully implemented,
  calibrated at both 125M and 350M scales, and pass all parameter accounting and gradient checks.
- Accelerated HIP/ROCm kernels compile and pass numerical validation against CPU/fp64 references
  with maximum absolute error $< 10^{-5}$ in float32.
- AURELIS achieves competitive validation perplexity on FineWeb-Edu token distributions within
  the preregistered margin of the Transformer and SSM-Attention Hybrid.
- AURELIS demonstrates a statistically significant matched-parameter advantage on targeted mixed
  recent/remote diagnostics (associative recall / pointer chasing).
- AURELIS-E strictly improves exact exception recall over AURELIS-B without degrading latent relation accuracy.
- Systems benchmarks confirm that AURELIS maintains constant $O(1)$ decoding state memory at inference,
  demonstrating a decisive memory advantage over Transformer at context lengths $\ge 2048$.
- All unit, regression, and property tests pass cleanly (`pytest`).
- Generated `results/phase6/PASS.md` satisfies the shared PASS record with full reproduction logs.

