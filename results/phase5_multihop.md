# Phase 5 multi-hop functional graphs

## Gate decision: PASS

- PASS: `controlled_codes_reproduce_multihop_behavior`
- PASS: `controlled_h16_vector_error_is_small`
- PASS: `error_propagation_bound_holds`
- PASS: `operator_amplification_counterexample_is_visible`

| Gate measurement | Observed |
|---|---:|
| controlled minimum success rate, all H | 1.000000 |
| controlled maximum H=16 vector error | 1.600e-07 |
| maximum relative excess over propagation bound | 7.972e-09 |
| maximum many-to-one read-operator norm | 3.000 |

## Controlled pointer chasing

Each state stores a complete functional graph `node -> successor(node)`. The start code is read adaptively and the output becomes the next query, while `S` and `C` remain unchanged. Nearest-code success and vector error are both reported because finite epsilon can shrink a vector without changing its decoded node.

| graph | method | H=1 success | H=16 success | H=16 error |
| --- | ---: | ---: | ---: | ---: |
| permutation | csm_chained | 1.000 | 1.000 | 1.600e-07 |
| permutation | softmax_repeated | 1.000 | 1.000 | 0.000e+00 |
| permutation | softmax_one | 1.000 | 0.257 | 1.326e+00 |
| many_to_one | csm_chained | 1.000 | 1.000 | 1.600e-07 |
| many_to_one | softmax_repeated | 1.000 | 1.000 | 0.000e+00 |
| many_to_one | softmax_one | 1.000 | 0.417 | 9.723e-01 |

Repeated softmax is a strong equal-access baseline and often succeeds on these easy controlled codes. Its distinction is structural: H adaptive accesses require H attention layers, whereas the tested CSM layer exposes H reads against one maintained state. One softmax access has one adaptive round and therefore generally cannot produce an H-hop target for `H>1`.

## Error accumulation and cause attribution

| geometry | capacity | epsilon | H=16 success | H=16 error | operator norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| orthogonal | at_or_below_capacity | 1e-08 | 1.000 | 1.600e-07 | 1.725 |
| orthogonal | at_or_below_capacity | 0.1 | 1.000 | 7.824e-01 | 1.568 |
| random | at_or_below_capacity | 1e-08 | 1.000 | 1.994e-06 | 10.611 |
| random | at_or_below_capacity | 0.1 | 0.615 | 9.636e-01 | 2.694 |
| correlated_0.8 | at_or_below_capacity | 1e-08 | 1.000 | 3.629e-06 | 8.734 |
| correlated_0.8 | at_or_below_capacity | 0.1 | 0.319 | 4.023e-01 | 1.364 |
| correlated_0.98 | at_or_below_capacity | 1e-08 | 1.000 | 9.251e-06 | 8.627 |
| correlated_0.98 | at_or_below_capacity | 0.1 | 0.175 | 1.481e-01 | 0.996 |
| random | over_capacity | 1e-08 | 0.177 | 9.985e-01 | 4.358 |
| random | over_capacity | 0.1 | 0.188 | 9.997e-01 | 2.570 |
| correlated_0.98 | over_capacity | 1e-08 | 0.257 | 1.326e-01 | 4.196 |
| correlated_0.98 | over_capacity | 0.1 | 0.031 | 1.384e-01 | 0.998 |

The sweep varies edge count, `d_key`, epsilon, key geometry, `H in {1,2,4,8,16}`, and load `K/d_key`. Controlled orthogonal codes are necessarily restricted to `K/d_key <= 1`; random nonorthogonal codes include the over-capacity `1.5` load. The raw rows include per-hop endpoint error, accumulated prefix error, system/Gram conditioning, coherence, confidence, and exact small-matrix operator norms.

Many-to-one graphs produce operator norms above one even though every successor code is unit norm. This retains the manuscript's corrected amplification counterexample: the simpler `H * epsilon_1` bound is not used unless the operator is contractive. The full geometric bound is checked on every controlled row.

Observed failures are attributable through the recorded axes: larger epsilon creates systematic per-hop shrinkage; correlated geometry increases system conditioning and amplification; `K>d_key` crosses the value-linear capacity limit; and `L>1` permits perturbation growth. These factors are shown rather than filtered.

## Architectural claim versus systems claim

The architectural result is that `csm_chained` performs all 16 adaptive reads with one fixed state and a declared one-layer read loop. The systems result is separate. The following is a prepared fp64 single-query diagnostic at `d_key=K=64`, where CSM reuses a Cholesky factor and explicit softmax retains all pairs:

| method | adaptive reads | layer depth | state bytes | FLOPs | microseconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| csm_chained | 16 | 1 | 65536 | 262144 | 12260.259 |
| softmax_repeated | 16 | 16 | 65536 | 267264 | 522.226 |
| softmax_one | 1 | 1 | 65536 | 16704 | 41.380 |

FLOPs are leading-operation estimates. Timings include Python/PyTorch dispatch and synchronization, use unoptimized reference kernels, and are not throughput claims. A successful architectural demonstration does not imply these reads are cheap enough to matter in a trained system.

## Plots

- [`controlled_success_by_hop.png`](../plots/phase5/controlled_success_by_hop.png)
- [`epsilon_error_accumulation.png`](../plots/phase5/epsilon_error_accumulation.png)
- [`operator_amplification.png`](../plots/phase5/operator_amplification.png)
- [`chain_latency.png`](../plots/phase5/chain_latency.png)

## Reproducibility

- source checkpoint: `a8ad6e59db43ee61adadf4888d69edcb9ac705c5`; dirty at experiment start: `True`
- config: [`configs/phase5_multihop.json`](../configs/phase5_multihop.json)
- device: `AMD Instinct MI300X VF`
- software: Python `3.12.3`, PyTorch `2.8.0+rocm7.0.2.git245bf6ed`, HIP `7.0.51831-7c9236b16`, NumPy `2.3.2`
- wall time: 92.646 seconds; peak allocated VRAM: 0.125867 GiB
- raw hop rows: [`phase5/pointer_chasing.csv`](phase5/pointer_chasing.csv)
- latency rows: [`phase5/latency.csv`](phase5/latency.csv)
- machine-readable record: [`phase5_metrics.json`](phase5_metrics.json)

## Scoped conclusion

Controlled codes reproduce accurate H-hop chasing through 16 adaptive reads against one unchanged CSM state, and the full operator-norm propagation bound holds. Random/correlated and over-capacity rows show where epsilon, geometry, capacity, and amplification degrade the chain. Reference latency remains a separate systems diagnostic, not evidence of practical efficiency.

This phase uses controlled and random representations, not learned encoders. It validates the state/read mechanism and exposes random-geometry failures; it does not satisfy the separate learned-memory or NLP gate.
