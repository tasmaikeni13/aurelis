This is Phase 3.

Goal:
Determine whether CSM's solver read has a genuine memory advantage over simpler competing reads, rather than merely demonstrating that the equations work.

Implement carefully controlled baselines:

1. Hebbian / outer-product fast-weight memory:
   C = sum v k^T
   read(q) = Cq

2. normalized dot-product / softmax memory over explicitly stored pairs

3. a simple linear-attention-style memory

4. ridge/least-squares oracle where useful

Keep implementations simple enough to audit.

Compare using TWO fairness regimes:

A. same key/value dimension
B. same total state-memory bytes

Where reasonable also report:
- FLOPs/query
- bytes of recurrent state
- wall-clock latency

Experiments:

1. random independent associative recall
2. highly correlated keys
3. almost-colliding keys
4. capacity sweep
5. epsilon sweep
6. value dimension sweep

Then test the manuscript's linear-functional separation.

Store keys k_i and values v_i.
Construct queries:

q = sum_i alpha_i k_i

where alpha contains:
- positive coefficients
- negative coefficients
- coefficients >1
- mixtures summing to values other than 1

Target:
sum_i alpha_i v_i

Compare CSM and normalized softmax reads.

Explicitly test whether normalized smoothing is constrained by convex combinations while the CSM solver can recover linear-span answers.

Do not describe results as superiority unless they survive equal-memory comparisons.

Required report:
results/phase3_baseline_separation.md

PASS GATE:
There should exist clearly characterized regimes in which the solver read delivers a reproducible fidelity advantage over simpler memories.
Document regimes where it does NOT.

If CSM only wins because it uses much more state, mark that as a serious negative result.
