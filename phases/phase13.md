This is Phase 13.

Goal:
Study the dyadic conjugate cascade as an independent long-context extension of a base CSM whose behavior has already been established.

Do not use the cascade to rescue a failed base CSM.

Implement the smallest correct dyadic hierarchy:
- live CSM memory
- frozen block sufficient statistics
- binary-counter merges
- level-specific reads
- controlled selection/salience policy

Begin with deterministic synthetic memories.

Verify:
1. exact merge equivalence
2. logarithmic number of maintained levels
3. memory scaling with sequence length
4. online selectable recall
5. age/fidelity behavior
6. competition within blocks
7. refresh/rehearsal behavior

Compare:
- flat discounted CSM
- dyadic CSM
- fixed-window attention
- explicitly stored memory under matched bytes

Test the manuscript prediction that flat exponential decay and dyadic competition-based forgetting produce qualitatively different fidelity horizons.

Only after synthetic validation should this extension enter an LM.

Treat the cascade as a separate ablation/research axis rather than bundling it into the first CSM paper.

Write:
results/phase13_dyadic_cascade.md
