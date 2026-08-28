This is Phase 14.

You are no longer optimizing the architecture.

Act as a hostile independent reviewer.

Read:
- the original manuscript
- CLAIMS.md from Phase 0
- every phase report
- raw experiment summaries
- baseline comparisons

For every claimed advantage, attempt to explain it away using:

- extra state
- extra parameters
- extra FLOPs
- easier optimization
- unfair baseline tuning
- data leakage
- seed selection
- favorable synthetic distributions
- epsilon tuning
- dimensionality differences
- precision differences
- implementation quality differences

Run only the minimum additional experiments required to distinguish these explanations.

Then produce:

PAPER_DECISION.md

Sections:

1. Claims robustly supported
2. Claims partially supported
3. Claims falsified
4. New phenomena discovered
5. Most serious unresolved weakness
6. Strongest baseline
7. Equal-budget comparison
8. Hardware reality on MI300X
9. Whether CSM deserves a full architecture paper
10. Whether the work should instead be published as a memory-mechanism paper
11. Recommended next research question

Also identify the smallest publishable central claim.

Do not optimize rhetoric.
Do not convert null results into wins.
Do not introduce claims that were not measured.
