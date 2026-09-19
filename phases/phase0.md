# Phase 0 — Environment Inventory, Formal Verification, & Preregistration

Phase 0 establishes the experimental foundation, verified formal baseline, runtime device inventory, and preregistration for AURELIS.

## Requirements

1. **Runtime Inventory:** Enumerate host CPU, RAM, and runtime accelerator devices via direct introspection.
2. **Formal Verification:** Build Lean 4 formal core (`lake build`) and record exact theorem-to-equation mappings.
3. **Module Specification Mapping:** Map equations (2)–(12) to planned modules and test contracts.
4. **Hypothesis Preregistration:** Preregister core hypotheses (H1–H4), workload configurations, paired seeds, compute limits, and raw-data provenance plans.
5. **Resource Budget:** Formulate compute, wall-clock, and memory budgets with explicit stop conditions.

## Deliverables in `results/phase0/`

- `environment.json` & `environment_inventory.md`
- `formal_build_report.md`
- `module_specification_map.json` & `module_specification_map.md`
- `preregistration.json` & `preregistration.md`
- `resource_budget.json` & `resource_budget.md`
- `report.md` & `PASS.md`

## Gates

1. Every claimed device is confirmed by direct runtime enumeration.
2. Every metric type has a raw-data provenance plan.
3. Compute limits (timeout, RSS cap) and stop conditions are explicit.
4. Lean formal build passes with zero admitted proofs (`sorry`) and zero project axioms.
5. Architecture module specification and experiment preregistration are complete.
