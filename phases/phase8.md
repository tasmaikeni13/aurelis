# Phase 8 — Independent reproduction, paper finalization, and release audit

Start only after Phase 7 PASS. Read the entire repository and
`phases/AUTONOMY_PROTOCOL.md`. Treat completion as unproven. Execute the
failure-repair loop for every discrepancy until all gates pass.

## Clean-room reproduction

From a fresh clone and clean environment on the target MI300X server:

- bootstrap without undocumented system state;
- run unit/property/pathology tests and `lake build`;
- reproduce every phase smoke gate;
- independently regenerate the central numerical, learned, systems, tiny-LM,
  and scale tables from pinned configs/checkpoints;
- verify hashes and that plots/tables are generated rather than hand-edited;
  and
- compare raw metrics with every claim in `CLAIMS.md` and the manuscript.

Use a second implementation or independently assembled oracle for the central
head equations and a second evaluator for primary diagnostics. Investigate any
agreement that could arise from shared helpers.

## Standalone paper update

Update `aurelis.md` as a first-presentation standalone paper. Add empirical
methods/results only when reproduced, distinguish preregistered from
exploratory analyses, include uncertainty and resource accounting, and remove
claims contradicted by evidence. Do not include migration history or describe
the architecture as a modification of a repository predecessor. Keep the
related-work/novelty boundary current through the final search date.

Reconcile every theorem with Lean coverage and every number with an artifact.
Render/check all Markdown, links, tables, figures, equation references, and
bibliography. If a publication-format PDF/TeX version is created, generate it
from an authoritative source and check the rendered pages.

## Repository and release audit

- Case-insensitive search confirms no obsolete identity in tracked files.
- There are exactly nine numbered phases, `phase0.md` through `phase8.md`.
- Licensing, data licenses, model cards, security notes, environment pins, and
  reproduction commands are complete.
- No credentials, downloaded corpora, oversized checkpoints, caches, or
  machine-specific secrets are tracked.
- Git status is clean after generated-artifact policy is applied.
- Remote URL and repository name are AURELIS; CI or the documented local
  equivalent runs the release gates.

## Completion audit

Create a requirement-by-requirement matrix for the paper, code, math, Lean,
all phase gates, MI300X/ROCm support, reproducibility, and public repository.
For each item cite direct evidence and classify it proved, contradicted,
incomplete, or missing. Continue working on every non-proved required item.

## PASS gates

- Fresh-clone reproduction passes without manual fixes.
- Primary numbers reproduce within preregistered tolerances and every paper
  claim has direct evidence.
- Independent implementation/evaluator checks agree within declared bounds.
- Full Lean build has no placeholders or unreviewed axioms and coverage is
  accurate.
- The standalone paper, source links, figures, and release documentation pass
  automated and visual checks.
- The repository naming/content audit, secret/large-file audit, and all
  inherited phase gates pass.
- `results/phase8/PASS.md` contains the full completion matrix and shared PASS
  record. Only then may the research program be called complete.
