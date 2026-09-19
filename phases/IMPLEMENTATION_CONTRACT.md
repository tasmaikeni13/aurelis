# AURELIS Implementation Contract

This contract defines the state layout, operating contracts, and equation-to-code mapping for implementing AURELIS.

## State and Operations

Per layer, batch item, and recurrent head:

- **Recurrent State $S$:** $[d_{\text{value}}, d_{\text{key}}]$, initially zero; FP32 accumulation baseline.
- **Ring Buffer:** $[w, d_{\text{key}}]$ and $[w, d_{\text{value}}]$; stored write gates $\alpha, \beta$, causal occurrence IDs, head pointer and valid count.
- **Write Gates $\alpha, \beta \in [0, 1]$:** Computed at the observation's source position; key normalization ensures $\|k\|_2 \le 1$ with a finite nonzero floor.
- **Archive (when enabled):** Encoded remote observations, counts, page IDs, key coordinate minima/maxima, value center/radius, and complete index metadata.
- **State Checkpoint:** $S$, ring buffer with pending gates, position, archive length, index version, and RNG state.

Use Eq. (2) for evicted writes and Eq. (3) for bounded output. In archive mode, selected sums and midpoint completion implement Eqs. (7)–(8); envelopes and certificate bounds implement Eqs. (9)–(12). All recent tokens within window $w$ are selected. Every unread remote occurrence belongs to exactly one frontier node.

Public read results must include output tensor, mode (`bounded` or `archive`), certificate bound or null, status (`approximate`, `certified`, `full_read`, `budget_exhausted`, `invalid_state`, `archive_error`), and fetched byte accounting.

## Module Architecture Specification

| Module Area | Implementation Contract | Equations Covered |
|---|---|---|
| `types.py` | State definitions (`DeltaState`), read output container, certificate metrics, archive descriptors | Eqs. (2), (3), (8), (10) |
| `functional.py` | Solve-free gated delta update, bounded read, normalized archive completion, residual certificate | Eqs. (2), (3), (7), (8), (9), (10), (11), (12) |
| `streaming.py` | Streaming sequence processor, ring buffer management, causal handoff, exact partitioning | Eqs. (2), (3) |
| `nn.py` | Neural projection layers, shared query/key coordinates, forward paths for bounded & archive modes | Eqs. (2), (3), (8), (14) |
| `models/` | Cached step decode, local attention, structured delta chunk training scans | Eqs. (2), (8), (15) |
| `baselines/` | Standard Transformer (RoPE, GQA, SwiGLU), SSM hybrid, and recurrence-free archive completion | Baselines |

## Shapes, Hardware, and Accounting

- Recurrent and key dimensions: $d_{\text{key}} = d_{\text{value}} \in \{32, 64, 128\}$.
- Attention window: $w \in \{64, 128, 256\}$.
- Page sizes: $p \in \{32, 64, 128\}$.
- Training chunk size: $c \in \{64, 128\}$.
