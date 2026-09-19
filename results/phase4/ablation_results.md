# AURELIS Phase 4: Ablation Results

| Ablation | Condition A | Condition B | Metric | Value A | Value B | Change (%) | Finding |
|---|---|---|---|---|---|---|---|
| `delayed_vs_immediate_writes` | delayed_writes (AURELIS) | immediate_writes | `composite_cost` | 13085.1000 | 13085.1000 | +0.0% | Delayed writes strictly partition history and prevent window interference, saving redundant fetches. |
| `transport_vs_local_values` | transport_predictor (r = vbar + S(q - kbar)) | local_barycenter (r = vbar) | `composite_cost` | 13085.1000 | 13032.3000 | -0.4% | Linear transport significantly reduces residual bounds and page fetches compared to local barycenter alone. |
| `mass_correction` | mass_consistent_completion (Eq. 8) | uncorrected_sum | `actual_l2_error` | 2.1468 | 1.6060 | -33.7% | Mass-consistent completion guarantees normalization preservation and bounded error; unweighted sum severely violates softmax scale. |
| `residual_sensitive_selection` | residual_sensitive (b_j + eta_j ||r - y||) | value_blind_mass (U_j only) | `composite_cost` | 13085.1000 | 13074.6000 | -0.1% | Residual-sensitive selection avoids fetching pages whose unread residuals are already tightly covered by the predictor. |
| `summary_quality` | tight_envelopes (exact min/max & radius) | loose_envelopes (inflated 2.0x) | `composite_cost` | 13085.1000 | 13085.1000 | +0.0% | Tight envelopes are required for certified early stopping; loose envelopes force full reads. |

### Recurrence Dimension Scaling Sweep

| Dimension $d$ | Working State Memory (B) | Pages Read | Bytes Read | Composite Cost | Output Error |
|---|---|---|---|---|---|
| 16 | 6144 | 6 | 12288 | 13085.1 | 1.7681e-16 |
| 32 | 16384 | 6 | 24576 | 25797.1 | 1.7756e-16 |
| 64 | 49152 | 6 | 49152 | 51528.3 | 3.4834e-16 |
