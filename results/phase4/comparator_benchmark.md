# AURELIS Phase 4: Required Comparator Benchmark

Evaluation across all 8 required comparators at identical Q/K/V, archive pages, encoding, and accounting.

### 1. Equal Fetched-KV Budget Comparison (Context $t=64$, Window $w=16$, Page $p=8$)

| Budget (Pages) | Comparator | Pages Read | Actual Error | Certified Bound | Composite Cost |
|---|---|---|---|---|---|
| 0 | `aurelis` | 0 | 1.7451 | 36.308 | 204.0 |
| 0 | `local_barycenter` | 0 | 0.6120 | 31.545 | 151.2 |
| 0 | `zero` | 0 | 0.4876 | 30.695 | 151.2 |
| 0 | `global_summary` | 0 | 0.0706 | 30.712 | 163.0 |
| 0 | `per_page_center` | 0 | 0.0942 | 28.172 | 151.2 |
| 0 | `simple_fusion` | 0 | 0.5425 | N/A | 202.4 |
| 0 | `sparse_no_completion` | 0 | 0.6120 | 58.047 | 151.2 |
| 0 | `value_blind_mass` | 0 | 1.7451 | 36.308 | 204.0 |
| 1 | `aurelis` | 1 | 1.5867 | 21.366 | 2353.4 |
| 1 | `local_barycenter` | 1 | 0.4928 | 17.501 | 2300.6 |
| 1 | `zero` | 1 | 0.3277 | 16.643 | 2300.6 |
| 1 | `global_summary` | 1 | 0.1277 | 16.843 | 2312.4 |
| 1 | `per_page_center` | 1 | 0.1156 | 15.143 | 2310.2 |
| 1 | `simple_fusion` | 1 | 0.6604 | N/A | 2351.8 |
| 1 | `sparse_no_completion` | 1 | 0.4848 | 39.792 | 2297.6 |
| 1 | `value_blind_mass` | 1 | 1.5867 | 21.366 | 2350.4 |
| 2 | `aurelis` | 2 | 1.4144 | 10.674 | 4501.8 |
| 2 | `local_barycenter` | 2 | 0.3948 | 7.126 | 4449.0 |
| 2 | `zero` | 2 | 0.2283 | 6.359 | 4449.0 |
| 2 | `global_summary` | 2 | 0.1972 | 6.656 | 4460.8 |
| 2 | `per_page_center` | 2 | 0.1177 | 5.663 | 4466.6 |
| 2 | `simple_fusion` | 2 | 0.7455 | N/A | 4500.2 |
| 2 | `sparse_no_completion` | 2 | 0.4523 | 20.525 | 4443.5 |
| 2 | `value_blind_mass` | 2 | 1.4144 | 10.674 | 4496.3 |
| 4 | `aurelis` | 4 | 0.7645 | 3.323 | 8795.5 |
| 4 | `local_barycenter` | 4 | 0.2359 | 1.713 | 8742.7 |
| 4 | `zero` | 4 | 0.0652 | 1.265 | 8742.7 |
| 4 | `global_summary` | 4 | 0.1160 | 1.422 | 8754.5 |
| 4 | `per_page_center` | 4 | 0.0727 | 1.077 | 8771.5 |
| 4 | `simple_fusion` | 4 | 0.5832 | N/A | 8793.9 |
| 4 | `sparse_no_completion` | 4 | 0.2236 | 6.967 | 8733.7 |
| 4 | `value_blind_mass` | 4 | 0.7630 | 5.362 | 8786.5 |

### 2. Tolerance Sweep at Matched Resources (Sample: Context 64, Page 8, $\epsilon=0.1$)

| Comparator | Status | Pages Read | Bytes Read | Actual Error | Bound | Working Memory (B) | Composite Cost |
|---|---|---|---|---|---|---|---|
| `aurelis` | `full_read` | 6 | 12288 | 0.0000 | 0.000 | 6144 | 13085.1 |
| `local_barycenter` | `full_read` | 6 | 12288 | 0.0000 | 0.000 | 4096 | 13032.3 |
| `zero` | `full_read` | 6 | 12288 | 0.0000 | 0.000 | 4096 | 13032.3 |
| `global_summary` | `full_read` | 6 | 12288 | 0.0000 | 0.000 | 4096 | 13044.1 |
| `per_page_center` | `full_read` | 6 | 12288 | 0.0000 | 0.000 | 4096 | 13065.9 |
| `simple_fusion` | `full_read` | 6 | 12288 | 0.0000 | 0.000 | 6144 | 13083.5 |
| `sparse_no_completion` | `full_read` | 6 | 12288 | 0.0000 | 0.000 | 4096 | 13021.8 |
| `value_blind_mass` | `full_read` | 6 | 12288 | 0.0000 | 0.000 | 6144 | 13074.6 |
