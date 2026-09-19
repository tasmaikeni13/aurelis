# AURELIS Phase 3: Stress Conditions & Pathologies

| Stress Case | Expected Status | Observed Status | Never Certified | Verdict |
|---|---|---|---|---|
| Underflow (extreme negative scores) | `full_read` | `full_read` | True | **PASS** |
| Overflow (extreme positive scores) | `full_read` | `full_read` | True | **PASS** |
| Nearly cancelled numerators | `full_read` | `full_read` | True | **PASS** |
| Zero unread mass (no remote archive) | `full_read` | `full_read` | True | **PASS** |
| Partial pages at tail (9 items with page_size=4) | `full_read` | `full_read` | True | **PASS** |
| Negative query coordinates | `full_read` | `full_read` | True | **PASS** |
| Deliberately injected unsound bounds (L > U) | `invalid_interval` | `invalid_interval` | True | **PASS** |
| Missing archive page from storage | `archive_error` | `archive_error` | True | **PASS** |
| NaN detected in inputs | `invalid_interval` | `invalid_interval` | True | **PASS** |
| Simulated I/O timeout | `archive_unavailable` | `archive_unavailable` | True | **PASS** |
| Budget exhaustion (max_pages=1) | `budget_exhausted` | `budget_exhausted` | True | **PASS** |
