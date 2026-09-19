# Phase 2: Live Tensor Memory Profile & Plateau Accounting

Accounting of live tensor bytes across context lengths labeled by storage tier.

- Window Size ($w$): 8
- Page Size ($p$): 4
- Bounded State Plateau Verified: **True** (Plateau: 2112 bytes)

| Context Length | Bounded State (Tier 1 Working RAM) | Archive KV (Tier 2 Host Archive) | Archive Index (Tier 3 Cold Index) | Total Archive Bytes |
|---|---|---|---|---|
| t=1 | 2112 B | 0 B | 0 B | 2112 B |
| t=4 | 2112 B | 0 B | 0 B | 2112 B |
| t=8 | 2112 B | 0 B | 0 B | 2112 B |
| t=12 | 2112 B | 640 B | 208 B | 2960 B |
| t=16 | 2112 B | 1280 B | 416 B | 3808 B |
| t=24 | 2112 B | 2560 B | 832 B | 5504 B |
| t=32 | 2112 B | 3840 B | 1248 B | 7200 B |
| t=48 | 2112 B | 6400 B | 2080 B | 10592 B |
| t=64 | 2112 B | 8960 B | 2912 B | 13984 B |
| t=96 | 2112 B | 14080 B | 4576 B | 20768 B |
| t=128 | 2112 B | 19200 B | 6240 B | 27552 B |
