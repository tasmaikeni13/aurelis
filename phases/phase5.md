This is Phase 5.

Goal:
Test the CSM claim that multiple adaptive memory accesses can be chained against the same state within one layer/computation.

Create pointer-chasing / functional-graph tasks.

A memory contains mappings:

node -> successor(node)

Query starts from one node.
Target is the node H hops downstream.

Sweep:

number of stored edges
d_k
epsilon
key conditioning
H in {1,2,4,8,16}
K/d_k

Test two key regimes:
1. orthogonal / controlled node codes
2. learned or random nonorthogonal node representations

Implement:

q_0 = start
q_{j+1} = read(q_j)

Measure:
- success by hop count
- per-hop error
- accumulated error
- operator norm estimates where practical
- confidence across hops
- total FLOPs and latency

Compare against:
- one softmax access
- repeated softmax accesses with equal number of adaptive reads
- equivalent-depth approaches where meaningful

Important:
Separate the architectural claim
"CSM supports H reads from one maintained state"
from the systems claim
"those H reads are cheap enough to matter."

PASS GATE:
Controlled-code experiments should reproduce the expected multi-hop behavior.
If errors explode, identify whether the cause is epsilon, key geometry, capacity, or operator amplification.

Write:
results/phase5_multihop.md
