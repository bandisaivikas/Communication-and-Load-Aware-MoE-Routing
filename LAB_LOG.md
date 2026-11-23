# Lab log

One entry per experiment: what changed, what I expected, what happened, what it means.

## Sep 2026: derivation v1
- **Changed:** wrote the objective `Σ z·g − β Σ z·d − Σ φ(n)` (IB-shaped) and derived the rule `top-K of g − β·d − p` with a Lagrangian. The prices are set by dual ascent.
- **Means:** the rule is derived, but the objective itself was assumed (g as usefulness, d made up, φ never chosen).

## Sep 2026: toy simulation and load tracking
- **Changed:** made-up gate scores. Compared a running load count with load prices.
- **Result:** about 13% less communication at the same quality and load. Prices were as good as the running count and much faster.
- **Means:** use prices. The scripts for this were lost when the workspace was reset; only the results remain.

## Sep 2026: v1 real model
- **Problems found:** quality was measured on training text, and a fixed β drifted (communication rose from 0.20 to 0.42 as the gate scores grew).
- **Fix:** held-out evaluation, and β as the price of a travel budget.

## Sep 2026: v2, 4 GPUs, 8 experts, K=2, 5 seeds
- **Expected:** some travel can be saved for free.
- **Result:** 21% less travel, with a perplexity change of +0.11% [−0.74, +0.96]. 39% less costs +0.99%, 56% less costs +3.22%. Travel hit the budget every time.
- **Means:** the first savings are free; pushing harder costs quality.

## Oct 2026: v3, 5–50 GPUs, 50 experts, K=2, 3 seeds
- **Result:** 20% less travel costs under 1% at every size. Node-limited (M=1) lost up to 7.6%.
- **Caveat:** M=1 is far stricter than DeepSeek-V3's setting, so this comparison was unfair to node-limited.

## Oct 2026: v4, fair node-limited test, K=4, M=2, 15–50 GPUs, 5 seeds
- **Expected:** our rule beats node-limited at equal cross-machine traffic.
- **Result:** no clear difference at any size (for example 50 GPUs: −0.32% [−1.21, +0.57]). 20% less travel is free compared with aux-free. 60% less costs 2–5%.
- **Means:** the expectation was wrong. The v1 objective charges travel per expert choice and load as a sum. Real systems charge each machine once per token, and the step waits for the slowest machine. Next: derivation v2 with that cost model.
