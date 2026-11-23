# Communication- and load-aware routing for Mixture-of-Experts models

A routing rule for Mixture-of-Experts (MoE) models that takes into account **where** each expert lives (which GPU, which machine) and **how busy** it is. The rule is derived from one objective that has the shape of the information bottleneck.

Status: ongoing. All experiments use a small model on a simulated cluster.

**Current claim (v1):** the derived rule matches DeepSeek-V3's node-limited routing on quality at equal cross-machine traffic, and adds a travel budget that can be set to any value. It does not beat node-limited routing yet. Version 2 of the derivation aims to: it models the real step time (the slowest machine, with each machine charged once per token). See [LAB_LOG.md](LAB_LOG.md).

## The rule

Each token picks the K experts with the highest

```
score = g − β·d − p
```

| Symbol | Meaning |
|---|---|
| `g` | gate score: how useful the expert looks for this token |
| `d` | travel cost from the token's GPU to the expert's GPU (0 same GPU, 0.3 same machine, 1.0 other machine) |
| `β` | travel price |
| `p` | load price of the expert (higher when it is busy) |

Both prices set themselves after every training step (dual ascent):

```
p_e ← p_e + η_p · (n_e / average load − 1)
β   ← max(0, β + η_β · (C − C*))        # C = measured travel, C* = travel budget
```

With `β = 0` this is DeepSeek-style aux-loss-free balancing. With `β = 0` and `p = 0` it is plain top-k.

## Where the rule comes from

The objective:

```
maximize   Σ_{t,e} z_te g_te  −  β Σ_{t,e} z_te d_te  −  Σ_e φ(n_e)
subject to Σ_e z_te = K for every token t
```

- `z_te = 1` if token t uses expert e
- `n_e` is the load of expert e
- `φ` is a convex busy cost

The first two terms mirror the information bottleneck `I(Z;Y) − β·I(X;Z)`: useful information kept, minus a price for the code. Here the cost is network travel instead of bits, so the objective is IB-shaped, not exact IB. The third term is added for load.

A Lagrangian with one price `p_e` per expert splits the objective into one independent piece per token. Each token then takes the top K of `g − β·d − p_e`, where `p_e = φ′(n_e)` at the optimum (the KKT conditions).

## Repository layout

```
notebooks/moe_routing_experiment_v2.ipynb   4 simulated GPUs: 3 routers, 5 seeds, held-out evaluation
experiments/sweep_v3.py                     5 to 50 simulated GPUs, 2 experts per token
experiments/sweep_v4.py                     fair node-limited test: 4 experts per token, up to 2 machines, 5 seeds
experiments/analyze_v4.py                   statistics and figures for v4 (reads results_v4.json)
results/v2, results/v3, results/v4          outputs and figures
```

Each experiment runs on one free Kaggle T4 GPU. Paste the script into one cell, with Accelerator = GPU T4 and Internet = On. The cluster is simulated: each expert gets a pretend GPU and the code counts travel; nothing is timed.

**Setup in all experiments:**
- 2-layer character-level language model on Tiny Shakespeare.
- Quality is held-out perplexity (last 10% of the text).
- Every router gets the same data order and starting weights for a given seed (paired comparison).

## Results

### v2: 4 GPUs on 2 machines, 8 experts, K = 2, 5 seeds

| Router | Travel | Held-out perplexity | Change vs top-k (95% CI) | Load |
|---|---|---|---|---|
| top-k + aux loss | 0.575 | 5.116 ± 0.038 | reference | 1.39 |
| aux-free prices (DeepSeek-style) | 0.575 | 5.111 ± 0.020 | −0.09% [−1.00, +0.82] | 1.10 |
| ours, budget 0.45 | 0.452 (−21%) | 5.122 ± 0.050 | +0.11% [−0.74, +0.96] | 1.12 |
| ours, budget 0.35 | 0.352 (−39%) | 5.167 ± 0.020 | +0.99% [+0.27, +1.71] | 1.18 |
| ours, budget 0.25 | 0.253 (−56%) | 5.281 ± 0.033 | +3.22% [+2.28, +4.15] | 1.20 |

21% less travel with no measurable quality change. Travel lands on the budget in every case. Figure: [results/v2/quality_vs_comm.png](results/v2/quality_vs_comm.png).

### v3: 5 to 50 GPUs, 50 experts, K = 2, 3 seeds

- 20% less travel costs under 1% perplexity at every size.
- 40% less travel costs 0.5% at 5 GPUs and 3.4% at 50 GPUs.

Node-limited routing here has M = 1 (both experts on one machine). That is much stricter than DeepSeek-V3's setting, so this comparison favours our rule. Full tables: [results/v3/v3_output.txt](results/v3/v3_output.txt). Figure: [results/v3/gpu_sweep.png](results/v3/gpu_sweep.png).

### v4: fair node-limited test, 15 to 50 GPUs, K = 4, M = 2, 5 seeds

Our rule at the **same number of other machines** as node-limited (positive = node-limited is worse):

| GPUs | Difference | 95% CI |
|---|---|---|
| 15 | −0.56% | [−1.27, +0.15] |
| 20 | −0.44% | [−0.93, +0.05] |
| 30 | +0.06% | [−0.98, +1.09] |
| 40 | −0.13% | [−0.69, +0.43] |
| 50 | −0.32% | [−1.21, +0.57] |

**No clear difference at any size:** in a fair setting, our rule matches node-limited routing on quality at equal cross-machine traffic.

Against aux-free balancing (our rule minus the travel term), 20% less travel changes perplexity by −0.32% to +0.13%, and every CI includes zero. Pushing to 60% less travel costs 2–5%.

Full output: [results/v4/v4_output.txt](results/v4/v4_output.txt). Figures: [results/v4/v4_tradeoff.png](results/v4/v4_tradeoff.png), [results/v4/v4_sweep.png](results/v4/v4_sweep.png).

## Limits

- Small model (2 layers, character tokens, about 1.1M characters of text).
- Simulated cluster: travel is counted in assumed cost units (0 / 0.3 / 1.0), not measured time.
- The objective is IB-shaped, not exact IB. The gate score stands in for usefulness, and the load term is a sum while real step time is set by the slowest GPU.
- TA-MoE and LocMoE are not yet compared.

## References

- Fedus et al., [Switch Transformer](https://arxiv.org/abs/2101.03961)
- Lepikhin et al., [GShard](https://arxiv.org/abs/2006.16668)
- Lewis et al., [BASE Layers](https://arxiv.org/abs/2103.16716)
- Wang et al., [Auxiliary-loss-free load balancing](https://arxiv.org/abs/2408.15664)
- DeepSeek-AI, [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- Han and Zhong, [the DeepSeek bias as a primal-dual method](https://arxiv.org/abs/2512.03915)
- Tishby et al., The information bottleneck method (1999)
