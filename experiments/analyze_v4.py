# Statistics and figures for the v4 run.
# Usage: python analyze_v4.py path/to/results_v4.json [output_folder]
import sys, os, math, json, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results_v4.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(OUT, exist_ok=True)

GPU_PER_NODE = 5
NODE_M = 2

results = json.load(open(RESULTS))
GPUS = sorted({r["G"] for r in results})                     # read the settings from the results file
SEEDS = sorted({r["seed"] for r in results})
SAVE_FRACS = sorted({float(r["key"][4:]) for r in results if r["key"].startswith("ours")})
LABELS = {"topk_aux": "top-k + aux loss", "auxfree": "aux-free prices (DeepSeek)",
          "nodelimit": f"node-limited, {NODE_M} machines (DeepSeek-V3)"}
for f in SAVE_FRACS:
    LABELS[f"ours{f}"] = f"ours, {round(100 * f)}% less travel"

try:
    from scipy import stats as sps
    tcrit = lambda n: sps.t.ppf(0.975, n - 1)
except ImportError:
    tcrit = lambda n: {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}.get(n, 2.0)

def get(key, seed, G):
    return next(r for r in results if r["key"] == key and r["seed"] == seed and r["G"] == G)

def ci(values):                                              # mean and 95% half-width over seeds
    m = st.mean(values)
    half = tcrit(len(values)) * st.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else float("nan")
    return m, half

def verdict(m, h):
    if m - h > 0: return "worse"
    if m + h < 0: return "better"
    return "can't tell from zero"

def change(key, G, ref="topk_aux"):                          # paired % perplexity change vs a reference router
    return [100 * (get(key, s, G)["ppl"] / get(ref, s, G)["ppl"] - 1) for s in SEEDS]

def mean_of(key, G, field):
    return st.mean(get(key, s, G)[field] for s in SEEDS)

for G in GPUS:
    print(f"\n{G} GPUs ({-(-G // GPU_PER_NODE)} machines)")
    print(f"  {'router':<38} {'travel':>7} {'other machines':>15} {'ppl change, 95% CI':>26} {'load':>6}")
    for key in LABELS:
        m, h = ci(change(key, G))
        print(f"  {LABELS[key]:<38} {mean_of(key, G, 'comm'):>7.3f} {mean_of(key, G, 'remote'):>15.2f} "
              f"{m:>+8.2f}% [{m - h:+.2f}, {m + h:+.2f}] {mean_of(key, G, 'load'):>6.2f}   "
              f"{verdict(m, h) if key != 'topk_aux' else ''}")

# Matched comparison: read our curve (0% = aux-free, 20, 40, 60% less travel) at node-limited's
# number of other machines, and compare perplexity there, seed by seed.
def ours_at(seed, G, remote_target):
    pts = [(get("auxfree", seed, G)["remote"], get("auxfree", seed, G)["ppl"])]
    pts += [(get(f"ours{f}", seed, G)["remote"], get(f"ours{f}", seed, G)["ppl"]) for f in SAVE_FRACS]
    pts.sort()
    if not pts[0][0] <= remote_target <= pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= remote_target <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (remote_target - x0) / (x1 - x0)

print("\nNode-limited vs ours at the SAME number of other machines (positive = node-limited is worse):")
for G in GPUS:
    diffs = []
    for s in SEEDS:
        nl = get("nodelimit", s, G)
        o = ours_at(s, G, nl["remote"])
        if o is not None:
            diffs.append(100 * (nl["ppl"] / o - 1))
    if len(diffs) < 2:
        print(f"  {G:>2} GPUs: node-limited's machine count is outside our curve")
        continue
    m, h = ci(diffs)
    word = {"worse": "node-limited is worse", "better": "node-limited is better"}.get(verdict(m, h), "no clear difference")
    print(f"  {G:>2} GPUs: {m:+.2f}%  95% CI [{m - h:+.2f}, {m + h:+.2f}]  ({len(diffs)} runs)  -> {word}")

print("\nAgainst aux-free (our rule = aux-free + a travel price), perplexity change with 95% CI:")
for G in GPUS:
    parts = []
    for key in ["nodelimit"] + [f"ours{f}" for f in SAVE_FRACS]:
        m, h = ci(change(key, G, ref="auxfree"))
        parts.append(f"{LABELS[key].split(' (')[0]}: {m:+.2f} [{m - h:+.2f}, {m + h:+.2f}]")
    print(f"  {G:>2} GPUs | " + " | ".join(parts))

# ---------------- figures ----------------
STYLE = {"topk_aux": ("#2a78d6", "o", "-"), "auxfree": ("#eb6834", "^", "--"), "nodelimit": ("#1baf7a", "D", "-.")}
for f, sty in zip(SAVE_FRACS, [("#eda100", "s", "-"), ("#e87ba4", "v", "-"), ("#4a3aa7", "P", "-")]):
    STYLE[f"ours{f}"] = sty

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for key, (color, marker, ls) in STYLE.items():
    if key == "topk_aux":
        continue
    cis = [ci(change(key, G)) for G in GPUS]
    axes[0].errorbar(GPUS, [c[0] for c in cis], yerr=[c[1] for c in cis], color=color, marker=marker, ls=ls,
                     lw=2, ms=6, capsize=3, label=LABELS[key])
    axes[1].plot(GPUS, [mean_of(key, G, "remote") for G in GPUS], color=color, marker=marker, ls=ls, lw=2, ms=6)
axes[0].axhline(0, color="#888780", lw=1)
axes[0].set_title("perplexity change vs top-k + aux, 95% CI (lower = better)", fontsize=10, loc="left")
axes[1].set_title("other machines each token is sent to (lower = better)", fontsize=10, loc="left")
for ax in axes:
    ax.set_xlabel("number of GPUs"); ax.set_xticks(GPUS)
    ax.grid(True, color="#e6e5e0", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "v4_sweep.png"), dpi=150)

fig, axes = plt.subplots(1, len(GPUS), figsize=(3.4 * len(GPUS), 3.8), sharey=True)
for ax, G in zip(axes, GPUS):
    curve = ["auxfree"] + [f"ours{f}" for f in SAVE_FRACS]
    ax.plot([mean_of(k, G, "remote") for k in curve], [ci(change(k, G))[0] for k in curve],
            color="#e87ba4", lw=2, marker="v",
            label="ours (" + ", ".join(["0"] + [str(round(100 * f)) for f in SAVE_FRACS]) + "% less travel)")
    m, h = ci(change("nodelimit", G))
    ax.errorbar([mean_of("nodelimit", G, "remote")], [m], yerr=[h], color="#1baf7a", marker="D", ms=8,
                capsize=3, ls="none", label=LABELS["nodelimit"])
    ax.set_title(f"{G} GPUs", fontsize=10, loc="left"); ax.set_xlabel("other machines per token")
    ax.grid(True, color="#e6e5e0", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("perplexity change vs top-k + aux (%)")
axes[0].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "v4_tradeoff.png"), dpi=150)
