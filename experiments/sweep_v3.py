# MoE routing v3: sweep over the number of pretend GPUs (5, 10, ... 50)
# Paste this whole thing into ONE Kaggle cell. Settings: Accelerator = GPU T4, Internet = On.
import math, time, urllib.request, statistics as st
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", DEVICE)

# ---------------- settings (only touch this part) ----------------
QUICK = False                       # True = small fast check (Mac / CPU)

GPUS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]   # the x-axis of the plots
GPU_PER_NODE = 5                    # 5 GPUs per machine -> 1, 2, ... 10 machines
SAME_NODE, CROSS_NODE = 0.3, 1.0    # travel cost: other GPU same machine / other machine

SEEDS = [1, 2, 3]
STEPS = 2000
BS, BLOCK = 32, 64
D_MODEL, N_LAYERS, HEADS, HIDDEN = 128, 2, 4, 256
LR = 2e-3
E, K = 50, 2                        # 50 experts (same model for every GPU count), each token uses 2
AUX_COEF, ETA_P, ETA_BETA = 0.01, 0.05, 0.1
SAVE_FRACS = [0.2, 0.4]             # ours: cut travel by 20% and by 40% vs a router that ignores location
EVAL_BATCHES = 50

if QUICK:
    GPUS, SEEDS, STEPS, BS = [5, 25, 50], [1, 2], 300, 16
    D_MODEL, HIDDEN, SAVE_FRACS, EVAL_BATCHES = 64, 128, [0.2], 10
assert E >= max(GPUS), "need at least one expert per GPU"
n_runs = len(SEEDS) * (2 + len(GPUS) * (1 + len(SAVE_FRACS)))
print(f"{n_runs} training runs of {STEPS} steps")

# ---------------- data ----------------
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
with urllib.request.urlopen(URL, timeout=30) as r:
    text = r.read().decode("utf-8")
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

def get_batch(src, gen):
    ix = torch.randint(0, len(src) - BLOCK - 1, (BS,), generator=gen)
    x = torch.stack([src[i:i + BLOCK] for i in ix])
    y = torch.stack([src[i + 1:i + BLOCK + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ---------------- pretend cluster with G GPUs ----------------
def make_topo(G):
    gpu_node = torch.arange(G) // GPU_PER_NODE               # machine of each GPU
    D = torch.full((G, G), CROSS_NODE)
    D[gpu_node[:, None] == gpu_node[None, :]] = SAME_NODE
    D.fill_diagonal_(0.0)                                    # travel cost table, G x G
    exp_gpu = torch.arange(E) % G                            # expert e lives on GPU e % G
    exp_node = gpu_node[exp_gpu]
    n_nodes = int(gpu_node.max()) + 1
    node_mask = exp_node[None, :] == torch.arange(n_nodes)[:, None]    # machines x experts
    return {"G": G, "n_nodes": n_nodes, "blind": D[:, exp_gpu].mean().item(),
            "D": D.to(DEVICE), "gpu_node": gpu_node.to(DEVICE), "exp_gpu": exp_gpu.to(DEVICE),
            "exp_node": exp_node.to(DEVICE), "node_mask": node_mask.to(DEVICE)}

topos = {G: make_topo(G) for G in GPUS}
for G in GPUS:
    print(f"{G:>2} GPUs = {topos[G]['n_nodes']:>2} machines | travel of a location-blind router: {topos[G]['blind']:.3f}")

# ---------------- model + the four routers ----------------
class MoELayer(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gate = nn.Linear(d, E)
        b1, b2 = 1 / math.sqrt(d), 1 / math.sqrt(h)
        self.W1 = nn.Parameter(torch.empty(E, d, h).uniform_(-b1, b1))   # all experts stored together
        self.B1 = nn.Parameter(torch.empty(E, h).uniform_(-b1, b1))
        self.W2 = nn.Parameter(torch.empty(E, h, d).uniform_(-b2, b2))
        self.B2 = nn.Parameter(torch.empty(E, d).uniform_(-b2, b2))
        self.register_buffer("price", torch.zeros(E))        # load price p_e
        self.register_buffer("beta", torch.zeros(()))        # travel price

    def forward(self, x, router, topo, tok_gpu):
        B, T, Dm = x.shape
        flat = x.reshape(-1, Dm)
        logits = self.gate(flat)                             # g: tokens x experts
        probs = F.softmax(logits, dim=-1)
        tg = tok_gpu.repeat_interleave(T)                    # home GPU of every token
        cost = topo["D"][tg][:, topo["exp_gpu"]]             # d: tokens x experts

        with torch.no_grad():
            if router == "topk_aux":                         # plain top-k (+ aux loss below)
                score = logits
            elif router in ("auxfree", "nodelimit"):         # DeepSeek-style load prices
                score = logits - self.price
            elif router == "ours":                           # g - beta*d - p
                score = logits - self.beta * cost - self.price
            if router == "nodelimit":                        # DeepSeek-V3: all K experts on ONE machine
                per_node = score[:, None, :].masked_fill(~topo["node_mask"][None], float("-inf"))
                best = per_node.topk(K, dim=-1).values.sum(-1).argmax(-1)     # machine with best top-K sum
                score = score.masked_fill(topo["exp_node"][None] != best[:, None], float("-inf"))
            top_i = torch.topk(score, K, dim=-1).indices     # chosen experts, tokens x K

        aux = torch.zeros((), device=x.device)
        if router == "topk_aux":
            frac = F.one_hot(top_i[:, 0], E).float().mean(0)
            aux = E * (frac * probs.mean(0)).sum()

        w = probs.gather(1, top_i)
        w = w / w.sum(-1, keepdim=True)
        h = F.gelu(torch.einsum("nd,edh->neh", flat, self.W1) + self.B1)
        o = torch.einsum("neh,ehd->ned", h, self.W2) + self.B2            # every expert's output
        out = (o.gather(1, top_i[:, :, None].expand(-1, -1, Dm)) * w[:, :, None]).sum(1)   # keep only the chosen K

        with torch.no_grad():
            n = F.one_hot(top_i.reshape(-1), E).sum(0).float()            # tokens per expert
            comm = cost.gather(1, top_i).mean()                           # average travel per choice
            dest, _ = torch.sort(topo["exp_node"][top_i], dim=1)
            first = torch.ones_like(dest, dtype=torch.bool)
            first[:, 1:] = dest[:, 1:] != dest[:, :-1]
            home = topo["gpu_node"][tg]
            remote = ((dest != home[:, None]) & first).sum(1).float().mean()   # other machines per token
        return out.reshape(B, T, Dm), aux, {"n": n, "comm": comm, "remote": remote}


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, HEADS, batch_first=True)
        self.moe = MoELayer(D_MODEL, HIDDEN)

    def forward(self, x, router, topo, tok_gpu, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        m, aux, stats = self.moe(self.ln2(x), router, topo, tok_gpu)
        return x + m, aux, stats


class TinyMoEGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D_MODEL), nn.Embedding(BLOCK, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln, self.head = nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, V)
        self.register_buffer("mask", torch.triu(torch.full((BLOCK, BLOCK), float("-inf")), 1))

    def forward(self, x, router, topo, tok_gpu):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        aux_total, all_stats = 0.0, []
        for blk in self.blocks:
            h, aux, stats = blk(h, router, topo, tok_gpu, self.mask[:T, :T])
            aux_total = aux_total + aux
            all_stats.append(stats)
        return self.head(self.ln(h)), aux_total, all_stats

# ---------------- train and score ----------------
def home_gpus(gen, G):                                       # which GPU each sequence sits on
    return (torch.rand(BS, generator=gen) * G).long().clamp_(max=G - 1).to(DEVICE)

def train(router, seed, topo, target=None):
    torch.manual_seed(seed)                                  # same starting weights for every router
    gen = torch.Generator().manual_seed(1000 + seed)         # same data order for every router
    ggen = torch.Generator().manual_seed(2000 + seed)
    model = TinyMoEGPT().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    moes = [blk.moe for blk in model.blocks]
    model.train()
    for step in range(STEPS):
        x, y = get_batch(train_data, gen)
        logits, aux, stats = model(x, router, topo, home_gpus(ggen, topo["G"]))
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        if router == "topk_aux":
            loss = loss + AUX_COEF * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():                                # Step 3: update the prices
            for moe, s in zip(moes, stats):
                if router != "topk_aux":
                    moe.price += ETA_P * (s["n"] / s["n"].mean() - 1.0)
                if router == "ours":
                    moe.beta.add_(ETA_BETA * (s["comm"] - target)).clamp_(min=0.0)
    return model

def evaluate(model, router, topo):
    model.eval()
    vgen, ggen = torch.Generator().manual_seed(7), torch.Generator().manual_seed(8)
    L, C, R = [], [], []
    n_tot = [torch.zeros(E, device=DEVICE) for _ in range(N_LAYERS)]
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = get_batch(val_data, vgen)
            logits, _, stats = model(x, router, topo, home_gpus(ggen, topo["G"]))
            L.append(F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).item())
            C.append(sum(s["comm"].item() for s in stats) / len(stats))
            R.append(sum(s["remote"].item() for s in stats) / len(stats))
            for tot, s in zip(n_tot, stats):
                tot += s["n"]
    mean = lambda v: sum(v) / len(v)
    return {"ppl": math.exp(mean(L)), "comm": mean(C), "remote": mean(R),
            "load": mean([(t.max() / t.mean()).item() for t in n_tot]),      # busiest expert / average expert
            "beta": mean([blk.moe.beta.item() for blk in model.blocks])}

# ---------------- run everything ----------------
LABELS = {"topk_aux": "top-k + aux loss", "auxfree": "aux-free prices (DeepSeek)",
          "nodelimit": "node-limited (DeepSeek-V3)"}
for f in SAVE_FRACS:
    LABELS[f"ours{f}"] = f"ours, {round(100 * f)}% less travel"
results, done, t0 = [], 0, time.time()

def record(key, seed, G, model, router):
    r = evaluate(model, router, topos[G])
    r.update(key=key, seed=seed, G=G)
    results.append(r)

for seed in SEEDS:
    for router in ("topk_aux", "auxfree"):                   # these ignore location: train once, score on every cluster
        model = train(router, seed, topos[GPUS[0]])
        for G in GPUS:
            record(router, seed, G, model, router)
        done += 1
    for G in GPUS:
        record("nodelimit", seed, G, train("nodelimit", seed, topos[G]), "nodelimit")
        done += 1
        for f in SAVE_FRACS:
            target = (1 - f) * topos[G]["blind"]
            record(f"ours{f}", seed, G, train("ours", seed, topos[G], target), "ours")
            done += 1
        print(f"seed {seed} | {G:>2} GPUs done | {done}/{n_runs} runs | {(time.time() - t0) / 60:.1f} min")

# ---------------- tables ----------------
def cell(key, G, field):
    v = [r[field] for r in results if r["key"] == key and r["G"] == G]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)

base = {(r["seed"], r["G"]): r["ppl"] for r in results if r["key"] == "topk_aux"}
for G in GPUS:
    print(f"\n{G} GPUs ({topos[G]['n_nodes']} machines)")
    print(f"  {'router':<28} {'travel':>7} {'other machines':>15} {'ppl':>7} {'ppl change':>11} {'load':>6}")
    for key in LABELS:
        d = [100 * (r["ppl"] / base[(r["seed"], G)] - 1) for r in results if r["key"] == key and r["G"] == G]
        print(f"  {LABELS[key]:<28} {cell(key, G, 'comm')[0]:>7.3f} {cell(key, G, 'remote')[0]:>15.2f} "
              f"{cell(key, G, 'ppl')[0]:>7.3f} {st.mean(d):>+10.2f}% {cell(key, G, 'load')[0]:>6.2f}")

# ---------------- plots ----------------
STYLE = {"topk_aux": ("#2a78d6", "o", "-"), "auxfree": ("#eb6834", "^", "--"), "nodelimit": ("#1baf7a", "D", "-.")}
for f, sty in zip(SAVE_FRACS, [("#eda100", "s", "-"), ("#e87ba4", "v", "-"), ("#008300", "P", "-")]):
    STYLE[f"ours{f}"] = sty
PANELS = [("comm", "travel cost per choice"), ("remote", "other machines each token is sent to"),
          ("ppl", "held-out perplexity"), ("load", "load: busiest expert / average")]
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for ax, (field, title) in zip(axes.flat, PANELS):
    for key, (color, marker, ls) in STYLE.items():
        m = [cell(key, G, field)[0] for G in GPUS]
        s = [cell(key, G, field)[1] for G in GPUS]
        ax.errorbar(GPUS, m, yerr=s, color=color, marker=marker, ls=ls, lw=2, ms=6, capsize=2, label=LABELS[key])
    ax.set_title(title + " (lower = better)", fontsize=10, loc="left")
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(GPUS)
for ax in axes[1]:
    ax.set_xlabel("number of GPUs")
axes[0, 0].legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig("gpu_sweep.png", dpi=150)
plt.show()
