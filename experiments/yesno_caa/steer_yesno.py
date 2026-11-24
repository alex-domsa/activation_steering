"""
Steering test on the yes/no judgement: does ADDING the analogy direction change what the model says?

Idea (CAA, Rimsky et al. 2024):
  direction v = mean activation of ANALOGY pairs - mean activation of UNRELATED pairs, taken at the final
  token of the joint prompt (cache_joint.py), on the TRAIN pairs.
  We then add  +m*v  to the model's activations while it reads UNRELATED pairs (does "yes" go up?) and
  -m*v  while it reads ANALOGY pairs (does "yes" go down?).  m is a multiplier: m=1 shifts by one
  group-difference.
Measure: change in log P(yes) - log P(no) per pair (paired: steered minus unsteered).

Controls (so we know it is THIS direction and not just "poke the model"):
  - K random directions with exactly the same length as v, applied the same way.
  - 'yes+no mass': how much probability the model still puts on answering yes or no. If it collapses,
    the steering is breaking the model rather than steering it.

Layer: chosen by steering effect on a TRAIN subset (real direction minus random direction), NOT on
validation. Results are then reported on a VALIDATION subset. TEST is not touched.

Run:   python steer_yesno.py
Needs: cache_joint/<model>/<contrast>/ (from cache_joint.py) for the MODEL set in cache_activations.py.
       The Qwen2.5-1.5B-Instruct run is the one that is meaningful (its own yes/no AUC was 0.69;
       Llama-3.2-1B base says 'yes' to everything, so there is little to steer).
Output: results/<model>/<contrast>/joint/steering_*.csv, steering.png, steering_vector.npy
Time:   roughly 8,000 short forward passes; set N_EVAL_PER_GROUP / K_RANDOM lower for a quick first look.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cache_activations import MODEL, pick_device
from cache_joint import PROMPT, load_joint, JOINT_DIR
from storyanalogy_loader import CONTRAST

warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Settings ---------------------------------------------------------------
POSITIONS = "last"        # where to add the vector: "last" = final prompt token only (where v was measured),
                          # "all" = every prompt token (the usual CAA choice)
M_SEL = 4.0               # multiplier used when choosing the layer (train subset)
M_GRID = (1, 2, 4, 8)     # multipliers reported on validation
N_SELECT_PER_GROUP = 50   # train pairs per group used to choose the layer
N_EVAL_PER_GROUP = 150    # validation pairs per group used for the final numbers
K_RANDOM = 4              # random control directions in the final evaluation
BATCH_SIZE = 4            # prompts per forward pass (set 1 if the batch self-check complains)
N_BOOT = 300
SEED = 0
RESULTS_DIR = Path("results") / MODEL.replace("/", "__") / CONTRAST / "joint"
# -----------------------------------------------------------------------------


class HFSteerer:
    """Wraps the Hugging Face model. logodds(prompts, layer, delta) returns, for every prompt,
    log P(yes) - log P(no) and log(P(yes) + P(no)), optionally with `delta` added to the residual
    stream after layer index `layer` (same indexing as the cached activations: 0 = embeddings,
    i = output of transformer block i)."""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = pick_device()
        print(f"Loading {MODEL} on {self.device} ...")
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"            # so the LAST position is always the real final token
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(self.device).eval()
        self.blocks = self.model.model.layers
        first = lambda ws: sorted({self.tok.encode(w, add_special_tokens=False)[0] for w in ws})
        self.yes_ids = torch.tensor(first(["yes", " yes", "Yes", " Yes"]), device=self.device)
        self.no_ids = torch.tensor(first(["no", " no", "No", " No"]), device=self.device)

    def logodds(self, prompts, layer=None, delta=None, batch_size=BATCH_SIZE):
        torch = self.torch
        lo, mass = [], []
        for i in range(0, len(prompts), batch_size):
            enc = self.tok(list(prompts[i:i + batch_size]), return_tensors="pt", padding=True).to(self.device)
            handle = None
            if delta is not None:
                vec = torch.tensor(np.asarray(delta), dtype=torch.float32, device=self.device)

                def hook(module, inputs, output, vec=vec):
                    is_tuple = isinstance(output, tuple)          # some versions return a tuple, some a tensor
                    hs = (output[0] if is_tuple else output).clone()
                    if POSITIONS == "last":
                        hs[:, -1, :] += vec
                    else:
                        hs += vec
                    return (hs,) + tuple(output[1:]) if is_tuple else hs

                handle = self.blocks[layer - 1].register_forward_hook(hook)   # hidden_states[i] = output of block i-1
            try:
                with torch.no_grad():
                    logits = self.model(**enc).logits[:, -1].float()
            finally:
                if handle is not None:
                    handle.remove()
            logp = torch.log_softmax(logits, dim=-1)
            yes = torch.logsumexp(logp[:, self.yes_ids], dim=1)
            no = torch.logsumexp(logp[:, self.no_ids], dim=1)
            lo.append((yes - no).cpu().numpy())
            mass.append(torch.logaddexp(yes, no).cpu().numpy())
        return np.concatenate(lo), np.concatenate(mass)


def boot_ci(x, rng):
    x = np.asarray(x)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def random_direction(rng, norm, d):
    g = rng.normal(size=d)
    return g / np.linalg.norm(g) * norm


def effect(steerer, prompts, base, layer, vec, m, sign):
    """steered minus unsteered log-odds and mass change, per pair."""
    lo, mass = steerer.logodds(prompts, layer=layer, delta=sign * m * vec)
    return lo - base[0], mass - base[1], lo


def main(steerer=None):
    rng = np.random.default_rng(SEED)
    A_tr, M_tr = load_joint("train")
    _, M_va = load_joint("val")
    n_blocks = A_tr["last"].shape[1] - 1
    y_tr = M_tr.label.to_numpy()
    print(f"Joint cache: {JOINT_DIR} | {n_blocks} blocks | steering positions: {POSITIONS}")

    # ---- the direction at every layer: mean(analogy) - mean(unrelated), raw activations, TRAIN ----
    V = A_tr["last"][y_tr == 1].mean(0) - A_tr["last"][y_tr == 0].mean(0)        # (layers+1, d)
    typical = np.linalg.norm(A_tr["last"], axis=2).mean(0)                        # typical activation length per layer
    d = V.shape[1]

    mk_prompts = lambda M, idx: [PROMPT.format(s1=M.s1[i], s2=M.s2[i]) for i in idx]

    def subset(M, n):
        out = {}
        for g in (0, 1):
            ids = np.flatnonzero(M.label.to_numpy() == g)
            out[g] = rng.choice(ids, size=min(n, len(ids)), replace=False)
        return out

    if steerer is None:
        steerer = HFSteerer()

    # ---- self-checks (so a silent bug cannot fool us) ----
    probe_prompts = mk_prompts(M_tr, range(8))
    a = steerer.logodds(probe_prompts, batch_size=BATCH_SIZE)[0]
    b = steerer.logodds(probe_prompts, batch_size=1)[0]
    print(f"\nSelf-check 1 - batch of {BATCH_SIZE} vs one-by-one: max difference {np.abs(a - b).max():.4f} "
          f"({'ok' if np.abs(a - b).max() < 0.1 else 'TOO LARGE: set BATCH_SIZE = 1'})")
    mid = max(1, n_blocks // 2)
    big = random_direction(rng, 20 * np.linalg.norm(V[mid]), d)
    moved = np.abs(steerer.logodds(probe_prompts, layer=mid, delta=big)[0] - a).mean()
    print(f"Self-check 2 - a large random push at layer {mid} moves the log-odds by {moved:.3f} on average "
          f"({'ok, the hook works' if moved > 1e-3 else 'NO CHANGE: the hook is not taking effect'})")

    # ---- choose the layer on a TRAIN subset ----
    sel = subset(M_tr, N_SELECT_PER_GROUP)
    P = {g: mk_prompts(M_tr, sel[g]) for g in (0, 1)}
    base = {g: steerer.logodds(P[g]) for g in (0, 1)}
    cand = [l for l in range(int(round(0.25 * n_blocks)), int(round(0.85 * n_blocks)) + 1, 2) if 1 <= l <= n_blocks - 2]
    print(f"\nChoosing the layer on {N_SELECT_PER_GROUP} train pairs per group, multiplier {M_SEL}:")
    print("layer | real: push unrelated up | real: push analogy down | random control | gain (real - |random|)")
    rows = []
    t0 = time.time()
    for L in cand:
        v = V[L]; r = random_direction(rng, np.linalg.norm(v), d)
        def score(vec):
            up = effect(steerer, P[0], base[0], L, vec, M_SEL, +1)[0].mean()        # unrelated, +m: want log-odds up
            down = -effect(steerer, P[1], base[1], L, vec, M_SEL, -1)[0].mean()     # analogy, -m: want log-odds down
            return up, down
        up, down = score(v); ru, rd = score(r)
        gain = (up + down) / 2 - abs((ru + rd) / 2)
        rows.append(dict(layer=L, up=up, down=down, rand_up=ru, rand_down=rd, gain=gain))
        print(f"{L:5d} | {up:+23.3f} | {down:+23.3f} | {(ru + rd) / 2:+15.3f} | {gain:+.3f}   ({time.time() - t0:.0f}s)")
    sel_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sel_df.to_csv(RESULTS_DIR / "steering_layer_selection.csv", index=False)
    L = int(sel_df.loc[sel_df.gain.idxmax(), "layer"])
    v = V[L]
    np.save(RESULTS_DIR / "steering_vector.npy", v)
    print(f"\nChosen layer: {L}  (length of v: {np.linalg.norm(v):.2f} vs typical activation length "
          f"{typical[L]:.1f} -> one multiplier unit is {np.linalg.norm(v) / typical[L]:.1%} of an activation)")

    # ---- evaluate on VALIDATION ----
    ev = subset(M_va, N_EVAL_PER_GROUP)
    E = {g: mk_prompts(M_va, ev[g]) for g in (0, 1)}
    ebase = {g: steerer.logodds(E[g]) for g in (0, 1)}
    rand_dirs = [random_direction(rng, np.linalg.norm(v), d) for _ in range(K_RANDOM)]
    print(f"\nValidation: {len(E[0])} unrelated + {len(E[1])} analogy pairs, layer {L}, {K_RANDOM} random controls")
    print("(steered minus unsteered, in log-odds of yes vs no; paired per pair)\n")
    print("test                              |  m | real change [95% CI]  | random: mean / largest |change| | flips real vs random | yes+no mass change real")
    out = []
    for g, sign, name in ((0, +1, "unrelated, add +m*v (want yes UP)"), (1, -1, "analogy, add -m*v (want yes DOWN)")):
        for m in M_GRID:
            dl, dm, steered = effect(steerer, E[g], ebase[g], L, v, m, sign)
            want = dl if g == 0 else -dl                       # positive = moved in the intended direction
            lo, hi = boot_ci(want, rng)
            flip = lambda s: np.mean((ebase[g][0] < 0) & (s > 0)) if g == 0 else np.mean((ebase[g][0] > 0) & (s < 0))
            rand_changes, rand_flips = [], []
            for r in rand_dirs:
                rdl, _, rs = effect(steerer, E[g], ebase[g], L, r, m, sign)
                rand_changes.append((rdl if g == 0 else -rdl).mean()); rand_flips.append(flip(rs))
            print(f"{name:34s}| {m:2d} | {want.mean():+6.3f} [{lo:+.3f}, {hi:+.3f}] | "
                  f"{np.mean(rand_changes):+6.3f} / {np.max(np.abs(rand_changes)):.3f}"
                  f"{'':8s}| {flip(steered):4.0%} vs {np.mean(rand_flips):4.0%}      | {dm.mean():+.3f}")
            out.append(dict(test=name, m=m, real=want.mean(), lo=lo, hi=hi, rand_mean=np.mean(rand_changes),
                            rand_max_abs=np.max(np.abs(rand_changes)), flip_real=flip(steered),
                            flip_rand=np.mean(rand_flips), mass_change=dm.mean()))
    res = pd.DataFrame(out)
    res.to_csv(RESULTS_DIR / "steering_results.csv", index=False)

    print("\nHow to read it:")
    print("- 'real change' is positive when the log-odds moved the way we pushed (up for unrelated, down for")
    print("  analogy). A real effect: clearly positive, its confidence interval excludes 0, it GROWS with m,")
    print("  and it is clearly bigger than the random controls ('largest |change|').")
    print("- 'flips' = share of pairs whose yes/no decision actually crossed over, real vs random direction.")
    print("- 'yes+no mass change' strongly negative (e.g. below -1) at large m means the model is being")
    print("  broken (it stops answering yes/no), so those rows do not count as steering.")
    print("- The layer was chosen on TRAIN pairs; these numbers are on VALIDATION pairs.")
    print(f"\nSaved: {RESULTS_DIR}/steering_results.csv, steering_layer_selection.csv, steering_vector.npy")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, (name, g) in zip(axes, res.groupby("test")):
            ax.errorbar(g.m, g.real, yerr=[g.real - g.lo, g.hi - g.real], marker="o", label="real direction")
            ax.plot(g.m, g.rand_mean, "s--", color="gray", label="random (mean)")
            ax.fill_between(g.m, -g.rand_max_abs, g.rand_max_abs, color="gray", alpha=0.2, label="random (+/- largest)")
            ax.axhline(0, color="red", lw=1); ax.set_xscale("log", base=2)
            ax.set_title(name, fontsize=9); ax.set_xlabel("multiplier m")
        axes[0].set_ylabel("change in log-odds (intended direction)"); axes[0].legend(fontsize=7)
        fig.suptitle(f"Steering the yes/no judgement - {MODEL.split('/')[-1]}, layer {L}")
        fig.tight_layout(); fig.savefig(RESULTS_DIR / "steering.png", dpi=150)
        print(f"Plot saved to {RESULTS_DIR / 'steering.png'}")
    except ImportError:
        print("matplotlib not installed - skipped the plot")


if __name__ == "__main__":
    main()
