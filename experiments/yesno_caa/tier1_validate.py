"""
Tier 1 validation: is there an ANALOGY component in the steering vector, beyond the model's yes/no answer?

Directions compared (all rescaled to the same length as v at their layer, so a multiplier m means the
same size of push for every direction):
  v        the original vector: mean(analogy) - mean(unrelated) at the final prompt token
  v_perp   v with the model's "answer direction" removed (what is left once yes-vs-no is subtracted)
  c        the label direction measured while HOLDING THE MODEL'S ANSWER FIXED
           (mean(analogy) - mean(unrelated), computed separately among pairs where the model said yes
            and among pairs where it said no, then averaged)
  (random directions of the same length are the control everywhere)

Stages (run all, or pick: python tier1_validate.py 1 | 2 | 3 ; add --quick for a faster, noisier run):
  1  Label-only steering.  Choose a layer for v_perp and c on TRAIN pairs, then steer the original yes/no
     judgement on VALIDATION pairs.  (Does the part of v that is not the answer still move the model?)
  2  Different readouts, so the word "yes" is not the target:
       multiple choice  "(A) same underlying relationship ... (B) unrelated ...", the analogous option is
                        put in A or B at random per pair; score = analogous letter minus other letter
       same/different   "Answer (same or different):"
     A real analogy direction should move the model toward the analogous option in BOTH letter positions.
  3  Specificity.  Steer 30 yes/no questions that have nothing to do with analogy ("Is Paris the capital of
     France?"). A pure "say yes" switch moves these too; an analogy direction should not.
Then a SCORECARD summarises each direction.  TEST is not touched.

Needs: steer_yesno.py already run (for v's layer) and the cache_joint folder, MODEL in cache_activations.py.
Output: results/<model>/<contrast>/joint/tier1_*.csv, tier1_layers.json
Time:   roughly 17,000 short passes (about 25 minutes); --quick is about half.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cache_activations import MODEL
from cache_joint import PROMPT, load_joint
from find_layer import auc
import steer_yesno as sy
from steer_yesno import HFSteerer, boot_ci, random_direction

warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Settings ---------------------------------------------------------------
QUICK = "--quick" in sys.argv
N_SELECT = 30 if QUICK else 50          # train pairs per group, layer choice
N_EVAL = 50 if QUICK else 100           # validation pairs per group, stage 1
N_STAGE2 = 30 if QUICK else 40          # validation pairs per group, stage 2
K_RANDOM = 2                            # random control directions
M_SEL = 8.0                             # multiplier for the layer choice
M_GRID = (8,) if QUICK else (4, 8, 16)  # multipliers, stage 1
M_STAGE2 = (4, 8)                       # multipliers, stages 2 and 3
SEED = 0
RESULTS_DIR = sy.RESULTS_DIR
DIRS = ("v", "v_perp", "c")

ANALOG_TEXT = "They describe the same underlying relationship or process, even though they are about different things."
UNREL_TEXT = "They describe unrelated things and do not share a relationship."
MCQ = ("Sentence A: {s1}\nSentence B: {s2}\n\n"
       "Question: Which statement best describes how these two sentences relate?\n"
       "(A) {a}\n(B) {b}\n"
       "Answer (A or B):")
SAMEDIFF = ("Sentence A: {s1}\nSentence B: {s2}\n\n"
            "Question: Is the underlying relationship or process in these two sentences the same or different?\n"
            "Answer (same or different):")
CONTROL_PROMPT = "Question: {q}\nAnswer (yes or no):"
CONTROL_QUESTIONS = [
    "Is Paris the capital of France?", "Is 17 a prime number?", "Does water boil at 100 degrees Celsius at sea level?",
    "Does the Earth orbit the Sun?", "Is Python a programming language?", "Is 2 plus 2 equal to 4?",
    "Is Dublin in Ireland?", "Is ice frozen water?", "Do humans need oxygen to live?", "Is the Pacific an ocean?",
    "Does a square have four sides?", "Is gold a metal?", "Is Spanish spoken in Mexico?", "Is Monday a day of the week?",
    "Do cats have fur?",
    "Is the sun a planet?", "Is a whale a fish?", "Is 10 smaller than 5?", "Is the sky green?",
    "Is Tokyo the capital of Italy?", "Is 9 a prime number?", "Do fish live in trees?", "Is the moon made of cheese?",
    "Is a triangle a four-sided shape?", "Is Mars the largest planet in the solar system?", "Is salt a liquid at room temperature?",
    "Is Canada in Europe?", "Is 100 smaller than 50?", "Is the Atlantic a desert?", "Is a bicycle a living animal?",
]
# -----------------------------------------------------------------------------


class ChoiceSteerer(HFSteerer):
    """HFSteerer plus choice(): log P(any positive word) - log P(any negative word) for the next token."""

    def choice(self, prompts, pos_words, neg_words, layer=None, delta=None, batch_size=sy.BATCH_SIZE):
        torch = self.torch
        ids = lambda ws: torch.tensor(sorted({self.tok.encode(w, add_special_tokens=False)[0] for w in ws}), device=self.device)
        pos, neg = ids(pos_words), ids(neg_words)
        out = []
        for i in range(0, len(prompts), batch_size):
            enc = self.tok(list(prompts[i:i + batch_size]), return_tensors="pt", padding=True).to(self.device)
            handle = None
            if delta is not None:
                vec = torch.tensor(np.asarray(delta), dtype=torch.float32, device=self.device)

                def hook(module, inputs, output, vec=vec):
                    is_tuple = isinstance(output, tuple)
                    hs = (output[0] if is_tuple else output).clone()
                    if sy.POSITIONS == "last":
                        hs[:, -1, :] += vec
                    else:
                        hs += vec
                    return (hs,) + tuple(output[1:]) if is_tuple else hs

                handle = self.blocks[layer - 1].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    logits = self.model(**enc).logits[:, -1].float()
            finally:
                if handle is not None:
                    handle.remove()
            logp = torch.log_softmax(logits, dim=-1)
            out.append((torch.logsumexp(logp[:, pos], 1) - torch.logsumexp(logp[:, neg], 1)).cpu().numpy())
        return np.concatenate(out)


# ---------------------------------------------------------------- directions
def build_directions(A, y, ans):
    """Per-layer directions from the TRAIN cache: v, v_perp, c (arrays of shape (layers+1, d))."""
    X = A["last"]
    V = X[y == 1].mean(0) - X[y == 0].mean(0)
    B = 0.5 * sum(X[(y == g) & (ans == 1)].mean(0) - X[(y == g) & (ans == 0)].mean(0) for g in (0, 1))   # answer dir
    C = 0.5 * sum(X[(ans == a) & (y == 1)].mean(0) - X[(ans == a) & (y == 0)].mean(0) for a in (0, 1))  # label | answer
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    VP = V - (V * Bn).sum(1, keepdims=True) * Bn
    return dict(v=V, v_perp=VP, c=C)


def make_vec(D, name, L):
    """Direction `name` at layer L, rescaled to the length of v at that layer."""
    x = D[name][L]
    return x / (np.linalg.norm(x) + 1e-8) * np.linalg.norm(D["v"][L])


def prompts_of(M, idx, template=PROMPT):
    return [template.format(s1=M.s1[i], s2=M.s2[i]) for i in idx]


def subset(M, n, rng):
    y = M.label.to_numpy()
    return {g: rng.choice(np.flatnonzero(y == g), size=min(n, int((y == g).sum())), replace=False) for g in (0, 1)}


def load_context():
    rng = np.random.default_rng(SEED)
    A_tr, M_tr = load_joint("train")
    _, M_va = load_joint("val")
    y = M_tr.label.to_numpy()
    ans = ((A_tr["yes"] - A_tr["no"]) > 0).astype(int)
    D = build_directions(A_tr, y, ans)
    sel = pd.read_csv(RESULTS_DIR / "steering_layer_selection.csv")
    return dict(rng=rng, M_tr=M_tr, M_va=M_va, D=D, n_blocks=A_tr["last"].shape[1] - 1,
                d=D["v"].shape[1], layer_v=int(sel.sort_values("gain").iloc[-1].layer))


# -------------------------------------------------------------------- stage 1
def stage1(st, ctx):
    rng, D, d = ctx["rng"], ctx["D"], ctx["d"]
    t0 = time.time()
    sel = subset(ctx["M_tr"], N_SELECT, rng)
    P = {g: prompts_of(ctx["M_tr"], sel[g]) for g in (0, 1)}
    base = {g: st.logodds(P[g]) for g in (0, 1)}
    n = ctx["n_blocks"]
    cand = [l for l in range(int(round(0.25 * n)), int(round(0.9 * n)) + 1, 2) if 1 <= l <= n - 2]
    print(f"\n[1] Choosing a layer for v_perp and c on {N_SELECT} train pairs per group (multiplier {M_SEL:g})")
    print("layer | v_perp gain | c gain | random control effect")
    rows = []
    for L in cand:
        r = random_direction(rng, np.linalg.norm(D["v"][L]), d)

        def score(vec):
            up = st.logodds(P[0], layer=L, delta=M_SEL * vec)[0] - base[0][0]
            down = base[1][0] - st.logodds(P[1], layer=L, delta=-M_SEL * vec)[0]
            return up.mean(), down.mean()
        ru, rd = score(r)
        line = {}
        for name in ("v_perp", "c"):
            up, down = score(make_vec(D, name, L))
            gain = (up + down) / 2 - abs((ru + rd) / 2)
            rows.append(dict(direction=name, layer=L, up=up, down=down, gain=gain)); line[name] = gain
        print(f"{L:5d} | {line['v_perp']:+11.3f} | {line['c']:+6.3f} | {(ru + rd) / 2:+.3f}   ({time.time() - t0:.0f}s)")
    sel_df = pd.DataFrame(rows)
    sel_df.to_csv(RESULTS_DIR / "tier1_stage1_selection.csv", index=False)
    layers = {"v": ctx["layer_v"]}
    for name in ("v_perp", "c"):
        layers[name] = int(sel_df[sel_df.direction == name].sort_values("gain").iloc[-1].layer)
    json.dump(layers, open(RESULTS_DIR / "tier1_layers.json", "w"))
    print(f"Layers used: v -> {layers['v']} (from steer_yesno.py), v_perp -> {layers['v_perp']}, c -> {layers['c']}")

    # ---- evaluate on validation ----
    ev = subset(ctx["M_va"], N_EVAL, rng)
    E = {g: prompts_of(ctx["M_va"], ev[g]) for g in (0, 1)}
    eb = {g: st.logodds(E[g]) for g in (0, 1)}
    print(f"\n[1] Validation: {len(E[0])} unrelated + {len(E[1])} analogy pairs, {K_RANDOM} random controls per direction")
    print("direction (layer)   | test                     |  m | real change [95% CI]  | random largest | flips real/random")
    rows = []
    for name in DIRS:
        L = layers[name]
        vec = make_vec(D, name, L)
        rdirs = [random_direction(rng, np.linalg.norm(D["v"][L]), d) for _ in range(K_RANDOM)]
        for g, sign, tname in ((0, +1, "unrelated, +m*dir (yes up)"), (1, -1, "analogy,   -m*dir (yes down)")):
            for m in M_GRID:
                lo = st.logodds(E[g], layer=L, delta=sign * m * vec)[0]
                want = (lo - eb[g][0]) * (1 if g == 0 else -1)
                ci_lo, ci_hi = boot_ci(want, rng)
                rl = [st.logodds(E[g], layer=L, delta=sign * m * r)[0] for r in rdirs]
                rmax = max(abs(((x - eb[g][0]) * (1 if g == 0 else -1)).mean()) for x in rl)
                flip = lambda s: np.mean((eb[g][0] < 0) & (s > 0)) if g == 0 else np.mean((eb[g][0] > 0) & (s < 0))
                print(f"{name:7s} (layer {L:2d})  | {tname:24s} | {m:2g} | {want.mean():+6.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}] | "
                      f"{rmax:14.3f} | {flip(lo):4.0%} / {np.mean([flip(x) for x in rl]):4.0%}")
                rows.append(dict(direction=name, layer=L, test=tname.split(",")[0], m=m, real=want.mean(), lo=ci_lo, hi=ci_hi,
                                 rand_max=rmax, flip_real=flip(lo)))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tier1_stage1_eval.csv", index=False)
    print(f"[1] done in {time.time() - t0:.0f}s")


# -------------------------------------------------------------------- stage 2
def mcq_scores(st, prompts, analog_first, layer=None, delta=None):
    """Preference for the analogous option (positive = prefers it), whatever letter it has."""
    out = np.zeros(len(prompts))
    for first in (True, False):
        idx = np.flatnonzero(analog_first == first)
        if len(idx) == 0:
            continue
        pos, neg = (["A", " A"], ["B", " B"]) if first else (["B", " B"], ["A", " A"])
        out[idx] = st.choice([prompts[i] for i in idx], pos, neg, layer=layer, delta=delta)
    return out


def stage2(st, ctx):
    rng, D, d = ctx["rng"], ctx["D"], ctx["d"]
    layers = json.load(open(RESULTS_DIR / "tier1_layers.json"))
    t0 = time.time()
    ev = subset(ctx["M_va"], N_STAGE2, rng)
    idx = np.concatenate([ev[0], ev[1]]); y = ctx["M_va"].label.to_numpy()[idx]
    analog_first = np.random.default_rng(SEED + 1).random(len(idx)) < 0.5
    texts = [(ANALOG_TEXT, UNREL_TEXT) if f else (UNREL_TEXT, ANALOG_TEXT) for f in analog_first]
    P_mcq = [MCQ.format(s1=ctx["M_va"].s1[i], s2=ctx["M_va"].s2[i], a=a, b=b) for i, (a, b) in zip(idx, texts)]
    P_sd = prompts_of(ctx["M_va"], idx, SAMEDIFF)
    readouts = {
        "multiple choice": (lambda layer=None, delta=None: mcq_scores(st, P_mcq, analog_first, layer, delta)),
        "same/different": (lambda layer=None, delta=None: st.choice(P_sd, ["same", " same", "Same", " Same"],
                                                                    ["different", " different", "Different", " Different"],
                                                                    layer=layer, delta=delta)),
    }
    rows = []
    for rname, fn in readouts.items():
        base = fn()
        print(f"\n[2] {rname}: the model's own preference separates the groups with AUC {auc(base, y):.3f} "
              f"(0.5 = no); prefers the analogous option on {np.mean(base > 0):.0%} of pairs")
        if rname == "multiple choice":
            says_a = np.where(analog_first, base > 0, base < 0)
            print(f"    chooses letter A on {says_a.mean():.0%} of pairs (letter bias if far from 50%)")
        print("direction (layer)   |  m | +m*dir change [95% CI] | by letter: analogy in A / in B | -m*dir change | random largest")
        for name in DIRS:
            L = layers[name]; vec = make_vec(D, name, L)
            rdirs = [random_direction(rng, np.linalg.norm(D["v"][L]), d) for _ in range(K_RANDOM)]
            for m in M_STAGE2:
                up = fn(layer=L, delta=m * vec) - base
                dn = fn(layer=L, delta=-m * vec) - base
                rmax = max(abs((fn(layer=L, delta=s * m * r) - base).mean()) for r in rdirs for s in (+1, -1))
                lo, hi = boot_ci(up, rng)
                by = (up[analog_first].mean(), up[~analog_first].mean()) if rname == "multiple choice" else (np.nan, np.nan)
                print(f"{name:7s} (layer {L:2d})  | {m:2g} | {up.mean():+7.3f} [{lo:+.3f}, {hi:+.3f}] | "
                      f"{by[0]:+8.3f} / {by[1]:+8.3f}        | {dn.mean():+8.3f}     | {rmax:.3f}")
                rows.append(dict(readout=rname, direction=name, layer=L, m=m, up=up.mean(), lo=lo, hi=hi,
                                 up_letter_A=by[0], up_letter_B=by[1], down=dn.mean(), rand_max=rmax, own_auc=auc(base, y)))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tier1_stage2.csv", index=False)
    print(f"[2] done in {time.time() - t0:.0f}s")


# -------------------------------------------------------------------- stage 3
def stage3(st, ctx):
    rng, D, d = ctx["rng"], ctx["D"], ctx["d"]
    layers = json.load(open(RESULTS_DIR / "tier1_layers.json"))
    t0 = time.time()
    P = [CONTROL_PROMPT.format(q=q) for q in CONTROL_QUESTIONS]
    base = st.logodds(P)[0]
    print(f"\n[3] Control yes/no questions unrelated to analogy ({len(P)} questions): the model says yes on {np.mean(base > 0):.0%}")
    print("direction (layer)   |  m | +m*dir change [95% CI] | -m*dir change | random largest")
    rows = []
    for name in DIRS:
        L = layers[name]; vec = make_vec(D, name, L)
        rdirs = [random_direction(rng, np.linalg.norm(D["v"][L]), d) for _ in range(K_RANDOM)]
        for m in M_STAGE2:
            up = st.logodds(P, layer=L, delta=m * vec)[0] - base
            dn = st.logodds(P, layer=L, delta=-m * vec)[0] - base
            rmax = max(abs((st.logodds(P, layer=L, delta=s * m * r)[0] - base).mean()) for r in rdirs for s in (+1, -1))
            lo, hi = boot_ci(up, rng)
            print(f"{name:7s} (layer {L:2d})  | {m:2g} | {up.mean():+7.3f} [{lo:+.3f}, {hi:+.3f}] | {dn.mean():+8.3f}     | {rmax:.3f}")
            rows.append(dict(direction=name, layer=L, m=m, up=up.mean(), lo=lo, hi=hi, down=dn.mean(), rand_max=rmax))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tier1_stage3.csv", index=False)
    print(f"[3] done in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ scorecard
def scorecard():
    paths = {k: RESULTS_DIR / f"tier1_{k}.csv" for k in ("stage1_eval", "stage2", "stage3")}
    if not all(p.exists() for p in paths.values()):
        print("\nScorecard needs all three stages to have been run.")
        return
    s1, s2, s3 = (pd.read_csv(paths[k]) for k in ("stage1_eval", "stage2", "stage3"))
    m = 8 if 8 in set(s1.m) else int(s1.m.max())
    print(f"\n================ SCORECARD (multiplier {m}; 'pass' rules are rough guides, read the tables) ================")
    print("direction | 1 steers the yes/no answer | 2a multiple choice, both letters | 2b same/different | 3 specific to analogy")
    for name in DIRS:
        a = s1[(s1.direction == name) & (s1.m == m)]
        t1 = bool(len(a) == 2 and (a.lo > 0).all() and (a.real > 3 * a.rand_max).all())
        b = s2[(s2.direction == name) & (s2.readout == "multiple choice") & (s2.m == m)].iloc[0]
        t2 = bool(b.lo > 0 and b.up > 3 * b.rand_max and b.up_letter_A > 0 and b.up_letter_B > 0 and b.down < 0)
        c = s2[(s2.direction == name) & (s2.readout == "same/different") & (s2.m == m)].iloc[0]
        t3 = bool(c.lo > 0 and c.up > 3 * c.rand_max and c.down < 0)
        ctrl = s3[(s3.direction == name) & (s3.m == m)].iloc[0]
        effect = a[a.test == "unrelated"].real.iloc[0]
        ratio = abs(ctrl.up) / effect if effect > 0 else np.nan
        spec = "SPECIFIC" if ratio < 0.25 else "NOT specific" if ratio > 0.5 else "partly"
        print(f"{name:9s} | {'pass' if t1 else 'fail':26s} | {'pass' if t2 else 'fail':32s} | {'pass' if t3 else 'fail':18s} | "
              f"{spec} (control effect is {ratio:.0%} of the analogy-prompt effect)")
    print("\nWhat a real analogy direction looks like: pass, pass, pass, SPECIFIC - for v_perp and/or c.")
    print("What a pure yes/no switch looks like: v passes 1, fails 2a/2b, NOT specific; v_perp and c do little.")


def main():
    stages = [a for a in sys.argv[1:] if a in ("1", "2", "3")] or ["1", "2", "3"]
    ctx = load_context()
    st = ChoiceSteerer()
    run(st, ctx, stages)


def run(st, ctx, stages):
    print(f"Model {MODEL} | directions {DIRS} | stages {stages}{' | QUICK' if QUICK else ''}")
    if "1" in stages:
        stage1(st, ctx)
    if "2" in stages:
        stage2(st, ctx)
    if "3" in stages:
        stage3(st, ctx)
    scorecard()
    print(f"\nSaved to {RESULTS_DIR}/tier1_*.csv")


if __name__ == "__main__":
    main()
