"""
Tier 1b: is v just the model's ANSWER direction?  (decisive control after tier1_validate.py)

Steers with four directions, all rescaled to the length of v at the same layer, on four readouts:
  v       original analogy-minus-unrelated direction
  b       the model's ANSWER direction (mean activation when it said yes minus no, label held fixed)
  v_perp  v with b removed  (tested at v's layer, either sign allowed)
  c       label direction with the answer held fixed
Readouts (validation pairs, same prompts for every direction):
  yes/no        the original question
  multiple choice (analogous option randomly in A or B); effect measured as "toward the analogous option"
  same/different
  control       30 yes/no questions with nothing to do with analogy
Key idea: if v does nothing that b does not also do, then v is the answer direction. If v moves the
multiple-choice readout (in both letter positions) MORE than b does for the same push on the control
questions, v has an analogy component.  'selectivity' = multiple-choice effect / control effect.

Run:   python tier1b_answer_control.py            (layer = v's layer from tier1_layers.json)
       python tier1b_answer_control.py 17 21      (extra layers can be given)
       add --quick for 30 pairs per group
Needs: tier1_validate.py in the same folder and already run (tier1_layers.json). TEST untouched.
Time:  about 10-15 minutes per layer.
Output: results/<model>/<contrast>/joint/tier1b_L<layer>.csv
"""
import json, sys, time
import numpy as np, pandas as pd
import tier1_validate as t
from tier1_validate import (ChoiceSteerer, make_vec, prompts_of, subset, MCQ, SAMEDIFF, ANALOG_TEXT, UNREL_TEXT,
                            CONTROL_PROMPT, CONTROL_QUESTIONS, mcq_scores)
from cache_joint import load_joint
from steer_yesno import boot_ci, random_direction

QUICK = "--quick" in sys.argv
N = 30 if QUICK else 50
MS = (4, 8)
K_RANDOM = 2
SEED = 0
NAMES = ("v", "b", "v_perp", "c")


def answer_dir(A, y, ans):
    X = A["last"]
    return 0.5 * sum(X[(y == g) & (ans == 1)].mean(0) - X[(y == g) & (ans == 0)].mean(0) for g in (0, 1))


def main(st=None, ctx=None, layers=None):
    ctx = ctx or t.load_context()
    rng = np.random.default_rng(SEED)
    A_tr, M_tr = load_joint("train")
    y_tr = M_tr.label.to_numpy(); ans = ((A_tr["yes"] - A_tr["no"]) > 0).astype(int)
    D = dict(ctx["D"]); D["b"] = answer_dir(A_tr, y_tr, ans)
    if layers is None:
        layers = [int(a) for a in sys.argv[1:] if a.isdigit()] or [json.load(open(t.RESULTS_DIR / "tier1_layers.json"))["v"]]
    st = st or ChoiceSteerer()
    M = ctx["M_va"]; ev = subset(M, N, rng)
    idx = np.concatenate([ev[0], ev[1]]); y = M.label.to_numpy()[idx]
    af = np.random.default_rng(SEED + 1).random(len(idx)) < 0.5
    texts = [(ANALOG_TEXT, UNREL_TEXT) if f else (UNREL_TEXT, ANALOG_TEXT) for f in af]
    P_yn = prompts_of(M, idx)
    P_mc = [MCQ.format(s1=M.s1[i], s2=M.s2[i], a=a, b=b) for i, (a, b) in zip(idx, texts)]
    P_sd = prompts_of(M, idx, SAMEDIFF)
    P_ct = [CONTROL_PROMPT.format(q=q) for q in CONTROL_QUESTIONS]
    SDP, SDN = ["same", " same", "Same", " Same"], ["different", " different", "Different", " Different"]
    read = {
        "yesno": lambda L=None, d=None: st.logodds(P_yn, layer=L, delta=d)[0],
        "mcq": lambda L=None, d=None: mcq_scores(st, P_mc, af, L, d),
        "samediff": lambda L=None, d=None: st.choice(P_sd, SDP, SDN, layer=L, delta=d),
        "control": lambda L=None, d=None: st.logodds(P_ct, layer=L, delta=d)[0],
    }
    base = {k: f() for k, f in read.items()}
    print(f"Model {t.MODEL} | {len(idx)} val pairs | layers {layers} | directions {NAMES}{' | QUICK' if QUICK else ''}")
    for L in layers:
        t0 = time.time()
        cosb = {n: float(D[n][L] @ D["b"][L] / (np.linalg.norm(D[n][L]) * np.linalg.norm(D["b"][L]) + 1e-8)) for n in NAMES}
        print(f"\n=== layer {L}: cosine with b: " + ", ".join(f"{n} {cosb[n]:+.2f}" for n in NAMES))
        rd = [random_direction(rng, np.linalg.norm(D["v"][L]), ctx["d"]) for _ in range(K_RANDOM)]
        rows = []
        for m in MS:
            rnd = {k: max(abs((f(L, s * m * r) - base[k]).mean()) for r in rd for s in (1, -1)) for k, f in read.items()}
            print(f"\n m={m}   (random-direction largest |change|: yes/no {rnd['yesno']:.2f}, mcq {rnd['mcq']:.2f}, "
                  f"same/diff {rnd['samediff']:.2f}, control {rnd['control']:.2f})")
            print("dir    | sign | yes/no | mcq (A / B)        mean [95% CI]       | same/diff | control | selectivity mcq/control")
            for n in NAMES:
                vec = make_vec(D, n, L)
                for s in (1, -1):
                    ch = {k: f(L, s * m * vec) - base[k] for k, f in read.items()}
                    mc = ch["mcq"]; a, b_ = mc[af].mean(), mc[~af].mean()
                    lo, hi = boot_ci(mc, rng)
                    ctl = ch["control"].mean()
                    sel = (min(a, b_) if s > 0 else -max(a, b_)) / abs(ctl) if abs(ctl) > 0.05 else np.nan
                    print(f"{n:6s} | {'+' if s > 0 else '-'}    | {ch['yesno'].mean():+6.2f} | {a:+6.2f} / {b_:+6.2f}  {mc.mean():+6.2f} [{lo:+.2f}, {hi:+.2f}] | "
                          f"{ch['samediff'].mean():+9.2f} | {ctl:+7.2f} | {sel:+.2f}")
                    rows.append(dict(layer=L, m=m, direction=n, sign=s, yesno=ch["yesno"].mean(), mcq_A=a, mcq_B=b_, mcq=mc.mean(),
                                     mcq_lo=lo, mcq_hi=hi, samediff=ch["samediff"].mean(), control=ctl, selectivity=sel,
                                     rnd_mcq=rnd["mcq"], rnd_yesno=rnd["yesno"]))
        df = pd.DataFrame(rows)
        df.to_csv(t.RESULTS_DIR / f"tier1b_L{L}.csv", index=False)
        print(f"\n[layer {L}] done in {time.time() - t0:.0f}s")
        k = df[(df.m == MS[0]) & (df.sign == 1)].set_index("direction")
        print("\nVERDICT at m=%d, +direction (selectivity = multiple-choice effect in the weaker letter / control-question effect):" % MS[0])
        for n in NAMES:
            print(f"  {n:6s} selectivity {k.loc[n, 'selectivity']:+.2f}   mcq {k.loc[n, 'mcq']:+.2f}   control {k.loc[n, 'control']:+.2f}")
        sv, sb = k.loc["v", "selectivity"], k.loc["b", "selectivity"]
        if np.isnan(sv) or np.isnan(sb):
            print("  -> a selectivity is undefined (control effect tiny); read the table.")
        elif sv > 2 * max(sb, 0.05) and k.loc["v", "mcq_lo"] > 0:
            print("  -> v is clearly MORE selective than the answer direction b: evidence for an analogy component.")
        elif sv > 1.3 * max(sb, 0.05):
            print("  -> v somewhat more selective than b: suggestive, not conclusive.")
        else:
            print("  -> v is no more selective than b: v behaves like the answer direction.")
    print("\nHow to read it: b is the model's own yes-vs-no direction. If b moves the multiple choice just as much as v while also")
    print("moving the control questions, then v's earlier multiple-choice effect was the answer direction. Negative selectivity for -dir")
    print("is expected to be a mirror image; compare the + rows.")


if __name__ == "__main__":
    main()
