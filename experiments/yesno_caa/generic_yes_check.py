"""
Is v (the analogy steering vector) just a GENERIC yes direction?

We build a yes-vs-no direction g from ~220 simple questions that have nothing to do with analogy
(number comparisons, capitals, categories, arithmetic), in the same format as our prompt
("Question: ...\nAnswer (yes or no):"), and ask how much of v it explains.

  g        = mean(final-token activation | model said yes) - mean(... | model said no) on the generic questions
  g_truth  = same but split by whether the true answer is yes
For each layer we print:
  cos(v,g)    cosine of the analogy vector with the generic yes direction
  cos(b,g)    cosine of the model's analogy-prompt ANSWER direction b with g   <- how well ANY yes direction carries over
              between two very different prompts. If cos(v,g) is about cos(b,g), v is no less generic than b.
  ceiling     split-half cosine of g with itself (how noisy g is)
  AUC(g)      validation AUC of the projection on g for the analogy label (does a no-analogy yes direction already
              separate analogy from unrelated?)
  rest        what is left of v once g is removed:  share of |v|^2, and its validation AUC for the label with the
              model's answer held fixed (>0.5 = something beyond generic-yes, but see caveat in the printout)

Run:   python generic_yes_check.py        
Needs: cache_joint train/val for MODEL (as for direction_check.py); steering_layer_selection.csv (optional).
Output: results/<model>/<contrast>/joint/generic_yes_check.csv, generic_yes_cache.npz
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cache_activations import MODEL
from cache_joint import load_joint
from storyanalogy_loader import CONTRAST
from find_layer import auc
from direction_check import weighted_auc

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 0
N_SPLIT = 30
RESULTS_DIR = Path("results") / MODEL.replace("/", "__") / CONTRAST / "joint"
PROMPT = "Question: {q}\nAnswer (yes or no):"


def make_questions(seed=0):
    r = np.random.default_rng(seed)
    qs = []
    for _ in range(60):                                   # number comparisons
        a, b = r.choice(np.arange(1, 100), 2, replace=False)
        qs.append((f"Is {a} greater than {b}?", a > b))
    caps = [("France", "Paris"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Germany", "Berlin"), ("Ireland", "Dublin"),
            ("Japan", "Tokyo"), ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("Russia", "Moscow"), ("Norway", "Oslo"),
            ("Greece", "Athens"), ("Poland", "Warsaw"), ("Portugal", "Lisbon"), ("Austria", "Vienna"), ("Cuba", "Havana"),
            ("Peru", "Lima"), ("Kenya", "Nairobi"), ("Sweden", "Stockholm"), ("Turkey", "Ankara"), ("China", "Beijing")]
    for i, (c, k) in enumerate(caps):
        qs.append((f"Is {k} the capital of {c}?", True))
        qs.append((f"Is {caps[(i + 7) % len(caps)][1]} the capital of {c}?", False))
    cats = {"an animal": ["dog", "horse", "eagle", "salmon", "frog", "whale", "bee", "snake"],
            "a fruit": ["apple", "banana", "grape", "mango", "pear", "plum", "cherry", "lemon"],
            "a vehicle": ["car", "bus", "train", "bicycle", "truck", "tractor", "ferry", "van"],
            "a tool": ["hammer", "saw", "wrench", "drill", "chisel", "spanner", "shovel", "pliers"]}
    names = list(cats)
    for ci, (cat, items) in enumerate(cats.items()):
        for j, x in enumerate(items):
            qs.append((f"Is a {x} {cat}?", True))
            other = cats[names[(ci + 1) % 4]][j]
            qs.append((f"Is a {other} {cat}?", False))
    for _ in range(40):                                   # arithmetic
        a, b = r.integers(2, 50, 2)
        c = a + b if r.random() < 0.5 else a + b + int(r.choice([-3, -2, -1, 1, 2, 3]))
        qs.append((f"Is {a} plus {b} equal to {c}?", bool(c == a + b)))
    return qs


def get_generic_cache(get_features):
    path = RESULTS_DIR / "generic_yes_cache.npz"
    if path.exists():
        d = np.load(path)
        return d["X"].astype(np.float32), d["yes"], d["no"], d["truth"]
    qs = make_questions(SEED)
    X, yes, no = [], [], []
    for i, (q, _) in enumerate(qs, 1):
        a, y, n = get_features(PROMPT.format(q=q))
        X.append(a); yes.append(y); no.append(n)
        if i % 50 == 0 or i == len(qs):
            print(f"  generic questions {i}/{len(qs)}")
    X = np.stack(X).astype(np.float32); yes = np.array(yes); no = np.array(no); truth = np.array([t for _, t in qs], bool)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X.astype(np.float16), yes=yes, no=no, truth=truth)
    return X, yes, no, truth


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def diff_dir(X, mask):
    return X[mask].mean(0) - X[~mask].mean(0)


def main(get_features=None):
    rng = np.random.default_rng(SEED)
    if get_features is None:
        from cache_joint import make_backend
        get_features = make_backend()
    Xg, yes, no, truth = get_generic_cache(get_features)
    said = (yes - no) > 0
    print(f"\nGeneric questions: {len(said)} | model says yes on {said.mean():.0%} | true answer is yes on {truth.mean():.0%} | "
          f"model correct on {np.mean(said == truth):.0%}")
    if min(said.sum(), (~said).sum()) < 20:
        print("WARNING: the model almost always gives the same answer, so g (yes minus no) is noisy; g_truth is shown too.")
    A_tr, M_tr = load_joint("train"); A_va, M_va = load_joint("val")
    y_tr, y_va = M_tr.label.to_numpy(), M_va.label.to_numpy()
    ans_tr = ((A_tr["yes"] - A_tr["no"]) > 0); ans_va = ((A_va["yes"] - A_va["no"]) > 0).astype(int)
    nl = A_tr["last"].shape[1]
    sel = RESULTS_DIR / "steering_layer_selection.csv"
    chosen = int(pd.read_csv(sel).sort_values("gain").iloc[-1].layer) if sel.exists() else None
    rows = []
    for L in range(nl):
        Xt, Xv, G = A_tr["last"][:, L], A_va["last"][:, L], Xg[:, L]
        v = diff_dir(Xt, y_tr == 1)
        b = 0.5 * sum(Xt[(y_tr == g) & ans_tr].mean(0) - Xt[(y_tr == g) & ~ans_tr].mean(0) for g in (0, 1))
        c = 0.5 * sum(Xt[(ans_tr == a) & (y_tr == 1)].mean(0) - Xt[(ans_tr == a) & (y_tr == 0)].mean(0) for a in (True, False))
        g = diff_dir(G, said); gt = diff_dir(G, truth)
        cs = []
        for _ in range(20):                               # split-half reliability of g
            p = rng.permutation(len(said)); h1, h2 = p[:len(p) // 2], p[len(p) // 2:]
            if said[h1].all() or (~said[h1]).all() or said[h2].all() or (~said[h2]).all():
                continue
            cs.append(cos(diff_dir(G[h1], said[h1]), diff_dir(G[h2], said[h2])))
        gh = g / (np.linalg.norm(g) + 1e-8)
        vr = v - (v @ gh) * gh
        rows.append(dict(layer=L, cos_v_g=cos(v, g), cos_b_g=cos(b, g), cos_c_g=cos(c, g), cos_v_gtruth=cos(v, gt),
                         ceiling=float(np.mean(cs)) if cs else np.nan,
                         auc_g_label=auc(Xv @ gh, y_va), auc_v_label=auc(Xv @ (v / np.linalg.norm(v)), y_va),
                         rest_share=float((vr @ vr) / (v @ v)),
                         auc_rest_given_answer=weighted_auc(Xv @ (vr / (np.linalg.norm(vr) + 1e-8)), y_va, ans_va)))
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True); df.to_csv(RESULTS_DIR / "generic_yes_check.csv", index=False)
    show = sorted(set(range(0, nl, 3)) | {nl - 1} | ({chosen} if chosen else set()))
    print("\nlayer | cos(v,g) | cos(b,g) | cos(c,g) | ceiling | AUC label: g / v | rest of v: share of |v|^2, AUC label|answer")
    for L in show:
        r = df[df.layer == L].iloc[0]
        print(f"{L:5d} | {r.cos_v_g:+8.2f} | {r.cos_b_g:+8.2f} | {r.cos_c_g:+8.2f} | {r.ceiling:7.2f} | {r.auc_g_label:.3f} / {r.auc_v_label:.3f}"
              f" | {r.rest_share:5.2f}, {r.auc_rest_given_answer:.3f} {'<- steering layer' if L == chosen else ''}")
    if chosen is not None:
        r = df[df.layer == chosen].iloc[0]
        print(f"\nAt the steering layer {chosen}: cos(v,g)={r.cos_v_g:+.2f} vs cos(b,g)={r.cos_b_g:+.2f} (ceiling {r.ceiling:.2f}); "
              f"{r.rest_share:.0%} of |v|^2 is NOT along g.")
        if abs(r.cos_v_g) > 0.6 * r.ceiling:
            print("  -> v is close to a generic yes direction (it overlaps g about as much as g overlaps itself).")
        elif abs(r.cos_v_g) > 1.5 * abs(r.cos_b_g) or abs(r.cos_v_g) < 0.5 * abs(r.cos_b_g):
            print("  -> v and b differ in how much they share with g: look at the table.")
        else:
            print("  -> v shares with g about what b does: v is as generic as the model's own answer direction, just not an exact match.")
    print("\nHow to read it:")
    print("- g comes from questions with no analogy in them. A big cos(v,g) means the analogy vector is mostly a generic yes direction.")
    print("- Directions from different prompts never match exactly, so compare cos(v,g) with cos(b,g) and with the ceiling, not with 1.")
    print("- 'rest of v' is v with g removed. Its AUC with the answer held fixed shows whether anything beyond generic-yes still")
    print("  separates analogy from unrelated; this is a read-out, not a causal claim (steer it next if it is clearly above 0.5).")
    print("- Random directions have cosine about +/-0.03 here.")
    print(f"\nSaved: {RESULTS_DIR}/generic_yes_check.csv")


if __name__ == "__main__":
    main()
