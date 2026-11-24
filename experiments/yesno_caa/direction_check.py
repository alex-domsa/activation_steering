"""
Direction check (offline, no GPU): is the steering direction v about ANALOGY, or just about the model's ANSWER?

Why: v = mean(analogy) - mean(unrelated) at the final token of the joint prompt. The model also says "yes"
more often to analogy pairs, so v could simply be "the model is about to say yes".  The two can be pulled
apart because the model's answer and the true label disagree on many pairs:

                         model says YES        model says NO
    analogy pair         (agree)               (model wrong)
    unrelated pair       (model wrong)         (agree)

Two questions, answered with the validation pairs, scoring each pair by its projection on v (built on TRAIN):
  1. LABEL given ANSWER: among pairs where the model gave the SAME answer, does v still separate analogy from
     unrelated?   (~0.5 -> v is only the answer;  clearly above 0.5 -> v carries the label beyond the answer)
  2. ANSWER given LABEL: among pairs with the SAME true label, does v separate "model said yes" from "said no"?
     (high -> v tracks what the model is about to say)
Plus the cosine between v and the model's own answer direction (mean activation when it said yes minus when
it said no, holding the true label fixed).

Run:   python direction_check.py
Needs: cache_joint/<model>/<contrast>/ (train, val) for the MODEL set in cache_activations.py.
Output: results/<model>/<contrast>/joint/direction_check.csv and .png
"""

import numpy as np
import pandas as pd
from pathlib import Path

from cache_activations import MODEL
from cache_joint import load_joint, JOINT_DIR
from storyanalogy_loader import CONTRAST
from find_layer import auc

N_BOOT = 300
SEED = 0
RESULTS_DIR = Path("results") / MODEL.replace("/", "__") / CONTRAST / "joint"


def cell_mean(X, mask):
    return X[mask].mean(0)


def weighted_auc(scores, target, group):
    """AUC of `scores` for `target` (0/1) inside each value of `group`, averaged with weight n1*n0."""
    num = den = 0.0
    for g in np.unique(group):
        m = group == g
        n1, n0 = int((target[m] == 1).sum()), int((target[m] == 0).sum())
        if n1 < 5 or n0 < 5:
            continue
        num += n1 * n0 * auc(scores[m], target[m]); den += n1 * n0
    return num / den if den else np.nan


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    rng = np.random.default_rng(SEED)
    A_tr, M_tr = load_joint("train")
    A_va, M_va = load_joint("val")
    y_tr = M_tr.label.to_numpy(); y_va = M_va.label.to_numpy()
    ans_tr = ((A_tr["yes"] - A_tr["no"]) > 0).astype(int)      # 1 = the model answered "yes"
    ans_va = ((A_va["yes"] - A_va["no"]) > 0).astype(int)
    n_layers = A_tr["last"].shape[1]
    print(f"Joint cache: {JOINT_DIR} | layers {n_layers}")
    print("\nTrain cells (label x model answer):")
    for g, gn in ((1, "analogy"), (0, "unrelated")):
        print(f"  {gn:9s}: said yes {int(((y_tr == g) & (ans_tr == 1)).sum()):4d} | said no {int(((y_tr == g) & (ans_tr == 0)).sum()):4d}")
    print("Validation cells:")
    for g, gn in ((1, "analogy"), (0, "unrelated")):
        print(f"  {gn:9s}: said yes {int(((y_va == g) & (ans_va == 1)).sum()):4d} | said no {int(((y_va == g) & (ans_va == 0)).sum()):4d}")

    sel = Path("results") / MODEL.replace("/", "__") / CONTRAST / "joint" / "steering_layer_selection.csv"
    chosen = int(pd.read_csv(sel).sort_values("gain").iloc[-1].layer) if sel.exists() else None

    rows = []
    for L in range(n_layers):
        Xtr, Xva = A_tr["last"][:, L], A_va["last"][:, L]
        v = cell_mean(Xtr, y_tr == 1) - cell_mean(Xtr, y_tr == 0)                     # the steering direction
        # answer direction holding the label fixed, label direction holding the answer fixed
        b = 0.5 * sum(cell_mean(Xtr, (y_tr == g) & (ans_tr == 1)) - cell_mean(Xtr, (y_tr == g) & (ans_tr == 0)) for g in (0, 1))
        c = 0.5 * sum(cell_mean(Xtr, (ans_tr == a) & (y_tr == 1)) - cell_mean(Xtr, (ans_tr == a) & (y_tr == 0)) for a in (0, 1))
        s = Xva @ (v / (np.linalg.norm(v) + 1e-8))
        rows.append(dict(layer=L, cos_v_answer=cos(v, b), cos_answer_vs_label_given_answer=cos(b, c),
                         auc_label=auc(s, y_va), auc_label_given_answer=weighted_auc(s, y_va, ans_va),
                         auc_answer_given_label=weighted_auc(s, ans_va, y_va)))
    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / "direction_check.csv", index=False)

    show = sorted(set(range(0, n_layers, 3)) | {n_layers - 1} | ({chosen} if chosen else set()))
    print("\n================ per layer (validation) ================")
    print("layer | AUC label | AUC label,    | AUC answer,   | cosine of v with the    | note")
    print("      | (overall) | answer fixed  | label fixed   | model's answer direction|")
    for L in show:
        r = res[res.layer == L].iloc[0]
        print(f"{L:5d} | {r.auc_label:9.3f} | {r.auc_label_given_answer:13.3f} | {r.auc_answer_given_label:13.3f} | "
              f"{r.cos_v_answer:+23.3f} | {'<- steering layer' if L == chosen else ''}")

    if chosen is not None:
        L = chosen
        Xtr, Xva = A_tr["last"][:, L], A_va["last"][:, L]
        v = cell_mean(Xtr, y_tr == 1) - cell_mean(Xtr, y_tr == 0)
        s = Xva @ (v / (np.linalg.norm(v) + 1e-8))
        la, aa = [], []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y_va), len(y_va))
            la.append(weighted_auc(s[i], y_va[i], ans_va[i])); aa.append(weighted_auc(s[i], ans_va[i], y_va[i]))
        print(f"\nAt the steering layer {L} (95% bootstrap intervals):")
        print(f"  label given answer : {weighted_auc(s, y_va, ans_va):.3f} [{np.nanpercentile(la, 2.5):.3f}, {np.nanpercentile(la, 97.5):.3f}]")
        print(f"  answer given label : {weighted_auc(s, ans_va, y_va):.3f} [{np.nanpercentile(aa, 2.5):.3f}, {np.nanpercentile(aa, 97.5):.3f}]")

    print("\nHow to read it:")
    print("- 'AUC label | answer' is the key number. Near 0.5 (interval includes 0.5) -> v carries NO information")
    print("  about the true label beyond what the model is about to answer: v is an answer direction.")
    print("  Clearly above 0.5 -> v carries something about the label beyond the answer.")
    print("- 'AUC answer | label' high -> v tracks the model's coming answer. Both can be high: v can be a mixture.")
    print("- The cosine of v with the model's own answer direction: near +1 -> v is mostly the answer direction.")
    print("  Near 0 -> unrelated to it. (Random directions in this many dimensions have cosine of about +/-0.03.)")
    print("- Cells are small for the 'model wrong' cases; treat differences under ~0.08 as noise.")
    print(f"\nSaved: {RESULTS_DIR}/direction_check.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(res.layer, res.auc_label, label="label (overall)")
        ax.plot(res.layer, res.auc_label_given_answer, label="label, answer held fixed")
        ax.plot(res.layer, res.auc_answer_given_label, label="answer, label held fixed")
        ax.axhline(0.5, color="red", lw=1)
        if chosen is not None:
            ax.axvline(chosen, color="gray", ls=":")
        ax.set_xlabel("layer"); ax.set_ylabel("validation AUC of projection on v"); ax.legend(fontsize=8)
        ax.set_title(f"Is v about the label or the answer? - {MODEL.split('/')[-1]}")
        fig.tight_layout(); fig.savefig(RESULTS_DIR / "direction_check.png", dpi=150)
        print(f"Plot saved to {RESULTS_DIR / 'direction_check.png'}")
    except ImportError:
        print("matplotlib not installed - skipped the plot")


if __name__ == "__main__":
    main()
