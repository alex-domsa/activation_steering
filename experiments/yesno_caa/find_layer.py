"""
Layer scan: at which layers are ANALOGY pairs separable from UNRELATED pairs?

Contrast (set in storyanalogy_loader.py, CONTRAST = "analogy_vs_unrelated"):
    analogy   = low entity overlap, high relation overlap   (label 1)
    unrelated = low entity overlap, low relation overlap    (label 0)
Entity overlap is matched between the groups, so what differs is RELATIONAL similarity.
(Run with CONTRAST = "analogy_vs_similar" in the loader to compare against the old setup.)

Uses TRAIN to build the direction / fit the probe and VALIDATION to score it.
The TEST split is never touched here.


Score = AUC on validation (0.5 = chance, 1.0 = perfect):
  caa   : CAA direction = mean(feature | analogy) - mean(feature | unrelated), built on train,
          each validation pair scored by projecting onto it
  probe : logistic-regression classifier trained on train
Baselines are printed at the end. A layer only counts if it clearly beats them.
"""

import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from pathlib import Path

from cache_activations import load_cached, CACHE_DIR
from first_look import row_cosine

# ---- Settings ---------------------------------------------------------------
POOLINGS = ("last", "mean")
FEATURES = {                                   # pair-level features built from s1 and s2
    "avg":     lambda a, b: (a + b) / 2,       # what the pair is "about" overall
    "absdiff": lambda a, b: np.abs(a - b),     # where the two sentences differ
    "product": lambda a, b: a * b,             # what the two sentences share
}
STANDARDISE = True       # z-score each dimension with train statistics (recommended)
PROBE_C = 0.05           # logistic regression regularisation (smaller = stronger)
N_RANDOM = 200           # random directions for the null band
SEED = 0
RESULTS_DIR = Path("results") / CACHE_DIR.parent.name / CACHE_DIR.name   # results/<model>/<contrast>
# -----------------------------------------------------------------------------


def auc(scores, y):
    """AUC via ranks (Mann-Whitney). y is 0/1; higher score should mean y == 1."""
    r = pd.Series(scores).rank().to_numpy()
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def oriented_auc(train_scores, y_train, val_scores, y_val):
    """AUC for a single number per pair; direction (higher/lower = analogy) is chosen on train."""
    sign = 1.0 if auc(train_scores, y_train) >= 0.5 else -1.0
    return auc(sign * val_scores, y_val)


def word_overlap(s1, s2):
    tok = lambda s: Counter(re.findall(r"[a-z']+", s.lower()))
    out = []
    for a, b in zip(s1, s2):
        ca, cb = tok(a), tok(b)
        num = sum(ca[w] * cb[w] for w in ca)
        den = (sum(v * v for v in ca.values()) ** 0.5) * (sum(v * v for v in cb.values()) ** 0.5)
        out.append(num / den if den else 0.0)
    return np.array(out)


def main(use_probe=True):
    rng = np.random.default_rng(SEED)
    A_tr, M_tr = load_cached("train")
    A_va, M_va = load_cached("val")
    y_tr = M_tr["label"].to_numpy(); y_va = M_va["label"].to_numpy()   # 1 = analogy
    n_layers = A_tr["last_s1"].shape[1]
    print(f"Cache: {CACHE_DIR}  | train {len(y_tr)}  val {len(y_va)}  | layers {n_layers}")
    print(f"Groups: {M_tr.groupby('label').group.first().to_dict()}  (label 1 = analogy)")

    if use_probe:
        from sklearn.linear_model import LogisticRegression

    rows = []
    for pooling in POOLINGS:
        for fname, fn in FEATURES.items():
            for layer in range(n_layers):
                Xtr = fn(A_tr[f"{pooling}_s1"][:, layer], A_tr[f"{pooling}_s2"][:, layer])
                Xva = fn(A_va[f"{pooling}_s1"][:, layer], A_va[f"{pooling}_s2"][:, layer])
                if STANDARDISE:
                    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd

                # CAA direction: mean difference between the two groups (train)
                d = Xtr[y_tr == 1].mean(0) - Xtr[y_tr == 0].mean(0)
                d_unit = d / (np.linalg.norm(d) + 1e-8)
                caa = auc(Xva @ d_unit, y_va)

                # null band: random unit directions (direction-free: max(a, 1-a), 95th percentile)
                R = rng.normal(size=(Xva.shape[1], N_RANDOM)); R /= np.linalg.norm(R, axis=0)
                null = np.array([auc(Xva @ R[:, j], y_va) for j in range(N_RANDOM)])
                null_hi = np.percentile(np.maximum(null, 1 - null), 95)

                probe = np.nan
                if use_probe:
                    clf = LogisticRegression(C=PROBE_C, max_iter=300)
                    clf.fit(Xtr, y_tr)
                    probe = auc(clf.decision_function(Xva), y_va)

                rows.append(dict(pooling=pooling, feature=fname, layer=layer, caa_auc=caa,
                                 probe_auc=probe, null95=null_hi))
            print(f"  done {pooling}/{fname}")

    res = pd.DataFrame(rows)

    # one-number-per-pair detector: cosine similarity of s1 and s2
    cos_rows = []
    for pooling in POOLINGS:
        for layer in range(n_layers):
            ctr = row_cosine(A_tr[f"{pooling}_s1"][:, layer], A_tr[f"{pooling}_s2"][:, layer])
            cva = row_cosine(A_va[f"{pooling}_s1"][:, layer], A_va[f"{pooling}_s2"][:, layer])
            cos_rows.append(dict(pooling=pooling, layer=layer, cosine_auc=oriented_auc(ctr, y_tr, cva, y_va)))
    cos_df = pd.DataFrame(cos_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / "layer_scan.csv", index=False)
    cos_df.to_csv(RESULTS_DIR / "cosine_scan.csv", index=False)

    # ---- baselines that use no model activations ----
    ov_tr = word_overlap(M_tr.s1, M_tr.s2); ov_va = word_overlap(M_va.s1, M_va.s2)
    overlap_auc = oriented_auc(ov_tr, y_tr, ov_va, y_va)
    entity_auc = oriented_auc(M_tr.entity.to_numpy(), y_tr, M_va.entity.to_numpy(), y_va)
    len_auc = np.nan
    if use_probe:
        L = lambda M: np.column_stack([M.len_s1, M.len_s2, (M.len_s1 - M.len_s2).abs()])
        len_auc = auc(LogisticRegression(max_iter=300).fit(L(M_tr), y_tr).decision_function(L(M_va)), y_va)

    # ---- summary ----
    print("\n================ SUMMARY (validation AUC; 0.5 = chance) ================")
    for (pooling, fname), g in res.groupby(["pooling", "feature"]):
        b = g.loc[g.caa_auc.idxmax()]
        line = (f"{pooling:4s}/{fname:8s} CAA best layer {int(b.layer):2d}: {b.caa_auc:.3f}"
                f" (random 95th pct {b.null95:.3f}; layer 0: {g[g.layer==0].caa_auc.iloc[0]:.3f})")
        if use_probe:
            p = g.loc[g.probe_auc.idxmax()]
            line += f" | probe best layer {int(p.layer):2d}: {p.probe_auc:.3f} (layer 0: {g[g.layer==0].probe_auc.iloc[0]:.3f})"
        print(line)
    print("\nOne-number detector (cosine between the two sentences):")
    for pooling, g in cos_df.groupby("pooling"):
        b = g.loc[g.cosine_auc.idxmax()]
        print(f"  {pooling:4s} best layer {int(b.layer):2d}: {b.cosine_auc:.3f}  (layer 0: {g[g.layer==0].cosine_auc.iloc[0]:.3f})")
    print("\nBaselines (no model activations):")
    print(f"  word-overlap cosine:                 {overlap_auc:.3f}")
    print(f"  length only (logistic regression):   {len_auc:.3f}")
    print(f"  human entity score alone:            {entity_auc:.3f}   <- should be near 0.5 if entity is matched")
    print("\nHow to read it: a layer is interesting only if its AUC is (1) above the random 95th")
    print("percentile, (2) clearly above layer 0 (otherwise the signal is already in the words),")
    print("(3) above the word-overlap and length baselines. If length alone scores well, the groups")
    print("differ in length and some of the signal is just that.")
    print(f"\nFull tables saved to {RESULTS_DIR}/")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(POOLINGS), figsize=(6 * len(POOLINGS), 4), sharey=True)
        for ax, pooling in zip(np.atleast_1d(axes), POOLINGS):
            for fname in FEATURES:
                g = res[(res.pooling == pooling) & (res.feature == fname)]
                ax.plot(g.layer, g.caa_auc, label=f"CAA {fname}")
                if use_probe:
                    ax.plot(g.layer, g.probe_auc, "--", label=f"probe {fname}")
            c = cos_df[cos_df.pooling == pooling]
            ax.plot(c.layer, c.cosine_auc, ":", color="k", label="cosine(s1,s2)")
            ax.axhline(overlap_auc, color="gray", lw=1, label="word overlap")
            if use_probe:
                ax.axhline(len_auc, color="brown", lw=1, ls="-.", label="length only")
            ax.axhline(0.5, color="red", lw=1)
            ax.set_title(f"{pooling} pooling"); ax.set_xlabel("layer"); ax.set_ylabel("validation AUC")
        np.atleast_1d(axes)[0].legend(fontsize=7)
        fig.suptitle(f"{CACHE_DIR.parent.name} - {CACHE_DIR.name}")
        fig.tight_layout(); fig.savefig(RESULTS_DIR / "layer_scan.png", dpi=150)
        print(f"Plot saved to {RESULTS_DIR / 'layer_scan.png'}")
    except ImportError:
        print("matplotlib not installed - skipped the plot (pip install matplotlib)")


if __name__ == "__main__":
    main(use_probe="--no-probe" not in sys.argv)
