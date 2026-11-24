"""
First look at the cached activations (TRAIN split only)


Part 1  sanity checks (shapes, NaNs, size of activations per layer)
Part 2  per layer: how similar are s1 and s2 inside a pair (cosine similarity)?
        - average for analogy vs same-domain pairs
        - how it tracks the relation and entity scores (regression, controlling for domain)
"""

import numpy as np
import pandas as pd

from cache_activations import load_cached, CACHE_DIR

SPLIT = "train"


def row_cosine(a, b):
    """Cosine similarity of matching rows. a, b: (n, d) -> (n,)"""
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    return num / den


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def regression_betas(y, relation, entity, domain):
    """Standardised regression: y ~ relation + entity + domain dummies.
    Returns (beta_relation, beta_entity, R^2). Betas are in 'standard deviations of y
    per standard deviation of the score', holding the other score and the domain fixed."""
    dummies = pd.get_dummies(domain, drop_first=True).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), zscore(relation), zscore(entity), dummies])
    yz = zscore(y)
    coef, *_ = np.linalg.lstsq(X, yz, rcond=None)
    resid = yz - X @ coef
    r2 = 1 - (resid ** 2).sum() / ((yz - yz.mean()) ** 2).sum()
    return coef[1], coef[2], r2


def main():
    acts, meta = load_cached(SPLIT)
    n = len(meta)
    n_layers = acts["last_s1"].shape[1]                        # layers + 1 (embeddings)
    layers_to_show = sorted(set(range(0, n_layers, 2)) | {n_layers - 1})   # every 2nd layer + last
    print(f"Reading cache from: {CACHE_DIR}")

    # ---- Part 1: sanity checks ----
    print(f"== Sanity check ({SPLIT}) ==")
    for k, v in acts.items():
        print(f"{k:8s} shape {v.shape}  NaN: {np.isnan(v).any()}  Inf: {np.isinf(v).any()}")
    norms = np.linalg.norm(acts["last_s1"], axis=2).mean(axis=0)     # (25,)
    print("average activation length per layer (last-token):")
    show = sorted({0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1})
    print("  " + "  ".join(f"L{l}:{norms[l]:.0f}" for l in show))
    print(f"groups: {meta.group.value_counts().to_dict()}")

    # ---- Part 2: per-layer similarity of s1 and s2 ----
    is_analogy = (meta.group == "analogy").to_numpy()
    for pooling in ("last", "mean"):
        s1, s2 = acts[f"{pooling}_s1"], acts[f"{pooling}_s2"]
        # Mean-centre: subtract the average activation per layer, so the large shared
        # component does not make every similarity look high.
        centre = np.concatenate([s1, s2], axis=0).mean(axis=0, keepdims=True)   # (1,25,896)
        for centred in (False, True):
            a = s1 - centre if centred else s1
            b = s2 - centre if centred else s2
            print(f"\n== {pooling} pooling, {'mean-centred' if centred else 'raw'} cosine ==")
            print("layer | cos analogy | cos same-dom | diff   | beta_relation | beta_entity |  R2")
            for l in layers_to_show:
                cos = row_cosine(a[:, l, :], b[:, l, :])
                ca, cl = cos[is_analogy].mean(), cos[~is_analogy].mean()
                br, be, r2 = regression_betas(cos, meta.relation.to_numpy(),
                                              meta.entity.to_numpy(), meta.domain.to_numpy())
                print(f"{l:5d} | {ca:11.3f} | {cl:12.3f} | {ca - cl:+.3f} | {br:+13.3f} | {be:+11.3f} | {r2:.3f}")

    print("\nHow to read it:")
    print("- diff: analogy minus same-domain similarity. Negative = analogy pairs are LESS similar inside.")
    print("- beta_relation / beta_entity: how much similarity moves with each score when the other")
    print("  score and the domain are held fixed. These are the numbers that matter most.")
    print("- Compare 'raw' with 'mean-centred': if a pattern only shows up in one, be careful.")


if __name__ == "__main__":
    main()
