"""Scores used by every experiment: AUC, AUC within groups, bootstrap intervals, word overlap."""
import re
from collections import Counter

import numpy as np
import pandas as pd


def auc(scores, y):
    """Area under the ROC curve via ranks (Mann-Whitney U): the chance that a random label-1 item scores higher than a
    random label-0 item. 0.5 = chance, 1.0 = perfect. `y` is 0/1."""
    y = np.asarray(y)
    r = pd.Series(np.asarray(scores)).rank().to_numpy()
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def oriented_auc(train_scores, y_train, val_scores, y_val):
    """AUC of a single number per pair (e.g. word overlap); whether higher or lower means label 1 is decided on train."""
    sign = 1.0 if auc(train_scores, y_train) >= 0.5 else -1.0
    return auc(sign * np.asarray(val_scores), y_val)


def grouped_auc(scores, y, groups, min_per_class=5):
    """AUC computed inside each group, averaged with weight n1 * n0 (groups with fewer than min_per_class of a class skipped).
    With word-overlap bins as groups this is the overlap-adjusted AUC: overlap cannot explain a score that holds inside each bin.
    With the model's answer as groups it is 'AUC of the label with the answer held fixed'."""
    scores, y, groups = np.asarray(scores), np.asarray(y), np.asarray(groups)
    num = den = 0.0
    for g in np.unique(groups):
        m = groups == g
        n1, n0 = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        if n1 < min_per_class or n0 < min_per_class:
            continue
        num += n1 * n0 * auc(scores[m], y[m]); den += n1 * n0
    return num / den if den else np.nan


def bootstrap_ci(stat, n, rng, n_boot=300, level=95):
    """Percentile interval of stat(idx) over n_boot resamples of n items (idx = resampled indices)."""
    vals = [stat(rng.integers(0, n, n)) for _ in range(n_boot)]
    a = (100 - level) / 2
    return tuple(np.nanpercentile(vals, [a, 100 - a]))


def word_overlap(s1, s2):
    """Cosine similarity of the word-count vectors of each pair (0 = no shared words, 1 = same words)."""
    tok = lambda s: Counter(re.findall(r"[a-z']+", s.lower()))
    out = []
    for a, b in zip(s1, s2):
        ca, cb = tok(a), tok(b)
        num = sum(ca[w] * cb[w] for w in ca)
        den = (sum(v * v for v in ca.values()) ** 0.5) * (sum(v * v for v in cb.values()) ** 0.5)
        out.append(num / den if den else 0.0)
    return np.array(out)


def overlap_bins(train_overlap, val_overlap, n_bins=4):
    """Assign each validation pair to a word-overlap quartile; the edges come from the TRAIN pairs."""
    edges = np.unique(np.quantile(train_overlap, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.digitize(val_overlap, edges)
