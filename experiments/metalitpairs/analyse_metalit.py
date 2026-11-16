"""
Analyse the cached activations: one metaphor - literal difference per pair, the CAA vector, and where (which layer, and
which word) the metaphor signal is strongest.

For every pair i and every layer L:
    d_i(L) = activation(metaphor_i) - activation(literal_i)        <- its OWN literal, never another pair's
The CAA vector is the mean of these differences over the TRAIN pairs:
    v(L) = mean_i d_i(L)                                              (metaphor minus literal)
Scores:
    pair score   cos(d_i, v)  - does this pair's difference point the same way as the vector? (1 = exactly, 0 = unrelated,
                                negative = opposite). Train pairs are scored against a vector built WITHOUT their own group.
    coherence    the mean pair score over pairs (Braun et al. 2025: high coherence predicts that steering works)
    paired acc.  fraction of pairs where the metaphor projects higher on v than its own literal (d_i . v > 0); 0.5 = chance
    effect size  mean(d_i . v_hat) / sd(d_i . v_hat): how cleanly the pairs separate along v (scale-free across layers)
Controls (validation pairs, so nothing here was used to build the vector):
    sign-flip null    vectors built after randomly swapping metaphor/literal inside each train pair: the real coherence must
                      be well above theirs (95th percentile), otherwise the method itself creates the result. (Coherence,
                      not paired accuracy: paired accuracy is easy - any shift shared by the pairs gives it, even to a
                      nearly random vector.)
    length            correlation of the projection with the token-length difference, and paired accuracy on pairs whose
                      two sentences have the same number of tokens (length cannot help there)
    leave-one-source-out   vector built from two sources (dictionary, BNC, poetry), tested on every pair of the third
Two activation summaries are analysed: "mean" (mean over the sentence's tokens) and "last" (final token).

Layer choice: by validation coherence, within the middle of the network (25%-85% of depth: the first layers are mostly
word identity, steering at the last ones rarely changes what is written). The test pairs are not touched: they are kept
for the generation test (heldout_pairs.csv).

Run:
    python analyse_metalit.py manual_e.tsv                       (numbers + figures, no model needed, seconds)
    python analyse_metalit.py manual_e.tsv --tokens              (also: which WORD carries the signal, loads the model)
    options: --model=...  (same as for caching)   --layer=N  (use your own layer instead of the recommended one)
Outputs in results/<model>/: report.md, layer_table_<feature>.csv, per_pair_<feature>.csv, heatmap_<feature>.png,
tokens.md (with --tokens); vectors/<model>/met_minus_lit_<feature>.npy (built from train pairs only) and heldout_pairs.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metalit_loader import load_pairs


def _arg(name, default):
    a = sys.argv[1:]
    for i, x in enumerate(a):
        if x.startswith(name + "="):
            return x.split("=", 1)[1]
        if x == name and i + 1 < len(a):
            return a[i + 1]
    return default


MODEL = _arg("--model", "Qwen/Qwen2.5-1.5B-Instruct")
LAYER = _arg("--layer", None)
FEATURES = ("mean", "last")
N_NULL = 50                 # sign-flip null vectors
MIDDLE = (0.25, 0.85)       # layers considered for the recommendation (fraction of depth)
SEED = 0
TAG = MODEL.rstrip("/").split("/")[-1]


# ---------------------------------------------------------------------- loading (with the pairing check)
def load(tsv, cache=None):
    cache = Path(cache or Path("cache") / TAG / "acts.npz")
    z = np.load(cache)
    df = load_pairs(tsv).set_index("pair_id").loc[z["pair_id"]].reset_index()     # same order as the cache rows
    assert (df.metaphor.values == z["metaphor"]).all() and (df.literal.values == z["literal"]).all(), \
        "cache rows and loader pairs do not match: rebuild the cache with cache_metalit.py --force"
    assert (df.split.values == z["split"]).all(), "the split changed since caching: rebuild the cache"
    print(f"Loaded {cache}: {len(df)} pairs, every cached row matches its pair id (metaphor and literal).")
    return z, df


# ---------------------------------------------------------------------- scores
def unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def cos_rows(D, V):
    """cos(D[i, L], V[i, L]) or cos(D[i, L], V[L]): (n, layers)."""
    return (unit(D) * unit(V)).sum(-1)


def caa(D, train):
    return D[train].mean(0)


def leave_group_out(D, train, groups):
    """For each train pair: the train mean WITHOUT the pair's group (no pair is scored against itself or its duplicates)."""
    tr_idx = np.where(train)[0]
    total, n_tr = D[tr_idx].sum(0), len(tr_idx)
    V = np.repeat(caa(D, train)[None], len(D), axis=0)
    for g in np.unique(groups[tr_idx]):
        members = tr_idx[groups[tr_idx] == g]
        V[members] = (total - D[members].sum(0)) / (n_tr - len(members))
    return V


def pair_scores(D, train, groups):
    return cos_rows(D, leave_group_out(D, train, groups))


def paired_acc(D, v):
    return ((D * v).sum(-1) > 0).mean(0)


def effect_size(D, v):
    p = (D * unit(v)).sum(-1)
    return p.mean(0) / (p.std(0) + 1e-12)


def sign_flip_null(D, train, val, rng, n=N_NULL):
    """(n, layers) validation coherence of vectors built from randomly side-swapped train pairs (absolute value)."""
    tr = np.where(train)[0]
    out = []
    for _ in range(n):
        s = rng.choice([-1.0, 1.0], size=len(tr))[:, None, None]
        out.append(np.abs(cos_rows(D[val], (s * D[tr]).mean(0)).mean(0)))
    return np.array(out)


def layer_table(D, df, ntok_diff, rng):
    tr, va = (df.split == "train").values, (df.split == "val").values
    v = caa(D, tr)
    scores = pair_scores(D, tr, df.group.values)
    null = sign_flip_null(D, tr, va, rng)
    proj = (D[va] * unit(v)).sum(-1)                                   # (n_val, layers)
    same_len = va & (ntok_diff == 0)
    t = pd.DataFrame({
        "layer": np.arange(D.shape[1]),
        "val_coherence": scores[va].mean(0),
        "val_paired_acc": paired_acc(D[va], v),
        "val_effect_size": effect_size(D[va], v),
        "train_coherence": scores[tr].mean(0),
        "null_coherence_95pct": np.percentile(null, 95, axis=0),
        "same_len_paired_acc": paired_acc(D[same_len], v) if same_len.any() else np.nan,
        "corr_proj_vs_len_diff": [np.corrcoef(proj[:, L], ntok_diff[va])[0, 1] if proj[:, L].std() > 0 else np.nan
                                  for L in range(D.shape[1])],
    })
    return t, scores, v


def recommend(t, n_layers):
    lo, hi = int(MIDDLE[0] * n_layers), int(np.ceil(MIDDLE[1] * n_layers))
    mid = t[(t.layer >= lo) & (t.layer <= hi)]
    return int(mid.loc[mid.val_coherence.idxmax(), "layer"])


def leave_one_source_out(D, df, L):
    rows = []
    for src in sorted(df.source.unique()):
        build = ((df.split == "train") & (df.source != src)).values
        test = (df.source == src).values
        v = caa(D, build)[L]
        rows.append({"held-out source": src, "pairs": int(test.sum()),
                     "paired_acc": float(((D[test, L] @ v) > 0).mean()),
                     "coherence": float(cos_rows(D[test, L], v).mean())})
    return pd.DataFrame(rows)


def md_table(t):
    """DataFrame -> markdown table (no extra package needed)."""
    cols = list(t.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] +
                      ["| " + " | ".join(str(x) for x in r) + " |" for r in t.itertuples(index=False)])


# ---------------------------------------------------------------------- figures
def heatmap(scores, df, path, title, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = df.sort_values(["split", "source", "pair_id"], key=lambda c: c.map({"train": 0, "val": 1, "test": 2})
                           if c.name == "split" else c).index.values
    S = scores[order][:, 1:]                                           # layer 0 (embeddings) left out
    fig, ax = plt.subplots(figsize=(10, max(6, 0.028 * len(order))))
    im = ax.imshow(S, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
                   extent=(0.5, S.shape[1] + 0.5, len(order), 0))
    splits = df.split.values[order]
    for i in range(1, len(order)):
        if splits[i] != splits[i - 1]:
            ax.axhline(i, color="black", lw=1)
    ax.axvline(chosen, color="black", lw=0.8, ls="--")
    ax.set_xlabel("layer"); ax.set_ylabel("pairs (train | val | test, by source, then pair id)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="cos(pair difference, CAA vector)", shrink=0.5)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------- token view (needs the model)
def token_view(df, v_mean, L, out_md, n_print=8):
    """Projection of every token's state at layer L on the (mean-feature) vector: does the peak sit on the metaphor word?"""
    import re
    from cache_metalit import Reader
    r = Reader(MODEL)
    vh = unit(v_mean[L])
    val = df[df.split == "val"]
    lines, hits, chance, hits_lit, chance_lit = [], [], [], [], []

    def spans(text, keep):
        return [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z']+", text) if m.group().lower() in keep]

    for k, row in enumerate(val.itertuples()):
        out = {}
        for side, text, other in (("metaphor", row.metaphor, row.literal), ("literal", row.literal, row.metaphor)):
            hs, offs = r.states(text)
            own = set(re.findall(r"[a-z']+", text.lower())) - set(re.findall(r"[a-z']+", other.lower()))
            sp = spans(text, own)
            proj = hs[L] @ vh
            toks = [text[a:b] for a, b in offs]
            changed = np.array([any(a < e and s < b for s, e in sp) for a, b in offs])
            body = np.arange(1, len(toks)) if len(toks) > 1 else np.arange(len(toks))   # first token left out
            out[side] = (toks, proj, changed, body)
        toks, proj, changed, body = out["metaphor"]
        if changed[body].any():
            hits.append(changed[body][proj[body].argmax()]); chance.append(changed[body].mean())
        toks_l, proj_l, changed_l, body_l = out["literal"]
        if changed_l[body_l].any():
            hits_lit.append(changed_l[body_l][proj_l[body_l].argmin()]); chance_lit.append(changed_l[body_l].mean())
        if True:                                                       # every pair goes to tokens.md
            def show(tk, pr, bd, top):
                return " ".join(f"**{t.strip()}**({pr[i]:+.1f})" if i == top else f"{t.strip()}({pr[i]:+.1f})"
                                for i, t in enumerate(tk) if i in set(bd))
            lines += [f"\n**[{row.pair_id}]** changed words: *{row.changed_words}*",
                      f"- metaphor: {show(toks, proj, body, body[proj[body].argmax()])}",
                      f"- literal : {show(toks_l, proj_l, body_l, body_l[proj_l[body_l].argmin()])}"]
            if k < n_print:
                print(lines[-3]); print(lines[-2]); print(lines[-1])
    summary = (f"Layer {L}, {len(val)} validation pairs. Highest-scoring token of the METAPHOR sentence falls on a word the "
               f"literal does not use in {np.mean(hits):.0%} of pairs (chance {np.mean(chance):.0%}).\n"
               f"Lowest-scoring token of the LITERAL sentence falls on its replacement word in {np.mean(hits_lit):.0%} "
               f"of pairs (chance {np.mean(chance_lit):.0%}).")
    print("\n" + summary)
    Path(out_md).write_text("# Token view: projection of each token on the metaphor - literal vector\n\n"
                            "Bold = highest token (metaphor) / lowest token (literal). First token left out.\n\n"
                            + summary + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out_md}")


# ---------------------------------------------------------------------- main
def main(tsv):
    z, df = load(tsv)
    res, vec = Path("results") / TAG, Path("vectors") / TAG
    res.mkdir(parents=True, exist_ok=True); vec.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ntok_diff = z["met_ntok"] - z["lit_ntok"]
    n_layers = z["met_last"].shape[1] - 1
    pd.set_option("display.width", 200)
    md = [f"# Metaphor - literal CAA ({MODEL})\n",
          f"{len(df)} pairs: " + ", ".join(f"{k} {v}" for k, v in df.split.value_counts().items()) +
          f". Metaphor has more tokens than its literal in {(ntok_diff > 0).mean():.0%} of pairs, fewer in "
          f"{(ntok_diff < 0).mean():.0%}, the same in {(ntok_diff == 0).mean():.0%}.\n"]
    chosen = {}
    vectors = {}
    for f in FEATURES:
        D = z[f"met_{f}"] - z[f"lit_{f}"]                                # row i = pair_id[i]: metaphor minus ITS literal
        t, scores, v = layer_table(D, df, ntok_diff, rng)
        L = int(LAYER) if LAYER else recommend(t, n_layers)
        chosen[f], vectors[f] = L, v
        np.save(vec / f"met_minus_lit_{f}.npy", v.astype(np.float32))
        t.round(3).to_csv(res / f"layer_table_{f}.csv", index=False)

        print(f"\n===== feature: {f} (vector = mean over {int((df.split == 'train').sum())} train pairs) =====")
        print(t.iloc[1:].round(2).to_string(index=False))
        lsso = leave_one_source_out(D, df, L)
        print(f"\nRecommended layer for '{f}': {L}  (highest validation coherence in layers "
              f"{int(MIDDLE[0] * n_layers)}-{int(np.ceil(MIDDLE[1] * n_layers))})")
        if t.loc[L, "val_coherence"] <= t.loc[L, "null_coherence_95pct"]:
            print(f"WARNING: at layer {L} the coherence does not beat the sign-flip null: no reliable direction here.")
        print(f"Leave-one-source-out at layer {L}:\n{lsso.round(2).to_string(index=False)}")

        per = df[["pair_id", "split", "source", "metaphor", "literal", "changed_words"]].copy()
        for l in range(1, n_layers + 1):
            per[f"L{l}"] = scores[:, l].round(3)
        per["best_layer"] = scores[:, 1:].argmax(1) + 1
        per["best_score"] = scores[:, 1:].max(1).round(3)
        per[f"score_at_L{L}"] = scores[:, L].round(3)
        per.to_csv(res / f"per_pair_{f}.csv", index=False)
        heatmap(scores, df, res / f"heatmap_{f}.png",
                f"{TAG}, feature '{f}': each pair's difference vs the CAA vector (dashed = layer {L})", L)

        row = t.loc[L]
        md += [f"\n## Feature: {f}\n",
               f"Recommended layer **{L}**: validation coherence {row.val_coherence:.2f}, paired accuracy "
               f"{row.val_paired_acc:.2f}, sign-flip null coherence (95th pct) {row.null_coherence_95pct:.2f}, "
               f"effect size {row.val_effect_size:.2f}, same-length pairs {row.same_len_paired_acc:.2f}, "
               f"correlation with length difference {row.corr_proj_vs_len_diff:+.2f}.\n",
               "### All layers\n", md_table(t.iloc[1:].round(2)),
               f"\n### Leave-one-source-out (layer {L})\n", md_table(lsso.round(2)),
               f"\n### Where each pair is strongest\n", "Best layer per pair (count of pairs):\n",
               " ".join(f"L{k}:{c}" for k, c in sorted(per.best_layer.value_counts().items())) + "\n"]
        ev = per[per.split != "train"].sort_values(f"score_at_L{L}")
        for title, part in (("Most aligned validation/test pairs", ev.tail(8)[::-1]), ("Least aligned", ev.head(8))):
            md += [f"\n**{title} at layer {L}:**\n"] + [
                f"- [{r.pair_id}] {r[f'score_at_L{L}']:+.2f} (best L{r.best_layer}: {r.best_score:+.2f}) "
                f"*{r.metaphor}* -> *{r.literal}*" for _, r in part.iterrows()]
        md.append(f"\n![heatmap](heatmap_{f}.png)\n")

    df[df.split == "test"][["pair_id", "source", "metaphor", "literal"]].to_csv(vec / "heldout_pairs.csv", index=False)
    json.dump({"model": MODEL, "vector": "mean(metaphor - literal) over train pairs, shape (layers + 1, d), index = hidden state",
               "recommended_layer": chosen, "use": "add -m * v[L] to push metaphor -> literal, +m * v[L] for literal -> metaphor",
               "train_pair_ids": df.pair_id[df.split == "train"].tolist()},
              open(vec / "vector_info.json", "w"), indent=1)
    (res / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nSaved {res}/report.md, per-pair CSVs and heatmaps; vectors in {vec}/ "
          f"(train pairs only; the {int((df.split == 'test').sum())} test pairs are in heldout_pairs.csv, never used)")
    if "--tokens" in sys.argv:
        print(f"\n===== token view ('mean' vector, layer {chosen['mean']}) =====")
        token_view(df, vectors["mean"], chosen["mean"], res / "tokens.md")


if __name__ == "__main__":
    files = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and sys.argv[i] not in ("--model", "--layer")]
    main(files[0] if files else "manual_e.tsv")
