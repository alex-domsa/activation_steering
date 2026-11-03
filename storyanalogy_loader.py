"""
StoryAnalogy loader: turns the raw CSV into labelled, balanced train/val/test tables.

Each pair in the CSV has two human scores (0-3):
    entity   = how similar the THINGS in the two sentences are
    relation = how similar the RELATIONSHIPS / events are
From those we define groups:
    analogy    low entity,  high relation   (same structure, different things)
    similar    high entity, high relation   (same structure, similar things)
    unrelated  low entity,  low relation    (neither)
A CONTRAST picks which two groups we compare (the vector = difference between them):
    analogy_vs_unrelated  entity is matched, only RELATION differs  <- tests relational structure
    analogy_vs_similar    relation is matched, only ENTITY differs  <- mostly topic overlap


"""

import sys
import numpy as np
import pandas as pd

# ---- Settings: everything I might want to change is here ----------------
CONTRAST = "analogy_vs_unrelated"        # or "analogy_vs_similar"
CONTRASTS = {                            # (positive group = label 1, negative group = label 0)
    "analogy_vs_unrelated": ("analogy", "unrelated"),
    "analogy_vs_similar":   ("analogy", "similar"),
}
RELATION_MIN = 2.0         # analogy and similar both need a strong relational match
ENTITY_MAX_ANALOGY = 1.0   # analogy = low entity overlap
ENTITY_MIN_SIMILAR = 2.0   # similar = high entity overlap
ENTITY_MAX_UNRELATED = 1.0 # unrelated uses the SAME entity ceiling as analogy, so entity is matched
RELATION_MAX_UNRELATED = 1.0   # unrelated = weak relational match
DOMAINS = ["propara", "wordanalogy", "conceptnet"]  # rocstories dropped: too few analogy pairs
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
LENGTH_BIN_WIDTH = 4       # match groups on total words (s1+s2) in buckets of this many words; None = off
SEED = 0                   # fixed seed -> the same sample and split every run
# ----------------------------------------------------------------------------


def assign_group(entity, relation):
    """Name one pair from its two scores."""
    if relation >= RELATION_MIN and entity <= ENTITY_MAX_ANALOGY:
        return "analogy"
    if relation >= RELATION_MIN and entity >= ENTITY_MIN_SIMILAR:
        return "similar"
    if relation <= RELATION_MAX_UNRELATED and entity <= ENTITY_MAX_UNRELATED:
        return "unrelated"
    return "other"


def load_storyanalogy(path, contrast=CONTRAST):
    positive, negative = CONTRASTS[contrast]
    labels = {negative: 0, positive: 1}          # analogy is always label 1
    df = pd.read_csv(path)

    # 1. Name every row, then keep only the two groups of this contrast.
    df["group"] = [assign_group(e, r) for e, r in zip(df["entity"], df["relation"])]
    df = df[df["group"].isin(labels)]

    # 2. Keep only the chosen domains.
    df = df[df["domain"].isin(DOMAINS)].copy()

    # 3. Balance: inside each domain AND each length bucket, keep the same number of rows per group.
    #    (so the domain mix and the length distribution are the same in both groups, and neither
    #    domain nor sentence length can give the answer away)
    df["len_s1"] = df["s1"].str.split().str.len()
    df["len_s2"] = df["s2"].str.split().str.len()
    keys = ["domain"]
    if LENGTH_BIN_WIDTH:
        df["len_bin"] = (df["len_s1"] + df["len_s2"]) // LENGTH_BIN_WIDTH   # e.g. 4 -> buckets 0-3, 4-7, ... words
        keys.append("len_bin")
    rng = np.random.default_rng(SEED)
    balanced = []
    for _, d in df.groupby(keys):
        n = int(d["group"].value_counts().reindex(list(labels)).fillna(0).min())   # size of the smaller group
        if n == 0:
            continue                                   # bucket has only one group: nothing to match against
        for group, g in d.groupby("group"):
            balanced.append(g.sample(n=n, random_state=int(rng.integers(1_000_000))))
    df = pd.concat(balanced)

    # 4. Split. Rows sharing the same s1 stay together, so no story leaks across splits.
    #    Done per (domain, group) so every split keeps the same domain/group mix.
    df["split"] = ""
    for (domain, group), d in df.groupby(["domain", "group"]):
        stories = np.array(d["s1"].unique(), dtype=object)   # plain numpy array so shuffle is safe
        rng.shuffle(stories)
        n_train = int(len(stories) * SPLIT_FRACTIONS["train"])
        n_val = int(len(stories) * SPLIT_FRACTIONS["val"])
        story_split = {}
        for i, s in enumerate(stories):
            story_split[s] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
        df.loc[d.index, "split"] = d["s1"].map(story_split)

    #    Leak guard on BOTH sentences: a sentence seen in train (as s1 OR s2) may not appear
    #    anywhere in val/test, and a sentence seen in val may not appear in test.
    train_sents = set(df.loc[df["split"] == "train", "s1"]) | set(df.loc[df["split"] == "train", "s2"])
    in_train = df["s1"].isin(train_sents) | df["s2"].isin(train_sents)
    df = df[~((df["split"] != "train") & in_train)]
    val_sents = set(df.loc[df["split"] == "val", "s1"]) | set(df.loc[df["split"] == "val", "s2"])
    in_val = df["s1"].isin(val_sents) | df["s2"].isin(val_sents)
    df = df[~((df["split"] == "test") & in_val)]

    # 5. Add handy columns for later: numeric label and word counts (for a length baseline).
    df["label"] = df["group"].map(labels)
    df = df.drop(columns=["len_bin"], errors="ignore").reset_index(drop=True)

    return {name: df[df["split"] == name].reset_index(drop=True) for name in SPLIT_FRACTIONS}


if __name__ == "__main__":
    contrast = sys.argv[2] if len(sys.argv) > 2 else CONTRAST
    splits = load_storyanalogy(sys.argv[1], contrast)
    all_rows = pd.concat(splits.values(), ignore_index=True)
    print("contrast:", contrast)
    print("rows per split:", {k: len(v) for k, v in splits.items()})
    print("\ngroup x split:")
    print(pd.crosstab(all_rows["group"], all_rows["split"]))
    print("\ndomain x group (all splits):")
    print(pd.crosstab(all_rows["domain"], all_rows["group"]))
    print("\ndomain x split:")
    print(pd.crosstab(all_rows["domain"], all_rows["split"]))
    print("\nmean words per side by group:")
    print(all_rows.groupby("group")[["len_s1", "len_s2"]].mean().round(1))
    print("\nmean human scores by group (entity should match when contrasting with 'unrelated'):")
    print(all_rows.groupby("group")[["entity", "relation"]].mean().round(2))
    # leakage checks: any sentence (as s1 or s2) shared between splits
    sents = {k: set(v["s1"]) | set(v["s2"]) for k, v in splits.items()}
    print("\nsentences shared between splits (train&test, train&val, val&test):",
          len(sents["train"] & sents["test"]), len(sents["train"] & sents["val"]),
          len(sents["val"] & sents["test"]))
