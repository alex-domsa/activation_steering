"""
StoryAnalogy loader: the raw CSV -> labelled, balanced train / validation / test tables.

Every pair has two human ratings (0-3):
    entity   = how similar the THINGS in the two stories are
    relation = how similar the RELATIONSHIPS / events are
Groups (Gentner's structure-mapping: an analogy shares relations, not attributes):
    analogy    relation >= 2, entity <= 1     same relational structure, different things
    similar    relation >= 2, entity >= 2     same structure AND similar things (literal similarity)
    unrelated  relation <= 1, entity <= 1     neither
The contrast used throughout is analogy vs unrelated: entity similarity is matched, only the relation differs.

Check it:  python -m analogy_steering.story_analogy_loader     (prints counts and leak checks)
"""
import numpy as np
import pandas as pd

from analogy_steering.config import CONTRAST, SEED, STORYANALOGY_CSV

CONTRASTS = {                                       # (positive group = label 1, negative group = label 0)
    "analogy_vs_unrelated": ("analogy", "unrelated"),
    "analogy_vs_similar": ("analogy", "similar"),
}
RELATION_MIN = 2.0              # analogy and similar both need a strong relational match
ENTITY_MAX_ANALOGY = 1.0        # analogy: low entity similarity
ENTITY_MIN_SIMILAR = 2.0        # similar: high entity similarity
ENTITY_MAX_UNRELATED = 1.0      # unrelated uses the SAME entity ceiling as analogy, so entity is matched
RELATION_MAX_UNRELATED = 1.0    # unrelated: weak relational match
DOMAINS = ["propara", "wordanalogy", "conceptnet"]   # rocstories dropped: too few analogy pairs
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
LENGTH_BIN_WIDTH = 4            # match the groups on total words (s1 + s2) in buckets of 4 words; None = off


def assign_group(entity, relation):
    """Name one pair from its two ratings."""
    if relation >= RELATION_MIN and entity <= ENTITY_MAX_ANALOGY:
        return "analogy"
    if relation >= RELATION_MIN and entity >= ENTITY_MIN_SIMILAR:
        return "similar"
    if relation <= RELATION_MAX_UNRELATED and entity <= ENTITY_MAX_UNRELATED:
        return "unrelated"
    return "other"


def load_storyanalogy(path=STORYANALOGY_CSV, contrast=CONTRAST):
    """-> {"train": df, "val": df, "test": df}; columns include s1, s2, entity, relation, domain, group, label, len_s1, len_s2."""
    positive, negative = CONTRASTS[contrast]
    labels = {negative: 0, positive: 1}                          # analogy is always label 1
    df = pd.read_csv(path)

    # 1. Name every row and keep only the two groups of this contrast, in the chosen domains.
    df["group"] = [assign_group(e, r) for e, r in zip(df["entity"], df["relation"])]
    df = df[df["group"].isin(labels) & df["domain"].isin(DOMAINS)].copy()

    # 2. Balance: inside each (domain, length bucket), keep the same number of pairs per group,
    #    so neither the domain nor the sentence length can give the label away.
    df["len_s1"] = df["s1"].str.split().str.len()
    df["len_s2"] = df["s2"].str.split().str.len()
    keys = ["domain"]
    if LENGTH_BIN_WIDTH:
        df["len_bin"] = (df["len_s1"] + df["len_s2"]) // LENGTH_BIN_WIDTH
        keys.append("len_bin")
    rng = np.random.default_rng(SEED)
    balanced = []
    for _, d in df.groupby(keys):
        n = int(d["group"].value_counts().reindex(list(labels)).fillna(0).min())    # size of the smaller group
        if n == 0:
            continue
        for _, g in d.groupby("group"):
            balanced.append(g.sample(n=n, random_state=int(rng.integers(1_000_000))))
    df = pd.concat(balanced)

    # 3. Split per (domain, group) by source story, so every split has the same mix and no story crosses splits.
    df["split"] = ""
    for _, d in df.groupby(["domain", "group"]):
        stories = np.array(d["s1"].unique(), dtype=object)
        rng.shuffle(stories)
        n_train = int(len(stories) * SPLIT_FRACTIONS["train"])
        n_val = int(len(stories) * SPLIT_FRACTIONS["val"])
        split_of = {s: "train" if i < n_train else "val" if i < n_train + n_val else "test" for i, s in enumerate(stories)}
        df.loc[d.index, "split"] = d["s1"].map(split_of)

    # 4. Leak guard on BOTH sentences: a sentence seen in train may not appear in val/test, one seen in val not in test.
    train_sents = set(df.loc[df["split"] == "train", "s1"]) | set(df.loc[df["split"] == "train", "s2"])
    df = df[~((df["split"] != "train") & (df["s1"].isin(train_sents) | df["s2"].isin(train_sents)))]
    val_sents = set(df.loc[df["split"] == "val", "s1"]) | set(df.loc[df["split"] == "val", "s2"])
    df = df[~((df["split"] == "test") & (df["s1"].isin(val_sents) | df["s2"].isin(val_sents)))]

    df["label"] = df["group"].map(labels)
    df = df.drop(columns=["len_bin"], errors="ignore").reset_index(drop=True)
    return {name: df[df["split"] == name].reset_index(drop=True) for name in SPLIT_FRACTIONS}


if __name__ == "__main__":
    splits = load_storyanalogy()
    rows = pd.concat(splits.values(), ignore_index=True)
    print(f"contrast: {CONTRAST}")
    print("pairs per split:", {k: len(v) for k, v in splits.items()})
    print("\ngroup x split:\n", pd.crosstab(rows["group"], rows["split"]))
    print("\ndomain x group:\n", pd.crosstab(rows["domain"], rows["group"]))
    print("\nmean words per story by group:\n", rows.groupby("group")[["len_s1", "len_s2"]].mean().round(1))
    print("\nmean human ratings by group (entity should match):\n", rows.groupby("group")[["entity", "relation"]].mean().round(2))
    s = {k: set(v["s1"]) | set(v["s2"]) for k, v in splits.items()}
    print("\nsentences shared between splits (train&test, train&val, val&test):",
          len(s["train"] & s["test"]), len(s["train"] & s["val"]), len(s["val"] & s["test"]))
