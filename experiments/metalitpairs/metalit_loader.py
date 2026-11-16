"""
Metaphor / literal pairs loader (manual_e.tsv).

Every line of the file is ONE pair:  metaphorical sentence <TAB> its literal paraphrase.
The line number (1, 2, 3, ...) is the pair id and never changes. A row is never split up: shuffling and splitting
move whole rows, and every array made from this table keeps the metaphor and the literal of a pair on the same row.

What the loader does (and why):
  1. cleans the text the SAME way on both sides, so the only difference left is metaphor vs literal:
       - the poetry rows mark the metaphorical word with double spaces ("And the Spirit ,  stooping  earthward ,"),
         the literal side has no marks -> the marks are remembered (target_word) and then removed
       - many literal sentences have spaced punctuation ("tribe ." vs "tribe.") -> punctuation spacing is normalised
       - two rows (202, 231) carry leftover CSV quote marks -> removed
       - trailing , ; : . are dropped on both sides (verse lines end in ","), ? and ! are kept
  2. groups rows that share a sentence: rows 1-112 and 253-358 are mostly the SAME metaphors paraphrased twice
     (two annotators), and a few poetry lines repeat. Rows of one group always go to the same split, otherwise the
     model would be tested on a metaphor it was built from.
  3. labels the source of each row (from the line ranges): dictionary examples, BNC (news / fiction / speech), poetry.
  4. splits whole groups 70 / 15 / 15 into train / val / test, separately inside each source (fixed seed).

Usage:
    from metalit_loader import load_pairs
    df = load_pairs("manual_e.tsv")       # one row per pair: pair_id, metaphor, literal, source, group, split, ...

First look (counts, checks, pairs side by side; also writes first_look.md with every pair):
    python metalit_loader.py manual_e.tsv [n_to_print]
"""
import re
import sys

import numpy as np
import pandas as pd

SEED = 0
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
# Sources, read off the file (line ranges, inclusive). Rows 1-112 and 253-358 paraphrase mostly the same sentences.
SOURCES = [(1, 112, "dictionary"), (113, 149, "bnc"), (150, 252, "poetry"), (253, 358, "dictionary"), (359, 10**9, "bnc")]


def source_of(pair_id):
    return next(name for lo, hi, name in SOURCES if lo <= pair_id <= hi)


def strip_quotes(s):
    """CSV leftovers such as  "Let your heart  speak  , Minnehaha ! \"\"\"  ->  Let your heart  speak  , Minnehaha !"""
    s = s.strip()
    if s.startswith('"') and s.count('"') >= 2:
        s = s[1:]
    return s.replace('"""', "").replace('""', '"').rstrip('"').strip()


def marked_word(raw):
    """The word the poetry rows mark with double spaces, or None."""
    raw = raw.strip()
    for pattern in (r" {2,}([^ ]+) {2,}", r"^([^ ]+) {2,}", r" {2,}([^ ]+)$"):   # middle, first word, last word
        m = re.search(pattern, raw)
        if m:
            return m.group(1)
    return None


def clean(s):
    s = re.sub(r"\s+", " ", strip_quotes(s)).strip()          # removes the double-space marks too
    s = re.sub(r" ([,.!?;:])", r"\1", s)                      # "tribe ." -> "tribe."
    s = re.sub(r" ('s|'re|'ve|'d|'ll|'m|n't)\b", r"\1", s)    # "Wolmar 's" -> "Wolmar's", "did n't" -> "didn't"
    s = s.replace("‘ ", "‘").replace(" ’", "’")
    if s.startswith("‘") and "’" not in s:                     # unmatched opening quote (row 113)
        s = s[1:]
    s = re.sub(r"[\s,;:.]+$", "", s)                           # same ending on both sides (? and ! kept)
    return s


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def norm_key(s):
    return " ".join(words(s))


def _groups(df):
    """Union-find: rows that share a metaphor OR a literal sentence (after normalising) form one group."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    first_seen = {}
    for i, (m, l) in enumerate(zip(df.metaphor, df.literal)):
        for key in ("M:" + norm_key(m), "L:" + norm_key(l)):
            if key in first_seen:
                parent[find(i)] = find(first_seen[key])
            else:
                first_seen[key] = i
    roots = [find(i) for i in range(len(df))]
    ids = {r: n for n, r in enumerate(dict.fromkeys(roots))}
    return [ids[r] for r in roots]


def load_pairs(path="manual_e.tsv"):
    rows = []
    with open(path, encoding="utf-8") as f:
        for pair_id, line in enumerate(f, start=1):           # the line number IS the pair id
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            assert len(parts) == 2, f"line {pair_id} does not have exactly 2 columns: {line!r}"
            m_raw, l_raw = parts
            rows.append({"pair_id": pair_id, "metaphor": clean(m_raw), "literal": clean(l_raw),
                         "target_word": marked_word(strip_quotes(m_raw)), "source": source_of(pair_id),
                         "metaphor_raw": m_raw, "literal_raw": l_raw})
    df = pd.DataFrame(rows)
    df["changed_words"] = [" ".join(w for w in dict.fromkeys(words(m)) if w not in set(words(l)))
                           for m, l in zip(df.metaphor, df.literal)]
    df["group"] = _groups(df)

    # split whole groups, separately inside each source
    rng = np.random.default_rng(SEED)
    split_of_group = {}
    for src in df.source.unique():
        gs = np.array(sorted(df.loc[df.source == src, "group"].unique()))
        rng.shuffle(gs)
        n_tr, n_va = int(len(gs) * SPLIT_FRACTIONS["train"]), int(len(gs) * SPLIT_FRACTIONS["val"])
        for i, g in enumerate(gs):
            split_of_group[g] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    df["split"] = df.group.map(split_of_group)

    # guards: ids unique, nothing empty, each group in one split, no pair identical after cleaning
    assert df.pair_id.is_unique
    assert (df.metaphor.str.len() > 0).all() and (df.literal.str.len() > 0).all()
    assert (df.groupby("group").split.nunique() == 1).all()
    return df


def check_no_leak(df):
    """Number of sentences shared between splits (should be 0 0 0)."""
    s = {k: set(map(norm_key, d.metaphor)) | set(map(norm_key, d.literal)) for k, d in df.groupby("split")}
    return len(s["train"] & s["val"]), len(s["train"] & s["test"]), len(s["val"] & s["test"])


if __name__ == "__main__":
    path = next((a for a in sys.argv[1:] if not a.isdigit()), "manual_e.tsv")
    n_show = int(next((a for a in sys.argv[1:] if a.isdigit()), 15))
    df = load_pairs(path)
    print(f"{len(df)} pairs (ids {df.pair_id.min()}-{df.pair_id.max()}), {df.group.nunique()} groups of rows sharing a sentence")
    print("\npairs per source x split:")
    print(pd.crosstab(df.source, df.split, margins=True))
    print("\nsentences shared between splits (train&val, train&test, val&test):", *check_no_leak(df))
    nm, nl = df.metaphor.str.split().str.len(), df.literal.str.split().str.len()
    print(f"\nwords per sentence: metaphor {nm.mean():.1f}, literal {nl.mean():.1f}; "
          f"literal longer in {(nl > nm).mean():.0%} of pairs, same length in {(nl == nm).mean():.0%}")
    print(f"poetry rows with a marked metaphor word: {df.target_word.notna().sum()}")
    print(f"pairs with no changed word (metaphor word kept, only context rephrased): {(df.changed_words == '').sum()}")

    print("\nCleaning (what changed in the text), a few examples:")
    ex = df[(df.metaphor != df.metaphor_raw.str.strip()) | (df.literal != df.literal_raw.str.strip())]
    for _, r in ex.sample(min(4, len(ex)), random_state=SEED).iterrows():
        print(f"  [{r.pair_id}] {r.metaphor_raw.strip()!r} -> {r.metaphor!r}")
        print(f"  {' ' * (len(str(r.pair_id)) + 2)} {r.literal_raw.strip()!r} -> {r.literal!r}")

    print(f"\n{n_show} random pairs (pair id, source, split):")
    for _, r in df.sample(n_show, random_state=SEED).sort_values("pair_id").iterrows():
        tw = f"   [marked word: {r.target_word}]" if isinstance(r.target_word, str) else ""
        print(f"\n  [{r.pair_id}] {r.source} / {r.split}{tw}")
        print(f"     METAPHOR: {r.metaphor}")
        print(f"     LITERAL : {r.literal}")

    md = ["# Metaphor / literal pairs - first look\n",
          f"{len(df)} pairs. Changed words = words of the metaphor that the literal sentence does not use.\n",
          "| id | source | split | metaphor | literal | changed words |", "|---|---|---|---|---|---|"]
    md += [f"| {r.pair_id} | {r.source} | {r.split} | {r.metaphor} | {r.literal} | {r.changed_words} |"
           for _, r in df.iterrows()]
    open("first_look.md", "w", encoding="utf-8").write("\n".join(md).replace("|  |", "| |") + "\n")
    print("\nSaved first_look.md (every pair in one table; open in VS Code with Cmd+Shift+V)")
