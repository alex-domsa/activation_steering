"""
Direction C, properly controlled: does ADDING the metaphor - literal vector make the model paraphrase a literal sentence
metaphorically, compared with no vector and with a random vector of the same size? Judged BLIND by a human.

Step 1 (model, about 5 minutes):
    python steer_c.py manual_e.tsv
  a) likelihood with confidence intervals, test pairs, literal prompt: how much +m*v raises (true metaphor - copy of the
     literal) BEYOND a same-size random vector (paired bootstrap over pairs, 95% CI). m = 4 and 8.
  b) generations for N test pairs (default 20) in 4 conditions: no vector, +4 v, +8 v, random (size 8).
     They are SHUFFLED into a rating sheet without the condition: results/<model>/c_rating/sheet.csv
     The answer key (which output came from which condition) is in key.csv: do not open it before rating.
Step 2 (you): open sheet.csv (Excel / Numbers / VS Code) and fill three columns with 1 or 0 for every row:
     metaphorical   1 = at least one word is used non-literally (e.g. "the markets soared", "I'm your rock", "fiscal
                    surgery"); 0 = everything is meant literally. Similes ("like a ...", "as if ...") count as 0.
     same_meaning   1 = it still says essentially what the source sentence says; 0 = meaning lost or changed
     fluent         1 = normal English; 0 = broken, rambling, repetitive or commentary ("To paraphrase ...")
Step 3:
    python steer_c.py --score
  prints, per condition: % metaphorical, % same meaning, % fluent, and % metaphorical AND same meaning AND fluent
  (the number that matters).
Options: --n=20  --layer=N  --model=...  --quant=4bit
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from steer_metalit import FEATURE, LAYER, PROMPT, TAG, VEC_DIR, _arg

N = int(_arg("--n", 20))
M_LIK = (4, 8)
N_RAND = 3
N_BOOT = 2000
SEED = 1                       # different from steer_metalit's sample, so most rated pairs are fresh
OUT = Path("results") / TAG / "c_rating"
COLS = ("metaphorical", "same_meaning", "fluent")


def bootstrap_ci(x, rng):
    means = [rng.choice(x, size=len(x)).mean() for _ in range(N_BOOT)]
    return np.percentile(means, [2.5, 97.5])


def run(tsv):
    from metalit_loader import load_pairs
    from steer_metalit import Steerer
    rng = np.random.default_rng(SEED)
    df = load_pairs(tsv)
    test = df[df.split == "test"].reset_index(drop=True)
    v = np.load(VEC_DIR / f"met_minus_lit_{FEATURE}.npy")[LAYER].astype(np.float32)
    rand = [x / np.linalg.norm(x) * np.linalg.norm(v) for x in rng.normal(size=(N_RAND, v.size)).astype(np.float32)]
    r = Steerer()
    print(f"Direction C: literal prompt + m * v ('{FEATURE}' vector, layer {LAYER}), {len(test)} test pairs")

    # a) likelihood: (target - copy) gain of +m v over a same-size random vector, per pair, with a bootstrap CI
    def gap(vec):
        return np.array([r.mean_logprob(PROMPT.format(s=lit), met, vec) - r.mean_logprob(PROMPT.format(s=lit), lit, vec)
                         for lit, met in zip(test.literal, test.metaphor)])
    base = gap(None)
    print("\nLikelihood: per-token log-prob of the TRUE METAPHOR minus a COPY of the literal, change from no vector")
    for m in M_LIK:
        d_v = gap(m * v) - base
        d_r = np.mean([gap(m * x) for x in rand], 0) - base
        diff = d_v - d_r
        lo, hi = bootstrap_ci(diff, rng)
        print(f"   m={m}: +m*v {d_v.mean():+.2f} | random {d_r.mean():+.2f} | difference {diff.mean():+.2f} "
              f"(95% CI {lo:+.2f} to {hi:+.2f}) | +m*v beats random in {(diff > 0).mean():.0%} of pairs")

    # b) blind generations
    sub = test.sample(n=min(N, len(test)), random_state=SEED).reset_index(drop=True)
    conds = [("no vector", None), ("+4 v", 4 * v), ("+8 v", 8 * v), ("random, size 8", 8 * rand[0])]
    rows = []
    for k, row in enumerate(sub.itertuples(), 1):
        for name, vec in conds:
            rows.append({"pair_id": row.pair_id, "condition": name, "source_sentence": row.literal,
                         "true_metaphor": row.metaphor, "output": r.generate(PROMPT.format(s=row.literal), vec)})
        print(f"   generated {k}/{len(sub)}")
    gen = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    gen.insert(0, "item", np.arange(1, len(gen) + 1))
    OUT.mkdir(parents=True, exist_ok=True)
    sheet = gen[["item", "source_sentence", "output"]].copy()
    for c in COLS:
        sheet[c] = ""
    sheet.to_csv(OUT / "sheet.csv", index=False, encoding="utf-8-sig")       # utf-8-sig: opens cleanly in Excel
    gen[["item", "pair_id", "condition", "true_metaphor"]].to_csv(OUT / "key.csv", index=False)
    print(f"\nSaved {OUT}/sheet.csv ({len(sheet)} rows to rate, conditions hidden and shuffled) and {OUT}/key.csv (do not open).")
    print("Fill metaphorical / same_meaning / fluent with 1 or 0 (definitions at the top of steer_c.py), save as CSV, then:")
    print("    python steer_c.py --score")


def score():
    sheet = pd.read_csv(OUT / "sheet.csv", encoding="utf-8-sig")
    key = pd.read_csv(OUT / "key.csv")
    d = sheet.merge(key, on="item")
    missing = d[list(COLS)].isna().any(axis=1).sum()
    if missing:
        print(f"Note: {missing} rows are not fully rated; they are left out.")
    d = d.dropna(subset=list(COLS))
    d[list(COLS)] = d[list(COLS)].astype(int)
    d["all_three"] = d[list(COLS)].all(axis=1).astype(int)
    order = ["no vector", "+4 v", "+8 v", "random, size 8"]
    t = d.groupby("condition")[list(COLS) + ["all_three"]].mean().reindex(order)
    t.insert(0, "rated", d.groupby("condition").size().reindex(order))
    print("Blind ratings (share of outputs):")
    print((t[["rated"]].join((t.drop(columns="rated") * 100).round(0).astype("Int64").astype(str) + "%")).to_string())
    print("\nThe number that matters: 'all_three' (metaphorical AND same meaning AND fluent) for +4 v / +8 v vs no vector "
          "and vs random.")


if __name__ == "__main__":
    if "--score" in sys.argv:
        score()
    else:
        flags = ("--model", "--quant", "--layer", "--n")
        files = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and sys.argv[i] not in flags]
        run(files[0] if files else "manual_e.tsv")
