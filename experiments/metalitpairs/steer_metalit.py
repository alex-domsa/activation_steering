"""
Steering test: does the metaphor - literal vector (from analyse_metalit.py) make the model turn an UNSEEN metaphor into
its literal version (subtract the vector), and a literal sentence into a metaphor (add it)?

Only the 62 TEST pairs are used (never used to build the vector or to choose the layer).
The vector is added at one layer to EVERY token (prompt and generated tokens), because the 'mean' vector is an
average per-token difference.

Prompt (neutral: it asks for a paraphrase, it never says "literal" or "metaphor"):
    "Sentence: {sentence}\nParaphrase:"

Part 1, likelihood (objective, no reading needed). For each test pair:
    metaphor -> literal:  prompt with the METAPHOR;  target = its true literal,  copy = the metaphor itself
    literal -> metaphor:  prompt with the LITERAL;   target = its true metaphor, copy = the literal itself
    score = mean log-probability per token of the target  -  of the copy   (per token, because literals are longer)
  Conditions: no vector; -m*v and +m*v for several m; random directions with the SAME norm as m*v (the control).
  Success: for metaphor -> literal, -m*v raises (target - copy) clearly more than a random vector of the same size,
  without making the target itself much less likely (that would be damage, not steering). +m*v the other way round.

Part 2, generation (to read): greedy paraphrases of the test metaphors with no vector, with -m*v, and with a random
vector; plus the reverse (+m*v on the literal) as a first look at direction C. Rough automatic hints: how often the
metaphor's changed word is still in the output, and word overlap with the true literal.

Run (in metalitpairs/, after analyse_metalit.py):
    python steer_metalit.py manual_e.tsv                       (layer recommended for the 'mean' vector, 20 generations)
    options: --layer=N  --mults=2,4,8  --n-gen=62  --feature=last  --model=...  --quant=4bit
Output: results/<model>/steer_L<layer>_<feature>.md      Time: about 5-10 minutes on the M2.
"""
import contextlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cache_metalit import Reader, _arg, cache_path, model_tag
from metalit_loader import load_pairs

TAG = model_tag()
FEATURE = _arg("--feature", "mean")
VEC_DIR = Path("vectors") / TAG
INFO = json.load(open(VEC_DIR / "vector_info.json"))
LAYER = int(_arg("--layer", INFO["recommended_layer"][FEATURE]))
MULTS = tuple(float(x) for x in _arg("--mults", "2,4,8").split(","))
N_GEN = int(_arg("--n-gen", 20))
N_RAND = 3                     # random directions averaged per strength (likelihood part)
MAX_NEW = 40
PROMPT = "Sentence: {s}\nParaphrase:"
SEED = 0


def overlap(a, b):
    ca, cb = (Counter(re.findall(r"[a-z']+", t.lower())) for t in (a, b))
    num = sum(ca[w] * cb[w] for w in ca)
    den = (sum(v * v for v in ca.values()) ** 0.5) * (sum(v * v for v in cb.values()) ** 0.5)
    return num / den if den else 0.0


class Steerer(Reader):
    @contextlib.contextmanager
    def steer(self, vec):
        """Adds vec to hidden state LAYER (output of block LAYER) at every position. vec=None: no change."""
        if vec is None:
            yield
            return
        v = torch.as_tensor(np.asarray(vec), dtype=torch.float32, device=self.device)

        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            hs = output[0] if is_tuple else output
            hs = hs + v.to(hs.dtype)
            return (hs,) + tuple(output[1:]) if is_tuple else hs
        handle = self.blocks[LAYER - 1].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def mean_logprob(self, prompt, cont, vec=None):
        """Mean log-probability per token of ' ' + cont after prompt."""
        p = self.tok(prompt)["input_ids"]
        c = self.tok(" " + cont, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p + c], device=self.device)
        with self.steer(vec):
            logits = self.model(input_ids=ids).logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        pos = torch.arange(len(p), len(p) + len(c), device=self.device)
        return lp[pos - 1, ids[0, pos]].mean().item()

    @torch.no_grad()
    def generate(self, prompt, vec=None):
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with self.steer(vec):
            out = self.model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip()


def main(tsv):
    t0 = time.time()
    df = load_pairs(tsv)
    test = df[df.split == "test"].reset_index(drop=True)
    held = set(pd.read_csv(VEC_DIR / "heldout_pairs.csv").pair_id)
    assert set(test.pair_id) == held, "test pairs differ from heldout_pairs.csv: re-run analyse_metalit.py"
    assert not set(test.pair_id) & set(INFO["train_pair_ids"]), "a test pair was used to build the vector"
    v = np.load(VEC_DIR / f"met_minus_lit_{FEATURE}.npy")[LAYER].astype(np.float32)
    r = Steerer()
    rng = np.random.default_rng(SEED)
    rand = [x / np.linalg.norm(x) * np.linalg.norm(v) for x in rng.normal(size=(N_RAND, v.size)).astype(np.float32)]
    typical = float(np.linalg.norm(np.load(cache_path())[f"met_{FEATURE}"][:, LAYER], axis=1).mean())
    print(f"{len(test)} test pairs | vector '{FEATURE}' at layer {LAYER}, norm {np.linalg.norm(v):.1f} "
          f"(typical activation at this layer {typical:.1f})")
    print("push size relative to a typical activation: " +
          ", ".join(f"m={m:g}: {m * np.linalg.norm(v) / typical:.0%}" for m in MULTS))

    conds = [("no vector", [None])]
    for m in MULTS:
        conds += [(f"-{m:g} v", [-m * v]), (f"+{m:g} v", [m * v]), (f"random, size {m:g}", [m * x for x in rand])]

    # ---- part 1: likelihood
    md = [f"# Steering test ({TAG}, '{FEATURE}' vector, layer {LAYER}, {len(test)} test pairs)\n",
          f"Prompt: `{PROMPT!r}`. Scores = mean log-prob per token. Delta = change from no vector.\n"]
    for task, src, tgt in (("metaphor -> literal", "metaphor", "literal"), ("literal -> metaphor", "literal", "metaphor")):
        rows, base = [], None
        for name, vecs in conds:
            T = np.mean([[r.mean_logprob(PROMPT.format(s=s), t, vec) for s, t in zip(test[src], test[tgt])] for vec in vecs], 0)
            C = np.mean([[r.mean_logprob(PROMPT.format(s=s), s, vec) for s in test[src]] for vec in vecs], 0)
            if base is None:
                base = (T, C)
            rows.append((name, T.mean(), C.mean(), (T - C).mean(), (T - base[0]).mean(), (C - base[1]).mean(),
                         ((T - C) - (base[0] - base[1])).mean(), (T > C).mean()))
        print(f"\n===== {task}: target = true {tgt}, copy = the {src} itself ({time.time() - t0:.0f}s) =====")
        head = (f"{'condition':18s} {'target':>7s} {'copy':>7s} {'tgt-copy':>8s} | {'d target':>8s} {'d copy':>7s} "
                f"{'d(tgt-copy)':>11s} | target preferred")
        print(head)
        md += [f"\n## {task} (target = true {tgt}, copy = the {src})\n",
               "| condition | target | copy | target - copy | d target | d copy | d(target - copy) | target preferred |",
               "|---|---|---|---|---|---|---|---|"]
        for name, T, C, D, dT, dC, dD, pref in rows:
            print(f"{name:18s} {T:7.2f} {C:7.2f} {D:+8.2f} | {dT:+8.2f} {dC:+7.2f} {dD:+11.2f} | {pref:.0%}")
            md.append(f"| {name} | {T:.2f} | {C:.2f} | {D:+.2f} | {dT:+.2f} | {dC:+.2f} | {dD:+.2f} | {pref:.0%} |")

    # ---- part 2: generation
    ms = MULTS[-2:]
    gen_conds = [("no vector", None)] + [(f"-{m:g} v", -m * v) for m in ms] + [(f"random, size {ms[-1]:g}", ms[-1] * rand[0])]
    sub = test.sample(frac=1, random_state=SEED).head(N_GEN)       # mixed sources
    stats = {name: {"kept": [], "ov_lit": [], "copy": []} for name, _ in gen_conds}
    md += [f"\n## Generations ({len(sub)} test pairs)\n",
           "Hints: [metaphor word still there? / overlap with the true literal]\n"]
    print(f"\n===== generations ({len(sub)} test pairs) =====")
    for k, row in enumerate(sub.itertuples(), 1):
        changed = set(row.changed_words.split())
        print(f"\n[{row.pair_id}] METAPHOR: {row.metaphor}\n      TRUE LITERAL: {row.literal}")
        md += [f"\n**[{row.pair_id}]** *{row.metaphor}*  ->  true literal: *{row.literal}*\n"]
        for name, vec in gen_conds:
            out = r.generate(PROMPT.format(s=row.metaphor), vec)
            kept = bool(changed & set(re.findall(r"[a-z']+", out.lower())))
            ov = overlap(out, row.literal)
            stats[name]["kept"].append(kept); stats[name]["ov_lit"].append(ov)
            stats[name]["copy"].append(re.sub(r"\W", "", out.lower()) == re.sub(r"\W", "", row.metaphor.lower()))
            print(f"   {name:18s} [{'kept' if kept else 'gone'} / {ov:.2f}] {out}")
            md.append(f"- {name} [{'kept' if kept else 'gone'} / {ov:.2f}]: {out}")
        out = r.generate(PROMPT.format(s=row.literal), ms[-1] * v)
        label = f"C: literal +{ms[-1]:g} v"
        print(f"   {label:18s} {out}")
        md.append(f"- C, literal + {ms[-1]:g} v: {out}")
        if k % 5 == 0:
            print(f"   ({time.time() - t0:.0f}s)")
    print("\nSummary over generations (metaphor word kept / exact copy / overlap with true literal):")
    md += ["\n## Generation summary\n", "| condition | metaphor word kept | exact copy | overlap with literal |", "|---|---|---|---|"]
    for name, s in stats.items():
        print(f"   {name:18s} {np.mean(s['kept']):.0%} kept | {np.mean(s['copy']):.0%} copies | overlap {np.mean(s['ov_lit']):.2f}")
        md.append(f"| {name} | {np.mean(s['kept']):.0%} | {np.mean(s['copy']):.0%} | {np.mean(s['ov_lit']):.2f} |")
    out_path = Path("results") / TAG / f"steer_L{LAYER}_{FEATURE}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nSaved {out_path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    flags = ("--model", "--quant", "--layer", "--mults", "--n-gen", "--feature")
    files = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and sys.argv[i] not in flags]
    main(files[0] if files else "manual_e.tsv")
