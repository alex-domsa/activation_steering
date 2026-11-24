"""
Transfer test: does the steering direction v still work when the QUESTION IS WORDED DIFFERENTLY?

v was built from the original prompt. If v only meant "the model is about to say yes", it would push
"yes" up in EVERY prompt, even where "yes" means the opposite. If v means "these two sentences share a
relationship", it should follow the meaning. Three new prompts, all using the same validation pairs:

  reworded         different wording of the same question          (+v should push "yes" UP)
  swapped order    same question, sentence A and B swapped         (+v should push "yes" UP)
  inverted         "Are these two about completely unrelated        (+v should push "yes" DOWN if v is
                   processes with nothing in common?"                about analogy; UP if v is just "yes")

The inverted prompt is the decisive one: there, a "no" answer is the analogy-like answer.
Same layer and same vector as steer_yesno.py (v is read from steering_vector.npy). Random directions of
the same length are the control. TEST is not touched.

Run:   python steer_transfer.py
Needs: steer_yesno.py already run (it saves steering_vector.npy and steering_layer_selection.csv) and
       the cache_joint folder; MODEL set in cache_activations.py (Qwen2.5-1.5B-Instruct).
Time:  about 7,000 short forward passes with the defaults (lower N_PER_GROUP / M_GRID for a quick look).
Output: results/<model>/<contrast>/joint/steering_transfer.csv and steering_transfer.png
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cache_activations import MODEL
from cache_joint import PROMPT, load_joint
from storyanalogy_loader import CONTRAST
from find_layer import auc
from steer_yesno import HFSteerer, boot_ci, random_direction

warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Settings ---------------------------------------------------------------
N_PER_GROUP = 60          # validation pairs per group (the same pairs for every prompt)
M_GRID = (2, 4, 8)
K_RANDOM = 2
SEED = 0
RESULTS_DIR = Path("results") / MODEL.replace("/", "__") / CONTRAST / "joint"

VARIANTS = {
    # name: (prompt template, expected sign of the change under 'v = analogy')
    "reworded": ("Here are two statements.\n1) {s1}\n2) {s2}\n"
                 "Do these two statements follow the same pattern or mechanism, even though they involve different things? "
                 "Respond with yes or no.\nAnswer:", +1),
    "swapped order": (PROMPT.replace("{s1}", "@@").replace("{s2}", "{s1}").replace("@@", "{s2}"), +1),
    "inverted": ("Sentence A: {s1}\nSentence B: {s2}\n\n"
                 "Question: Are these two sentences about completely unrelated processes with nothing in common?\n"
                 "Answer (yes or no):", -1),
}
# -----------------------------------------------------------------------------


def main(steerer=None):
    rng = np.random.default_rng(SEED)
    sel_path, vec_path = RESULTS_DIR / "steering_layer_selection.csv", RESULTS_DIR / "steering_vector.npy"
    if not (sel_path.exists() and vec_path.exists()):
        sys.exit("Run steer_yesno.py first (it saves steering_vector.npy and steering_layer_selection.csv).")
    layer = int(pd.read_csv(sel_path).sort_values("gain").iloc[-1].layer)
    v = np.load(vec_path)
    _, M_va = load_joint("val")
    y = M_va.label.to_numpy()
    idx = np.concatenate([rng.choice(np.flatnonzero(y == g), size=min(N_PER_GROUP, int((y == g).sum())), replace=False)
                          for g in (0, 1)])
    labels = y[idx]
    rand_dirs = [random_direction(rng, np.linalg.norm(v), len(v)) for _ in range(K_RANDOM)]
    if steerer is None:
        steerer = HFSteerer()
    print(f"Layer {layer} (from steer_yesno.py) | {len(idx)} validation pairs | {K_RANDOM} random controls | steering at the last token")

    rows = []
    for name, (template, expected) in VARIANTS.items():
        prompts = [template.format(s1=M_va.s1[i], s2=M_va.s2[i]) for i in idx]
        base, _ = steerer.logodds(prompts)
        # does the model itself answer this prompt sensibly? (AUC in the sense of the question)
        own = auc(expected * base, labels)
        print(f"\n=== {name} prompt: the model's own answers separate the groups with AUC {own:.3f} "
              f"(0.5 = no; in the sense 'yes' = {'analogy' if expected > 0 else 'unrelated'}), says yes on {np.mean(base > 0):.0%} of pairs")
        print("  m |   +m*v change [95% CI]   |   -m*v change [95% CI]   | random: largest |change| | verdict at +m*v")
        for m in M_GRID:
            up = steerer.logodds(prompts, layer=layer, delta=m * v)[0] - base
            dn = steerer.logodds(prompts, layer=layer, delta=-m * v)[0] - base
            r_all = [steerer.logodds(prompts, layer=layer, delta=s * m * r)[0] - base for r in rand_dirs for s in (+1, -1)]
            rmax = max(abs(x.mean()) for x in r_all)
            ul, uh = boot_ci(up, rng); dl, dh = boot_ci(dn, rng)
            if abs(up.mean()) <= rmax or ul <= 0 <= uh:
                verdict = "no clear effect"
            elif np.sign(up.mean()) == expected:
                verdict = "follows the MEANING (analogy sense)" if expected < 0 else "pushes yes up (as expected)"
            else:
                verdict = "pushes 'yes' up regardless of meaning" if expected < 0 else "pushes yes DOWN (unexpected)"
            print(f"{m:3d} | {up.mean():+7.3f} [{ul:+.3f}, {uh:+.3f}] | {dn.mean():+7.3f} [{dl:+.3f}, {dh:+.3f}] | {rmax:22.3f} | {verdict}")
            rows.append(dict(prompt=name, m=m, up=up.mean(), up_lo=ul, up_hi=uh, down=dn.mean(), down_lo=dl, down_hi=dh,
                             rand_max_abs=rmax, own_auc=own, expected_sign=expected, verdict=verdict))
    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / "steering_transfer.csv", index=False)

    print("\nHow to read it:")
    print("- 'change' = steered minus unsteered log-odds of yes vs no, averaged over the pairs.")
    print("- reworded / swapped order: if v transfers, +v pushes 'yes' up (positive) well above the random largest.")
    print("- inverted (the decisive one): NEGATIVE with +v means v follows the meaning (more analogous -> less")
    print("  'unrelated'); POSITIVE means v just pushes the word 'yes' whatever it is asked.")
    print("- If the model's own AUC for a prompt is near 0.5, that prompt is not informative about v.")
    print(f"\nSaved: {RESULTS_DIR}/steering_transfer.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for name, g in res.groupby("prompt", sort=False):
            ax.errorbar(g.m, g.up, yerr=[g.up - g.up_lo, g.up_hi - g.up], marker="o", label=f"{name}  (+v)")
        ax.axhline(0, color="red", lw=1); ax.set_xscale("log", base=2)
        ax.set_xlabel("multiplier m"); ax.set_ylabel("change in log-odds of 'yes' with +m*v"); ax.legend(fontsize=8)
        ax.set_title(f"Does v transfer to other wordings? - layer {layer}")
        fig.tight_layout(); fig.savefig(RESULTS_DIR / "steering_transfer.png", dpi=150)
        print(f"Plot saved to {RESULTS_DIR / 'steering_transfer.png'}")
    except ImportError:
        print("matplotlib not installed - skipped the plot")


if __name__ == "__main__":
    main()
