"""
Joint-prompt cache: show the model BOTH sentences in one prompt and record what it does.

Before, each sentence went through the model on its own and were compared afterwards.
Here the model reads the pair together, with a question at the end:

    Sentence A: <s1>
    Sentence B: <s2>

    Question: Do these two sentences describe the same underlying relationship or process,
    even if they are about different things?
    Answer (yes or no):

For every pair we save
    last      the activation of the FINAL token at every layer (the point where the model is about
              to answer), shape (n_pairs, n_layers + 1, d_model)
    yes, no   the model's log-probability of answering "yes" / "no" (its behaviour)
Row i of each array is row i of the matching _meta.csv (same pairs and splits as before).

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cache_activations import MODEL, pick_device, SAVE_DTYPE
from storyanalogy_loader import load_storyanalogy, CONTRAST

# ---- Settings --------------------------------------------------------------
PROMPT = ("Sentence A: {s1}\n"
          "Sentence B: {s2}\n\n"
          "Question: Do these two sentences describe the same underlying relationship or process, "
          "even if they are about different things?\n"
          "Answer (yes or no):")
JOINT_ROOT = Path("cache_joint")
JOINT_DIR = JOINT_ROOT / MODEL.replace("/", "__") / CONTRAST
# ----------------------------------------------------------------------------


def make_backend():
    """Load the model once; return get_features(prompt) -> (activations, yes_logprob, no_logprob).
    activations has shape (n_layers + 1, d_model): the final token at every layer (0 = embeddings)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device()
    print(f"Loading {MODEL} on {device} ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    model.eval()

    def first_ids(words):
        # first token of each spelling ("yes", " yes", "Yes", " Yes"); the answer may start with any of them
        return sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})
    yes_ids = torch.tensor(first_ids(["yes", " yes", "Yes", " Yes"]), device=device)
    no_ids = torch.tensor(first_ids(["no", " no", "No", " No"]), device=device)
    print(f"answer token ids  yes: {yes_ids.tolist()}  no: {no_ids.tolist()}")

    def get_features(prompt):
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = torch.stack(out.hidden_states)[:, 0, -1, :]               # (layers+1, d): final token only
        logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)    # next-token distribution
        yes = torch.logsumexp(logp[yes_ids], dim=0).item()
        no = torch.logsumexp(logp[no_ids], dim=0).item()
        return hs.float().cpu().numpy(), yes, no

    return get_features


def cache_split(name, df, get_features):
    out_npz, out_meta = JOINT_DIR / f"{name}.npz", JOINT_DIR / f"{name}_meta.csv"
    if out_npz.exists() and out_meta.exists():
        print(f"[{name}] already cached - skipping (delete {out_npz} to redo)")
        return
    n = len(df)
    last, yes, no = [], [], []
    for i, (s1, s2) in enumerate(zip(df["s1"], df["s2"]), start=1):
        a, y, nn = get_features(PROMPT.format(s1=s1, s2=s2))
        last.append(a); yes.append(y); no.append(nn)
        if i % 100 == 0 or i == n:
            print(f"[{name}] {i}/{n} pairs")
    last = np.stack(last)
    assert last.shape[0] == n and not np.isnan(last).any(), "problem in activations"
    np.savez(out_npz, last=last.astype(SAVE_DTYPE), yes=np.array(yes, dtype=np.float32), no=np.array(no, dtype=np.float32))
    df.to_csv(out_meta, index=False)
    print(f"[{name}] saved {out_npz} {last.shape}")


def load_joint(name, joint_dir=JOINT_DIR):
    """Use this in analysis scripts. Returns (dict with last/yes/no as float32 arrays, meta DataFrame)."""
    data = np.load(Path(joint_dir) / f"{name}.npz")
    return {k: data[k].astype(np.float32) for k in data.files}, pd.read_csv(Path(joint_dir) / f"{name}_meta.csv")


def main(csv_path):
    JOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Joint cache folder: {JOINT_DIR}")
    splits = load_storyanalogy(csv_path, CONTRAST)
    example = PROMPT.format(s1=splits["train"].s1[0], s2=splits["train"].s2[0])
    print("\nExample prompt:\n" + "-" * 60 + f"\n{example}\n" + "-" * 60)
    get_features = make_backend()
    for name in ("train", "val", "test"):
        cache_split(name, splits[name], get_features)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "StoryAnalogy.csv")
