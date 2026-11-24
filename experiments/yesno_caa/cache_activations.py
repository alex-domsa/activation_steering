import sys
from pathlib import Path

import numpy as np
import pandas as pd

from storyanalogy_loader import load_storyanalogy, CONTRAST   # CONTRAST is set in the loader file

# ---- Settings --------------------------------------------------------------
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BACKEND = "hf"            # "hf" = Hugging Face hooks (tested idea), "tl" = TransformerLens (untested sketch)
SAVE_DTYPE = np.float16   # float16 halves file size; use np.float32 forfull precision
MEAN_SKIP_FIRST = True    # leave the first token out of the mean (see notes below)
CACHE_ROOT = Path("cache")
CACHE_DIR = CACHE_ROOT / MODEL.replace("/", "__") / CONTRAST   # one folder per model AND contrast: different
                                                               # contrasts use different pairs, so they must not share a cache
# ----------------------------------------------------------------------------


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"      # school PC (NVIDIA) and Colab
    if torch.backends.mps.is_available():
        return "mps"       # Apple silicon Mac
    return "cpu"


def make_backend():
    """Load the model once and return get_activations(text).

    get_activations returns two numpy arrays, each (n_layers + 1, d_model):
        last : activation of the final token at every layer
        mean : average activation over tokens at every layer
    Index 0 = embeddings, index i = output of transformer block i.
    This is the ONLY place that knows which library is used - swap it here."""
    if BACKEND == "hf":
        return _backend_hf()
    if BACKEND == "tl":
        return _backend_transformer_lens()
    raise ValueError(f"Unknown BACKEND {BACKEND!r}")


def _backend_hf():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device()
    print(f"Loading {MODEL} on {device} ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    model.eval()

    def get_activations(text):
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = torch.stack(out.hidden_states)[:, 0]            # (25, n_tokens, 896)
        last = hs[:, -1, :]                                   # (25, 896)
        body = hs[:, 1:, :] if (MEAN_SKIP_FIRST and hs.shape[1] > 1) else hs
        mean = body.mean(dim=1)                               # (25, 896)
        return last.float().cpu().numpy(), mean.float().cpu().numpy()

    return get_activations


def _backend_transformer_lens():
    # UNTESTED SKETCH - check against the TransformerLens docs before relying on it.
    # Known difference: the last hidden state in Hugging Face is taken AFTER the final
    # layer norm, but resid_post of the last block is BEFORE it, so index 24 won't match.
    import torch
    from transformer_lens import HookedTransformer

    device = pick_device()
    print(f"Loading {MODEL} with TransformerLens on {device} ...")
    model = HookedTransformer.from_pretrained(MODEL, device=device)
    model.eval()
    n_layers = model.cfg.n_layers

    def get_activations(text):
        tokens = model.to_tokens(text, prepend_bos=False)     # no extra start token, to match HF
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        layers = [cache["hook_embed"]] + [cache["resid_post", i] for i in range(n_layers)]
        hs = torch.stack([h[0] for h in layers])              # (25, n_tokens, 896)
        last = hs[:, -1, :]
        body = hs[:, 1:, :] if (MEAN_SKIP_FIRST and hs.shape[1] > 1) else hs
        mean = body.mean(dim=1)
        return last.float().cpu().numpy(), mean.float().cpu().numpy()

    return get_activations


def cache_split(name, df, get_activations):
    out_npz = CACHE_DIR / f"{name}.npz"
    out_meta = CACHE_DIR / f"{name}_meta.csv"
    if out_npz.exists() and out_meta.exists():
        print(f"[{name}] already cached - skipping (delete {out_npz} to redo)")
        return

    n = len(df)
    arrays = {k: [] for k in ("last_s1", "last_s2", "mean_s1", "mean_s2")}
    for i, (s1, s2) in enumerate(zip(df["s1"], df["s2"]), start=1):
        last1, mean1 = get_activations(s1)
        last2, mean2 = get_activations(s2)
        arrays["last_s1"].append(last1); arrays["mean_s1"].append(mean1)
        arrays["last_s2"].append(last2); arrays["mean_s2"].append(mean2)
        if i % 100 == 0 or i == n:
            print(f"[{name}] {i}/{n} pairs")

    stacked = {k: np.stack(v) for k, v in arrays.items()}      # (n, 25, 896)
    for k, v in stacked.items():
        assert v.shape[0] == n and not np.isnan(v).any(), f"problem in {k}"
    np.savez(out_npz, **{k: v.astype(SAVE_DTYPE) for k, v in stacked.items()})
    df.to_csv(out_meta, index=False)
    print(f"[{name}] saved {out_npz} {stacked['last_s1'].shape}")


def load_cached(name, cache_dir=CACHE_DIR):
    """Use this in your analysis scripts. Returns (arrays_dict, meta_dataframe)."""
    data = np.load(Path(cache_dir) / f"{name}.npz")
    meta = pd.read_csv(Path(cache_dir) / f"{name}_meta.csv")
    return {k: data[k].astype(np.float32) for k in data.files}, meta


def main(csv_path):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cache folder: {CACHE_DIR}")
    splits = load_storyanalogy(csv_path, CONTRAST)
    get_activations = make_backend()
    for name in ("train", "val", "test"):
        cache_split(name, splits[name], get_activations)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "StoryAnalogy.csv")
