"""
Cache the model's activations for every metaphor sentence and every literal sentence, pair by pair.

Each sentence is run through the model on its own (no prompt, no padding), and two summaries of its hidden states are
stored for every layer:
    last  = the state of the final token
    mean  = the mean state over the tokens (the first token is left out: it is an attention sink with a huge norm)

How the pairing is kept (the arrays are row-aligned):
    pair_id[i]          the line number of the pair in manual_e.tsv (shared by both sentences)
    metaphor[i]         its metaphorical sentence      met_last[i], met_mean[i]   its activations   (kind = metaphor)
    literal[i]          its literal sentence           lit_last[i], lit_mean[i]   its activations   (kind = literal)
So row i of EVERY array belongs to the same pair; the metaphor - literal difference of pair i is met_*[i] - lit_*[i].
After saving, the script reloads the file and checks this: texts against the loader, and three random pairs re-run
through the model and compared with their cached rows.

Run:
    python cache_metalit.py manual_e.tsv
    python cache_metalit.py manual_e.tsv --model=Qwen/Qwen2.5-7B-Instruct --quant=4bit      (Colab T4; pip install bitsandbytes)
Output: cache/<model>/acts.npz 
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metalit_loader import load_pairs


def _arg(name, default):
    """--name=value or --name value"""
    a = sys.argv[1:]
    for i, x in enumerate(a):
        if x.startswith(name + "="):
            return x.split("=", 1)[1]
        if x == name and i + 1 < len(a):
            return a[i + 1]
    return default


MODEL = _arg("--model", "Qwen/Qwen2.5-1.5B-Instruct")
QUANT = _arg("--quant", None)        # "8bit" or "4bit" on an NVIDIA GPU (needs: pip install bitsandbytes)
SEED = 0


def model_tag(name=MODEL):
    return name.rstrip("/").split("/")[-1]


def cache_path(name=MODEL):
    return Path("cache") / model_tag(name) / "acts.npz"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Reader:
    """Loads the model and reads hidden states. Hidden state 0 = embeddings, hidden state L = output of block L."""

    def __init__(self, name=MODEL, quant=QUANT):
        self.name, self.device = name, pick_device()
        print(f"Loading {name} on {self.device} ...")
        self.tok = AutoTokenizer.from_pretrained(name)
        if quant and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            qc = (BitsAndBytesConfig(load_in_8bit=True) if quant == "8bit" else
                  BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
            self.model = AutoModelForCausalLM.from_pretrained(name, quantization_config=qc, device_map={"": 0}).eval()
            kind = f"{quant} quantised"
        else:
            dtype = ((torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
                     if self.device == "cuda" else torch.float32)          # Mac / CPU: full precision
            self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(self.device).eval()
            kind = str(dtype)
        self.blocks = self.model.model.layers
        self.n_layers = len(self.blocks)
        print(f"{self.n_layers} layers, {kind}")

    @torch.no_grad()
    def states(self, text):
        """(n_layers + 1, n_tokens, d) hidden states of one sentence, plus the tokens' character offsets."""
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        out = self.model(**enc.to(self.device), output_hidden_states=True)
        return torch.stack(out.hidden_states)[:, 0].float().cpu().numpy(), offsets

    def features(self, text):
        """(last, mean, n_tokens): final-token state and mean state (first token left out), each (n_layers + 1, d)."""
        hs, _ = self.states(text)
        body = hs[:, 1:] if hs.shape[1] > 1 else hs
        return hs[:, -1], body.mean(axis=1), hs.shape[1]


def verify(path, df, reader=None, n_check=3):
    """Reload the cache and prove that row i of every array is pair pair_id[i]."""
    z = np.load(path)
    by_id = df.set_index("pair_id")
    ids = z["pair_id"]
    assert len(set(ids.tolist())) == len(ids), "duplicate pair ids in the cache"
    assert (by_id.loc[ids, "metaphor"].values == z["metaphor"]).all(), "a cached metaphor does not match its pair id"
    assert (by_id.loc[ids, "literal"].values == z["literal"]).all(), "a cached literal does not match its pair id"
    n = len(ids)
    for k in ("met_last", "met_mean", "lit_last", "lit_mean", "met_ntok", "lit_ntok", "split"):
        assert len(z[k]) == n, f"{k} has {len(z[k])} rows, expected {n}"
    print(f"Check 1: all {n} cached texts match their pair id in the loader (metaphor AND literal).")
    if reader is not None:
        rng = np.random.default_rng(SEED + 1)
        for i in rng.choice(n, size=min(n_check, n), replace=False):
            ml, mm, _ = reader.features(str(z["metaphor"][i]))
            ll, lm, _ = reader.features(str(z["literal"][i]))
            ok = all(np.allclose(a, b, rtol=1e-3, atol=1e-3 * np.abs(b).max())
                     for a, b in ((ml, z["met_last"][i]), (mm, z["met_mean"][i]), (ll, z["lit_last"][i]), (lm, z["lit_mean"][i])))
            print(f"Check 2: pair {ids[i]} re-run through the model, matches cached row {i}: {ok}")
            assert ok, f"pair {ids[i]}: cached activations do not match a fresh run"


def main(tsv):
    t0 = time.time()
    df = load_pairs(tsv)
    out = cache_path()
    if out.exists() and "--force" not in sys.argv:
        print(f"{out} already exists (add --force to rebuild). Verifying it instead.")
        verify(out, df)
        return
    r = Reader()
    n = len(df)
    store = {}
    for i, row in enumerate(df.itertuples()):
        for side, text in (("met", row.metaphor), ("lit", row.literal)):
            last, mean, ntok = r.features(text)
            if not store:
                shape = (n,) + last.shape
                store = {k: np.zeros(shape, np.float32) for k in ("met_last", "met_mean", "lit_last", "lit_mean")}
                store.update(met_ntok=np.zeros(n, int), lit_ntok=np.zeros(n, int))
            store[f"{side}_last"][i], store[f"{side}_mean"][i], store[f"{side}_ntok"][i] = last, mean, ntok
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n} pairs ({time.time() - t0:.0f}s)")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = {k: np.array(df[k].tolist(), dtype=str) for k in ("split", "source", "metaphor", "literal")}   # plain unicode
    np.savez(out, pair_id=df.pair_id.to_numpy(), model=np.array(MODEL), **text, **store)
    print(f"Saved {out}: {n} pairs x 2 sentences, {store['met_last'].shape[1]} hidden states of size "
          f"{store['met_last'].shape[2]} ({out.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)")
    verify(out, df, r)


if __name__ == "__main__":
    files = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and sys.argv[i] not in ("--model", "--quant")]
    main(files[0] if files else "manual_e.tsv")
