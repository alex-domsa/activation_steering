"""
Play with the metaphor direction: type a prompt, see what the model writes with and without the vector.

The vector (mean of metaphor - literal activations, train pairs) is ADDED at layer 17 to every token.

Run (in metalitpairs/):
    python try_vector.py                                  uses PROMPT below
    python try_vector.py "Describe your morning routine."  uses your prompt
Change PROMPT, STRENGTHS, CHAT or MAX_NEW below as you like.
"""
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- change these ----------------------------------------------------------
PROMPT = "Describe what happens when the stock market falls."
STRENGTHS = (0, 2, 4, 8, 16)      # 0 = no vector; 4 is about 25% of a typical activation at layer 17
CHAT = True                   # True: the prompt is sent as a chat message (normal assistant answer)
                              # False: plain text, e.g. "Sentence: prices went up\nParaphrase:"
MAX_NEW = 80                  # tokens to generate
LAYER = 17
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
VECTOR = "vectors/Qwen2.5-1.5B-Instruct/met_minus_lit_mean.npy"
# ----------------------------------------------------------------------------

if len(sys.argv) > 1:
    PROMPT = " ".join(sys.argv[1:])

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()
vec = torch.tensor(np.load(VECTOR)[LAYER], dtype=torch.float32, device=device)
strength = 0.0


def add_vector(module, inputs, output):
    """Adds strength * vector to the output of block 17 (hidden state 17), at every token."""
    if strength == 0:
        return output
    if isinstance(output, tuple):
        return (output[0] + strength * vec,) + tuple(output[1:])
    return output + strength * vec


model.model.layers[LAYER - 1].register_forward_hook(add_vector)

text = (tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
        if CHAT else PROMPT)
enc = tok(text, return_tensors="pt").to(device)
print(f"PROMPT: {PROMPT}\n")
for strength in STRENGTHS:
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
    answer = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"=== {'no vector' if strength == 0 else f'+{strength:g} x metaphor vector'} ===\n{answer}\n")
