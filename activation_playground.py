import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
model.eval()

a = "The cornea takes the light and reflects it back to the eye. Clarity is provided in vision"
b = "A person takes the culture around them and reflects it outward. Understanding emerges through connection"
print(a)
print(b)

def last_token_acts(text):
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return torch.stack([h[0, -1, :] for h in out.hidden_states]).cpu()

acts_a, acts_b = last_token_acts(a), last_token_acts(b)
print(f"Activation tensor shape per sentence: {tuple(acts_a.shape)} (layers+1, d_model)")

cos = torch.nn.functional.cosine_similarity(acts_a, acts_b, dim=-1)
for layer, c in enumerate(cos):
    print(f"layer {layer:2d}  cosine similarity = {c:.3f}")