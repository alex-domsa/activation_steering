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






#Testing adding activation hooks to steer the model's output in the direction of the difference between the two sentences' activations.
LAYER = 12
direction = acts_b[LAYER] - acts_a[LAYER]
direction = (direction / direction.norm()).to(device)
SCALE = 4.0

def steer_hook(module, args, output):
    if isinstance(output, tuple):
        return (output[0] + SCALE * direction, *output[1:])
    return output + SCALE * direction

prompt = tok("Complete the sentence: The sun is to the planets as", return_tensors="pt").to(device)

def generate():
    with torch.no_grad():
        ids = model.generate(**prompt, max_new_tokens=50, do_sample=False)
    return tok.decode(ids[0], skip_special_tokens=True)

print("\nBaseline :", generate())
handle = model.model.layers[LAYER].register_forward_hook(steer_hook)
print("Steered  :", generate())
handle.remove()