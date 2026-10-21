import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct" # ~0.5B params, 24 layers, ~2GB in float32
device = "mps" if torch.backends.mps.is_available() else "cpu" #M-series Mac GPU (mps) else CPU

tok = AutoTokenizer.from_pretrained(MODEL) #text <-> token ids
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device) #Load weights onto device
model.eval() #inference mode (no dropout)

a = "The cornea takes the light and reflects it back to the eye. Clarity is provided in vision"
b = "A person takes the culture around them and reflects it outward. Understanding emerges through connection"
print(a)
print(b)

def last_token_acts(text):
    inputs = tok(text, return_tensors="pt").to(device) # tokenise. move tensors to the device
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True) # forward pass, also return every layer's activations
    # hidden_states = 25 tensors (embeddings + 24 layers), each (1, n_tokens, 896).
    # Take the last token's vector from each and stack into (25, 896).
    return torch.stack([h[0, -1, :] for h in out.hidden_states]).cpu()

acts_a, acts_b = last_token_acts(a), last_token_acts(b) # one (25, 896) tensor per sentence
print(f"Activation tensor shape per sentence: {tuple(acts_a.shape)} (layers+1, d_model)")

cos = torch.nn.functional.cosine_similarity(acts_a, acts_b, dim=-1) # similarity per layer: 25 values
for layer, c in enumerate(cos):
    print(f"layer {layer:2d}  cosine similarity = {c:.3f}")





#Testing adding activation hooks to steer the model's output in the direction of the difference between the two sentences' activations

LAYER = 12 # layer to read the direction from and to add it at
direction = acts_b[LAYER] - acts_a[LAYER] # vector pointing from sentence a's state to b's
direction = (direction / direction.norm()).to(device) # scale to length 1 so SCALE alone sets the strength
SCALE = 4.0 # steering strength

def steer_hook(module, args, output): # runs automatically each time the hooked layer finishes
    if isinstance(output, tuple): # some versions return (hidden_states, extras...)  //// older version = tuple //// newer = activation ////
        return (output[0] + SCALE * direction, *output[1:]) # add the vector to every token's activation
    return output + SCALE * direction # other versions return the tensor alone

prompt = tok("Complete the sentence: The sun is to the planets as", return_tensors="pt").to(device) # prompt as token IDs

def generate():
    with torch.no_grad():
        ids = model.generate(**prompt, max_new_tokens=50, do_sample=False) # greedy: same output each run
    return tok.decode(ids[0], skip_special_tokens=True) # IDs back to text (includes the prompt)

print("\nBaseline :", generate()) # output with no steering
handle = model.model.layers[LAYER].register_forward_hook(steer_hook) # attach the hook to that layer
print("Steered  :", generate()) # output with the vector added
handle.remove() # detaching the hook so later runs aren't affected