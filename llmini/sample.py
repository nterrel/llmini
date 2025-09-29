# sample.py
import torch
from llmini.data import load_char_data
from llmini.utils import parse_arguments, get_model_from_args

# Load vocabulary and mappings
vocab_size, get_batch, decode, stoi, itos = load_char_data()

# Parse command-line arguments
args = parse_arguments()

# Initialize the model
model = get_model_from_args(args, vocab_size, block_size=256, device="cpu")

# Define DEBUG based on arguments
DEBUG = args.debug

# Load the checkpoint
ckpt = torch.load("checkpoints/tinygpt_char_small.pt", map_location="cpu")

# Check if the checkpoint contains the 'model' key
if "model" in ckpt:
    state_dict = ckpt["model"]
    print("Loaded full checkpoint.")
else:
    state_dict = ckpt  # Assume the checkpoint contains only the model weights
    print("Loaded model-only checkpoint.")

# Load the state_dict into the model
model.load_state_dict(state_dict, strict=True)
model.eval()

# prime with a prompt
prompt = "Duke of Dukington:"
# Debugging: Print the prompt and its tokenized representation
print(f"Prompt: \n {prompt}")
ids = torch.tensor([[stoi[c] for c in prompt]])  # Use stoi for encoding

if DEBUG:
    print(f"Tokenized prompt: {ids}")

with torch.no_grad():
    # Generate text with temperature and top-k sampling
    out = model.generate(ids, max_new_tokens=600, temperature=0.9, top_k=50)

# Fix the decode function to handle unknown tokens


def decode(token_ids):
    try:
        # Convert token IDs to integers and decode
        return ''.join(itos[int(token_id)] for token_id in token_ids[0])
    except KeyError as e:
        print(f"Error: Unknown token ID {e}")
        return ''.join(itos.get(int(token_id), '?') for token_id in token_ids[0])


# Decode the output and print the generated text
generated_text = decode(out)

if DEBUG:
    print(f"Generated token IDs: {out}")
    print(f"itos mapping: {itos}")

print(f"Generated text: \n{generated_text}")
