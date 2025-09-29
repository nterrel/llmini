# sample.py
import torch
from llmini.data import CharDataLoader
from llmini.utils import parse_arguments, get_model_from_args
from llmini.config import BLOCK_SIZE, DEVICE


def load_checkpoint(checkpoint_path, model):
    """
    Load a model checkpoint from the specified path.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the checkpoint into.
    """
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    if "model" in ckpt:
        state_dict = ckpt["model"]
        print("Loaded full checkpoint.")
    else:
        state_dict = ckpt  # Assume the checkpoint contains only the model weights
        print("Loaded model-only checkpoint.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()


# Initialize data loader
data_loader = CharDataLoader(block_size=BLOCK_SIZE, device=DEVICE)
vocab_size = data_loader.vocab_size

# Parse command-line arguments
args = parse_arguments()

# Initialize the model
model = get_model_from_args(args, vocab_size, BLOCK_SIZE, DEVICE)

# Load the checkpoint
load_checkpoint("checkpoints/tinygpt_char_small.pt", model)

prompt = "Duke of Dukington:"
print(f"Prompt: \n {prompt}")

# Tokenize the prompt
ids = torch.tensor([[data_loader.stoi[c] for c in prompt]])

with torch.no_grad():
    # Generate text with temperature and top-k sampling
    out = model.generate(ids, max_new_tokens=600, temperature=0.9, top_k=50)

# Decode the output and print the generated text
generated_text = data_loader.decode(out[0].tolist())
print(f"Generated text: \n{generated_text}")
