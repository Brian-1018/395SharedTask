import torch
from tokenizers import Tokenizer
from types import SimpleNamespace
import json
from model_extra import Bert
from transformers import AutoModelForMaskedLM, AutoTokenizer

# --- Load model config ---
with open("../configs/small.json", "r") as f:
    config_dict = json.load(f)
config = SimpleNamespace(**config_dict)

# --- Load tokenizer ---
tokenizer = Tokenizer.from_file("../tokenizers/tokenizer_10M.json")
PAD_ID = tokenizer.token_to_id("<pad>") or 0
MASK_ID = tokenizer.token_to_id("<mask>") or tokenizer.token_to_id("<unk>")
CLS_ID = tokenizer.token_to_id("<s>") or tokenizer.token_to_id("<cls>")
print(f"PAD_ID: {PAD_ID}")
print(f"MASK_ID: {MASK_ID}")
print(f"UNK_ID: {tokenizer.token_to_id('<unk>')}")
print(f"<mask> in vocab?: {'<mask>' in tokenizer.get_vocab()}")
good_token_id = tokenizer.token_to_id("Ġgood")
print(f"'good' token ID: {good_token_id}")
print(f"Token ID for 'Ġgood': {tokenizer.token_to_id('Ġgood')}")
print(f"Token ID 829 corresponds to: '{tokenizer.id_to_token(829)}'")
if good_token_id == 829:
    print("✅ Token ID for 'good' matches expected value (829).")
else:
    print("❌ Token ID for 'good' does NOT match expected value! Check tokenizer consistency.")

# --- Load model ---
model = Bert(config)
# Load the state dict from the checkpoint (using the state_dict file)
checkpoint = torch.load("../checkpoints/run_004_4_11_2025_babylm_10M_state_dict.bin", map_location="cpu")
state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
print("\n--- Checkpoint Keys ---")
for key in state_dict:
    print(key)
print("------------------------")
model.load_state_dict(state_dict)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def prepare_input(text):
    mask_token_str = next((tok for tok, idx in tokenizer.get_vocab().items() if idx == MASK_ID), None)
    if mask_token_str is None:
        raise ValueError("Could not find <mask> token string in vocab.")

    # Only replace if the raw string is not equal to the token string
    if "<mask>" != mask_token_str:
        text = text.replace("<mask>", mask_token_str)

    if "<mask>" not in text:
        raise ValueError("Input text must contain the '<mask>' token.")
 
    # Split the input text using "<mask>" to locate it without relying on tokenization
    prefix, _, suffix = text.partition("<mask>")
    prefix_ids = tokenizer.encode(prefix.strip()).ids
    suffix_ids = tokenizer.encode(suffix.strip()).ids
    input_ids = prefix_ids + [MASK_ID] + suffix_ids
    mask_index = len(prefix_ids)
 
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    return input_ids_tensor, mask_index

def get_predictions(model, input_ids, mask_index, top_k=5):
    # Transpose input_ids to shape [seq_len, batch] as expected by the model's get_contextualized method
    input_ids_transposed = input_ids.t()
    seq_len = input_ids_transposed.size(0)
    
    # Create an attention mask of shape [batch, seq_len] (all tokens are valid here)
    # In training, this comes from the dataset; here we assume no tokens are masked
    attention_mask = torch.zeros(input_ids.size(0), seq_len, dtype=torch.bool, device=device)

    # Prepare masked language modeling labels: set all positions to -100 except the masked token
    # Mark the masked position with any dummy value != -100 to indicate where to compute logits
    masked_lm_labels = torch.full_like(input_ids_transposed, fill_value=-100)
    masked_lm_labels[mask_index, 0] = 0  # any value != -100 is fine; model uses this to know where to predict

    with torch.no_grad():
        # First, get the contextualized embeddings
        embeddings = model.get_contextualized(input_ids_transposed, attention_mask)
        # Then pass the embeddings and the masked LM labels to the classifier
        logits = model.classifier(embeddings, masked_lm_labels, None)

        # The classifier returns logits only for positions where masked_lm_labels != -100.
        # Since we masked one token, logits should be of shape [1, vocab_size].
        if logits.dim() == 2:
            mask_logits = logits[0]
        else:
            mask_logits = logits[mask_index, 0]

        topk_debug = min(10, mask_logits.numel())
        top_logit_vals, top_logit_ids = torch.topk(mask_logits, k=topk_debug)
        print("\n--- Raw Logits (Top 10) ---")
        for i, logit_id in enumerate(top_logit_ids):
            token = tokenizer.id_to_token(int(logit_id))
            print(f"{i+1}. {token}: {top_logit_vals[i].item():.4f}")
        print("----------------------------")

        # Get softmax probabilities and then the top-k predictions
        probs = torch.softmax(mask_logits, dim=-1)
        k = min(top_k, probs.numel())
        top_probs, top_indices = torch.topk(probs, k=k)

        return top_indices, top_probs

# --- Run inference ---
input_text = "Can you please close the <mask>?"
input_ids, mask_index = prepare_input(input_text)

# Debug prints
print(f"Input IDs shape (batch, seq_len): {input_ids.shape}")
print(f"Sequence length: {input_ids.size(1)}")
print(f"Mask index: {mask_index}")

# Get predictions using the revised approach
result = get_predictions(model, input_ids, mask_index)
if result is None:
    print("No predictions could be made.")
    exit()
top_indices, top_probs = result

# Display results
print(f"\nInput: {input_text}")
print("Top predictions for masked token:")
for idx, prob in zip(top_indices, top_probs):
    token = tokenizer.id_to_token(int(idx))
    print(f"{token}: {prob.item():.4f}")


print("\n--- HuggingFace Model Output ---")

hf_tokenizer = AutoTokenizer.from_pretrained("ltg/gpt-bert-babylm-small", trust_remote_code=True)
hf_model = AutoModelForMaskedLM.from_pretrained("ltg/gpt-bert-babylm-small", trust_remote_code=True)
hf_model.eval()

hf_encoded = hf_tokenizer(input_text, return_tensors="pt")
hf_mask_index = (hf_encoded.input_ids == hf_tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()

with torch.no_grad():
    hf_outputs = hf_model(**hf_encoded)
    hf_logits = hf_outputs.logits[0, hf_mask_index]
    hf_top_k = torch.topk(hf_logits, 10)

print("Input:", input_text)
print("Top predictions for masked token:")
for score, token_id in zip(hf_top_k.values, hf_top_k.indices):
    token = hf_tokenizer.decode(token_id)
    prob = torch.softmax(hf_logits, dim=0)[token_id].item()
    print(f"{token.strip()}: {prob:.4f}")