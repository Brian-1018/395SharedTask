import torch
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
import os

# === PATHS ===
tokenizer_path = "/home/dl5ja/shared_cfinegan/group4_work/395SharedTask/tokenizers/tokenizer_10M.json"
model_state_path = "/home/dl5ja/shared_cfinegan/group4_work/395SharedTask/checkpoints/run_004_4_11_2025_babylm_10M_state_dict.bin"
output_dir = "/home/dl5ja/shared_cfinegan/group4_work/395SharedTask/hf_format_models/run_004_4_11_2025_babylm_10M"

# === STEP 1: Ensure output directory exists ===
os.makedirs(output_dir, exist_ok=True)

# === STEP 2: Build config to match training ===
config = BertConfig(
    vocab_size=8197,
    hidden_size=256,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=1024,
    max_position_embeddings=512,
    type_vocab_size=1
)

# === STEP 3: Load model and weights ===
print("Loading state dict from:", model_state_path)
model = BertForMaskedLM(config)
checkpoint = torch.load(model_state_path, map_location="cpu")
state_dict = checkpoint["model"]  # or checkpoint["ema_model"] if you want EMA weights
model.load_state_dict(state_dict, strict=False)
print("Saving HuggingFace model to:", output_dir)
model.save_pretrained(output_dir)

# === STEP 4: Load tokenizer and wrap ===
print("Loading tokenizer from:", tokenizer_path)
tokenizer = Tokenizer.from_file(tokenizer_path)
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

# === STEP 5: Define special tokens (matching tokenizer JSON and BabyLM expectations) ===
hf_tokenizer.add_special_tokens({
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "mask_token": "<mask>"
})

print("Saving HuggingFace tokenizer to:", output_dir)
hf_tokenizer.save_pretrained(output_dir)

print("✅ DONE: Your model and tokenizer are now HuggingFace-compatible!")