import argparse
import os
import torch
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer

# === COMMAND LINE ARGUMENTS ===
parser = argparse.ArgumentParser(
    description="Convert BabyLM model and tokenizer to HuggingFace format"
)
parser.add_argument(
    "--tokenizer_path",
    type=str,
    default="/home/dl5ja/shared_cfinegan/group4_work/395SharedTask/tokenizers/tokenizer_10M.json",
    help="Path to the BabyLM tokenizer JSON file"
)
parser.add_argument(
    "--model_state_path",
    type=str,
    default="/home/dl5ja/shared_cfinegan/group4_work/395SharedTask/checkpoints/modified_03_babylm_10M_state_dict.bin",
    help="Path to the BabyLM model state dict (.bin)"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/dl5ja/shared_cfinegan/group4_work/395SharedTask/hf_format_models/modified_03_babylm_10M",
    help="Directory where the HuggingFace model and tokenizer will be saved"
)
args = parser.parse_args()

# === PATHS ===
tokenizer_path = args.tokenizer_path
model_state_path = args.model_state_path
output_dir = args.output_dir

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
state_dict = checkpoint.get("model", checkpoint)
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