import logging
import torch
import os
import json
from types import SimpleNamespace
from tokenizers import Tokenizer
from model_extra import Bert

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

logger.info("Hello from generation script")

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load tokenizer
tokenizer = Tokenizer.from_file("../tokenizers/tokenizer_10M.json")
CLS_ID = tokenizer.token_to_id("<s>")
PAD_ID = tokenizer.token_to_id("<pad>")

# Load config
with open("../configs/small.json", "r") as f:
    config = SimpleNamespace(**json.load(f))

# Load model
model = Bert(config)
checkpoint = torch.load("../checkpoints/run_004_4_11_2025_babylm_10M_state_dict.bin", map_location="cpu")
state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# Generate text autoregressively
def generate_text(prompt, max_new_tokens=50):
    model_input = tokenizer.encode(prompt).ids
    input_ids = model_input + [CLS_ID]  # Append <s> for new segment

    for _ in range(max_new_tokens):
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask = torch.ones(len(input_ids), len(input_ids), dtype=torch.bool).to(device)
        attention_mask = ~torch.tril(attention_mask)  # causal mask

        with torch.no_grad():
            contextualized = model.get_contextualized(input_tensor.T, attention_mask.unsqueeze(0))
            logits = model.classifier(contextualized, torch.zeros_like(input_tensor.T), num_masked=0)[1]
            next_token_logits = logits[-1]

            from torch.nn import functional as F
            temperature = 0.8
            repetition_penalty = 1.2
            top_p = 0.9

            for token_id in set(input_ids):
                next_token_logits[token_id] /= repetition_penalty

            probs = F.softmax(next_token_logits / temperature, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs < top_p
            top_p_mask[1:] = top_p_mask[:-1].clone()
            top_p_mask[0] = True

            filtered_logits = sorted_probs * top_p_mask
            filtered_logits /= filtered_logits.sum()
            next_token = sorted_indices[torch.multinomial(filtered_logits, num_samples=1)].item()

            input_ids.append(next_token)

        if next_token == PAD_ID:
            break

    decoded = tokenizer.decode(input_ids)
    return decoded

# Example usage
prompt = "The BabyLM challenge will be"
logger.info(f"Generating from prompt: {prompt}")
generated = generate_text(prompt, max_new_tokens=50)
print("Generated Text:")
print(generated)



# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# import logging
# import torch
# import os
# import json
# from types import SimpleNamespace
# from model_extra import Bert
# from tokenizers import Tokenizer

# # Set cache dir
# os.environ['HF_HOME'] = '/scratch/cfinegan/hf_cache/'

# # Setup logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     level=logging.INFO,
# )
# logger.info("Hello from pipeline.py")

# # Check CUDA
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logger.info(f"CUDA available: {torch.cuda.is_available()}")

# # ============================
# # Hugging Face MODEL
# # ============================
# hf_tokenizer = AutoTokenizer.from_pretrained("ltg/gpt-bert-babylm-small", trust_remote_code=True)
# hf_pipe = pipeline(
#     "text-generation",
#     model="ltg/gpt-bert-babylm-small",
#     tokenizer=hf_tokenizer,
#     trust_remote_code=True,
#     use_cache=False,
#     truncation=True
# )
# logger.info("Generating text with HF model")
# hf_output = hf_pipe("The BabyLM Challenge will be", max_length=50, num_return_sequences=1, do_sample=True)
# logger.info(f"[HF] Generated: {hf_output[0]['generated_text']}")