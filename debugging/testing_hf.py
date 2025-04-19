import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def main():
    # Path to your HuggingFace model directory
    model_dir = "/home/dl5ja/shared_cfinegan/group4_work/395SharedTask/hf_format_models/run_004_4_11_2025_babylm_10M"
    
    print(f"Loading model and tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True)

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model embedding size: {model.get_input_embeddings().weight.shape[0]}")

    print("HF tokenizer vocab size:", tokenizer.vocab_size)
    print("HF tokenizer max token ID:", max(tokenizer.get_vocab().values()))

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Ensure mask token is set
    if tokenizer.mask_token is None:
        raise ValueError("The tokenizer does not have a mask token set.")
    print(f"Using mask token: {tokenizer.mask_token}")

    # Replace <mask> in your input with the actual mask token
    test_sentence = "Can you please close the <mask>?"
    if "<mask>" not in test_sentence:
        raise ValueError("The test sentence must contain '<mask>' token.")
    
    test_sentence = test_sentence.replace("<mask>", tokenizer.mask_token)

    # Tokenize and move tensors to device
    inputs = tokenizer(test_sentence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    mask_token_id = tokenizer.mask_token_id
    mask_index = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[1].item()
    print(f"Mask token index: {mask_index}")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0, mask_index]
    probs = torch.softmax(logits, dim=-1)
    top_k = 5
    top_probs, top_indices = torch.topk(probs, k=top_k)
    
    print(f"\nInput: {test_sentence}")
    print("Top predictions for the masked token:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token = tokenizer.decode([idx])
        print(f"{i+1}. '{token.strip()}': prob = {prob.item():.4f}")

if __name__ == "__main__":
    main()