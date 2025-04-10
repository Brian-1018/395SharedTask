<h2 align="center"><b><h3>GPT-BERT with Modified Pre-Training and Curriculum Learning</h3></b></h2>
<br>

<p align="center">
  <b>Authors: Ethan Rodgers-Gates, Brian Ma, Damien Liebov</b>
</p>

<p align="center">
  <i>CMSC395: Natural Language Processing, University of Richmond</i>
</p>
<br>

---

## Overview

This repository contains the codebase and documentation for our project. Our model builds upon GPT-BERT (BabyLM 10M) and is pretrained on the Baby-cosmo-fine-10M dataset. It unifies causal and masked language modeling in a single transformer, allowing it to be used seamlessly as either a generative (causal) or a bidirectional (masked) model.

## Model Information
- Pretrained Model Name: ltg/gpt-bert-babylm-small
- Hugging Face URL: https://huggingface.co/ltg/gpt-bert-babylm-small
- Baseline Implementation: Cloned from ltgoslo/gpt-bert and extended with our modifications.


---

## Planned Modifications

1. **Curriculum Learning Integration:**  
  We are implementing two pipelines for curriculum learning:
    - Pseudo-labeling using SpaCy to compute syntactic complexity
    - Readability Scores using textstat to compute FKGL scores.  
    The selected method will be integrated into the training loop via a curriculum scheduler that feeds data in progressively increasing difficulty.

2. **Experimental Evaluation:**  
  We will run experiments using the strict-small dataset and evaluate performance improvements on BLiMP, GLUE, and EWOK benchmarks.

3. **Documentation and Reproducibility:**  
  The repository will include instructions for environment setup, data preprocessing, and running experiments, ensuring that our work is fully reproducible.

_______

## Setup

**Set Up Your Environment**
> python -m venv venv \
source venv/bin/activate \
pip install -r requirements.txt

**Pretraining the Baseline Model**
> python pretraining/train_baseline.py --config configs/baseline_config.json

This script uses our modified pre-training methods on the Baby-cosmo-fine-10M dataset.

**Running the Evaluation Pipeline**
Once the model is pretrained, run the evaluation pipeline via:
> sbatch group4.slurm

**Test Case**
```
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("ltg/gpt-bert-babylm-small")
model = AutoModelForCausalLM.from_pretrained("ltg/gpt-bert-babylm-small")

# Example text for tokenization and generation
text = "The quick brown fox jumps over the lazy dog."

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Generate text output from the model
outputs = model.generate(inputs["input_ids"], max_length=20)

# Decode the generated tokens and print the output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Download data**

[Click here to download dataset](https://osf.io/ad7qg/) and save the dev, text, train_10M and train_100M to the `data/raw` folder

## Configure

Configure the following config parameters in conf/config.yaml : 

| Config Param | Values | Description |
|--------------|--------|-------------|
| general.exp_name | Any | Name of the experiment where trained tokenizers, model configs, trained model checkpoints will be saved |
| general.wandb_log | True/False | Flag to enable/disable wandb logging |
| general.wandb_project | Any | Name of wandb project where the training metrics will be logged |
| general.wandb_run_name | Any | Name of wandb run within the project. Datetime will be appended to this to ensure uniqueness |
| preprocess.tokenizer_type | from_scratch/pretrained/pretrained_hf | Whether to train a tokenizer from scratch or use pretrained one |
| preprocess.train_data_path | Any | Path to preprocessed training data with .train extension |
| preprocess.dev_data_path | Any | Path to preprocessed dev data with .dev extension |
| WML.distillation_method | mutual/vanilla | Select either mutual (without teacher) or vanilla (with teacher supervision) distillation |
| WML.hf_model_name | Any | Name of huggingface model used as teacher model when distillation_method = vanilla | 
| WML.use_opt_config | True/False | This enable/disables architecture search to find peer models. If you have already configs saved in models/exp_name/dataset_name/arch_search_results, then you can set to False |
| WML.model_type | MLM/CLM | MLM (Masked language models eg. RoBERTa) or CLM (Causal language models eg. GPT2) |
| WML.num_peers | 1/2/4 | Works best for num_peer = 4 | 

<!-- **Train custom tokenizer**
```
$ python scripts/run_tokenizer.py
```

**Run Weighted deep mutual learning (WDML) training script for num_peers = 4**
```
$ python scripts/train_WML.py
``` -->

## Citation
<!-- Informal Version:

[When Babies Teach Babies: Can student knowledge sharing outperform Teacher-Guided Distillation on small datasets?](https://aclanthology.org/2024.conll-babylm.17/) (Iyer, CoNLL-BabyLM 2024) -->

<!-- ACM Version: -->

Lucas Georges Gabriel Charpentier and David Samuel. 2024. [BERT or GPT: why not both?](https://aclanthology.org/2024.conll-babylm.24/). In *The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning*, pages 262–283, Miami, FL, USA. Association for Computational Linguistics.
