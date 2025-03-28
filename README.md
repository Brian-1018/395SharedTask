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

This repository contains the codebase and documentation for our project:
**GPT-BERT with Modified Pre-Training and Curriculum Learning.**  
Building upon the GPT-BERT combined model (Charpentier et al., 2024), our work integrates curriculum learning via two preprocessing methods pseudo-labeling with a dependency parser and readability scoring using the FKGL formula. Our goal is to enhance sample efficiency and improve performance across syntactic, reasoning, and world knowledge tasks.

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

**Virtual Environment manager** <br>
This project uses mamba as the package and virtual environment manager. To install mamba, here are some quick steps to doing so:

For macOS/Linux : 
```bash
$ curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" 

$ bash Miniforge3-$(uname)-$(uname -m).sh
```

For Windows :<br>
refer [here](https://github.com/conda-forge/miniforge?tab=readme-ov-file#windows)

**Create environment and install dependencies**

For macOS (Apple Chip):
```bash
$ mamba env create -f dependencies/babylm-conda-metal.yaml
```

For Linux :
```bash
$ mamba env create -f dependencies/babylm-conda.yaml
```

**Download data**

[Click here to download dataset](https://osf.io/ad7qg/) and save the dev, text, train_10M and train_100M to the `data/raw` folder

## Data pre-processing

Assuming you're already on root folder

Run preprocessing shell script:
```
$ bash ./scripts/preprocess.sh
```
This should create and save preprocesed training, test and dev babylm datasets a folder under data called ```preprocessed```. 

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

**Train custom tokenizer**
```
$ python scripts/run_tokenizer.py
```

**Run Weighted deep mutual learning (WDML) training script for num_peers = 4**
```
$ python scripts/train_WML.py
```

## Citation
<!-- Informal Version:

[When Babies Teach Babies: Can student knowledge sharing outperform Teacher-Guided Distillation on small datasets?](https://aclanthology.org/2024.conll-babylm.17/) (Iyer, CoNLL-BabyLM 2024) -->

<!-- ACM Version: -->

Srikrishna Iyer. 2024. [When Babies Teach Babies: Can student knowledge sharing outperform Teacher-Guided Distillation on small datasets?](https://aclanthology.org/2024.conll-babylm.17/). In *The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning*, pages 197–211, Miami, FL, USA. Association for Computational Linguistics.