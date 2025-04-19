#!/bin/bash -e

###############################################################################
#                              SLURM Job Options                             
###############################################################################
#SBATCH --job-name=GROUP4_RUN               # Job name
#SBATCH --ntasks=2                          # Number of tasks (processes)
#SBATCH --cpus-per-task=11                 # Cores per task
#SBATCH --gres=gpu:tesla_a40:2             # GPUs per node
#SBATCH --partition=NLP                    # Partition
#SBATCH --mem=128000                       # Memory in MB (128 GB)
#SBATCH --time=8-15:00:30                  # Time limit (days-hours:minutes:seconds)
#SBATCH --mail-user=damien.liebov@richmond.edu
#SBATCH --mail-type=ALL                   # Mail notifications for all events
###############################################################################
#                             Environment Setup                               
###############################################################################
# Master settings for distributed training
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=9998
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# Disable Weights & Biases logging
export WANDB_MODE=disabled

# Hugging Face cache directory
export HF_HOME=/scratch/cfinegan/hf_cache

###############################################################################
#                              Job Initialization                             
###############################################################################
# Print start time
date


# Display compute nodes
echo "Running on:"
echo "  SLURM_NODELIST=$SLURM_NODELIST"

# Return to submission directory
cd $SLURM_SUBMIT_DIR

###############################################################################
#                         Change to Project Directory                          
###############################################################################
echo "Changing to project directory..."
cd /home/dl5ja/shared_cfinegan/group4_work/395SharedTask

###############################################################################
#                         Conda & Dependency Check                           
###############################################################################
echo "Initializing Conda..."
conda init bash

echo "Verifying pip packages..."
pip freeze | grep minicons

###############################################################################
#                           STEP 1: TOKENIZER CREATION                        
###############################################################################
# Uncomment to run tokenizer creation:

# cd tokenizer_creation
# python tokenizer_10m.py
# echo "Tokenization complete, moving to next step..."

###############################################################################
#                        STEP 2: CORPUS TOKENIZATION                          
###############################################################################
# Uncomment to tokenize the dataset:

# cd corpus_tokenization
# python tokenize_corpus_split.py \
#     --train_file="babycosmofine_10M.jsonl" \
#     --tokenizer_file="tokenizer_10M.json"
# echo "Tokenization of data complete, moving to next step..."

###############################################################################
#                         STEP 3: MODEL TRAINING                              
###############################################################################
# Uncomment to train the model:

# cd pretraining
# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128 # Helps with CUDA memory fragmentation
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Helps with CUDA memory fragmentation
# srun python train_10m_modified.py \
#     --train_path="../data/babycosmofine_10M_split_train_tokenized.bin" \
#     --valid_path="../data/babycosmofine_10M_valid_tokenized.bin" \
#     --name="modified_03_babylm_10M" \
#     --output_dir="../checkpoints" \
#     --hybrid_denominator=1 \
#     --save_every=15 \
#     --local_batch_size=128 \
#     --seq_length=96

# # --checkpoint_filename="../checkpoints/{CHECKPOINT-NAME}_state_dict].bin" \ # Optional checkpoint filename
# echo "Model training complete!"

###############################################################################
#                       STEP 4: CONVERT TO HF FORMAT                           
###############################################################################
# Uncomment to run conversion to HF format:

# cd hf_format_models
# python convert_model_to_hf.py \
#     --model_state_path="../checkpoints/modified_halfway_state_dict.bin" \
#     --output_dir="./modified_03_babylm_10M" \
#     --tokenizer_path="../tokenizers/tokenizer_10M.json"
# echo "Conversion to HF format complete!"

###############################################################################
#                       STEP 5: BLIMP EVALUATION                              
###############################################################################
# Uncomment to run BLiMP evaluation:

# echo "Starting BLiMP evaluation..."
# cd evaluation-pipeline-2024
# export CUDA_LAUNCH_BLOCKING=1
# export TF_CPP_MIN_LOG_LEVEL=2

# ./eval_blimp.sh ../hf_format_models/modified_03_babylm_10M

###############################################################################
#                       STEP 6: EWOK EVALUATION                               
###############################################################################
# Uncomment to run EWOK evaluation:

# echo "Starting EWOK evaluation..."
# cd evaluation-pipeline-2024
# export CUDA_LAUNCH_BLOCKING=1
# export TF_CPP_MIN_LOG_LEVEL=2

# ./eval_ewok.sh ../hf_format_models/modified_03_babylm_10M

###############################################################################
#                             Job Completion                                  
###############################################################################
# Print end time
date