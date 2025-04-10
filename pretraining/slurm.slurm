#!/bin/bash -e

# This is the name that will show when jobs are listed.
# You can name your jobs however you like; it is a good
# idea to make the name recognizeable in the first few 
# characters.
#SBATCH --job-name=GROUP4_EVAL

# This is usually the right value -- most of the time
# you are running one program at a time no matter how
# many cores you are using.
#SBATCH --ntasks=1

# This is ten seconds not ten minutes. Adjust accordingly.
#SBATCH --time=8-15:00:30
# Change /netid/ to *your* netid.
#SBATCH --mail-user=damien.liebov@richmond.edu
#SBATCH --mail-type=ALL
#####
#BEGIN — send mail when the job begins.
#END — send mail when the job ends.
#FAIL — send mail if the job fails.
#REQUEUE — send mail if the job is requeued.
#ALL — send mail for all events (i.e., BEGIN, END, FAIL, REQUEUE).
#STAGE_OUT — send mail when stage-out is completed (for advanced I/O setups).
#TIME_LIMIT — send mail when the job approaches its time limit.
#TIME_LIMIT_90 — job has used 90% of its time limit.
#TIME_LIMIT_80 — job has used 80% of its time limit.
#TIME_LIMIT_50 — job has used 50% of its time limit.
#ARRAY_TASKS — send separate mail for each array task (instead of aggregate).
#####


###
# This statement requests the use of a GPU. The type of GPU
# is not required if there is only one type on the node.
# The final ":1" says, "I want one GPU."
###
# SBATCH --gres=gpu:tesla_a40:2

# basic is the default collection of compute nodes. They
# each have 52 cores and 384GB of memory.
#SBATCH --partition=NLP

# Memory requests are in megabytes by default. This is 24 GB.
#SBATCH --mem=128000

# This figure means cores not CPUs. 
# SBATCH --cpus-per-task=7

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=1
export RANK=0
export WANDB_MODE=disabled

# Print the start date/time
date

# This step makes sure you have a directory on the /scratch
# mount point. Be sure to change netid to your netid.
mkdir -p /scratch/dl5ja

# Print the node your job is running on
echo "I ran on:"
echo "SLURM_NODELIST=$SLURM_NODELIST"

# Return the context/PWD to the directory where *this* file is located.
cd $SLURM_SUBMIT_DIR

# Set any environment variables like PATH, LD_LIBRARY_PATH, etc.
export HF_HOME=/scratch/cfinegan/hf_cache

echo "cd to target directory"
cd /home/dl5ja/shared_cfinegan/evaluation-pipeline-2024/

# Load the necessary program dependencies.
echo "setting up conda"
conda init bash

echo "checking pip installs"
pip freeze | grep minicons



# Run jobs. /usr/bin/time -v will print a number of useful
# diagnostics that will help us understand how the cluster
# is being used. Sleep is a program that does nothing, in this
# case for 60 seconds. Your program probably does a bit more.

# echo "Running python script"
# python /home/dl5ja/shared_cfinegan/group4_work/group4.slurm

# Code to create the tokenizer:
# cd /home/dl5ja/shared_cfinegan/group4_work/395SharedTask/tokenizer_creation

# python tokenizer_10m.py
#  # ERROR IN THIS FILE, NEEDED TO ADD ["TEXT"] TO .strip COMMAND

# echo "Tokenization complete, moving to next step..."

# Code to tokenize the dataset:
# cd /home/dl5ja/shared_cfinegan/group4_work/395SharedTask/corpus_tokenization

# python tokenize_corpus.py \
#     --train_file="babycosmofine_10M.jsonl" \
#     --tokenizer_file="tokenizer_10M.json" 
# # ERROR IN THIS FILE, NEEDED TO FIX LINE 26 text = text["text"].strip()


# python tokenize_corpus_split.py \
#     --train_file="babycosmofine_10M.jsonl" \
#     --tokenizer_file="tokenizer_10M.json" 

# echo "Tokenization of data complete, moving to next step..."

cd /home/dl5ja/shared_cfinegan/group4_work/395SharedTask/pretraining

python train_10m.py \
    --train_path="../data/babycosmofine_10M_split_train_tokenized.bin" \
    --valid_path="../data/babycosmofine_10M_valid_tokenized.bin" \
    --name="run_001_4_10_2025_babylm_10M" \
    --hybrid_denominator=1

# cd /home/dl5ja/shared_cfinegan/group4_work/395SharedTask/pretraining

# python train_multi_gpu.py \
#     --train_path="../data/train_10M_tokenized.bin" \
#     --valid_path="../data/dev_10M_tokenized.bin" \
#     --config_file="../configs/base.json" \
#     --tokenizer_path="../tokenizers/tokenizer_10M.json" \
#     --output_dir="../model_checkpoints" \
#     # # WandB information + model name
#     # --name="RUN_NAME" \
#     # --wandb_project="YOUR_WANDB_PROJECT_NAME" \
#     # --wandb_entity="YOUR_WANDB_ENTITY" \
#     # Training Configuration
#     --hybrid_numerator=15 \
#     --hybrid_denominator=16 \
#     --seq_length=128 \
#     --local_batch_size=128 \
#     --global_batch_size=32768 \
#     --learning_rate=1e-2 \
#     --max_steps=15625 \
#     # Optimizer & Scheduler
#     --optimizer="lamb" \
#     --weight_decay=0.1 \
#     --warmup_proportion=0.016 \
#     --cooldown_proportion=0.016 \
#     # Masking Configuration
#     --mask_p_start=0.3 \
#     --mask_p_end=0.15 \
#     --mask_random_p=0.1 \
#     --mask_keep_p=0.1 \
#     # Training Control
#     --mixed_precision \
#     --validate_every=1000 \
#     --save_every=1000 \
#     --seed=42

# Print the ending date/time
date




# #!/bin/bash

# #SBATCH --job-name=GRP4_TRAIN
# #SBATCH --mail-user=damien.liebov@richmond.edu
# #SBATCH --mail-type=ALL
# #SBATCH --time=20:00:00
# #SBATCH --cpus-per-task=7
# #SBATCH --mem=96000
# #SBATCH --ntasks=1
# #SBATCH --gres=gpu:tesla_a40:2
# #SBATCH --partition=NLP
# #SBATCH --output=bert-%j.out

# #SBATCH --mail-user=damien.liebov@richmond.edu
# #SBATCH --mail-type=ALL

# # Print the start date/time
# date

# # Create scratch directory
# mkdir -p /scratch/dl5ja

# # Print node information
# echo "I ran on:"
# echo "SLURM_NODELIST=$SLURM_JOB_NODELIST"

# # Change to submission directory
# cd $SLURM_SUBMIT_DIR


# # This is an example script for running the BERT model pretraining on a single node with 8 GPUs.
# # You'll most likely have to adjust the script to match your setup.

# # distributed setup
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_PORT=9999
# export WORLD_SIZE=$SLURM_NTASKS
# export HF_HOME=/scratch/cfinegan/hf_cache

# CONTAINER=""
# SING_BIND="SINGULARITY_BINDING"

# set -euo pipefail

# CMD="train_multi_gpu.py"

# echo $CMD
# echo "START $SLURM_JOBID: $(date)"

# srun \
#     --label \
#     singularity exec \
#     -B "$SING_BIND" \
#     "$CONTAINER" \
#    launch.sh \
#     $CMD

# echo "END $SLURM_JOBID: $(date)"
