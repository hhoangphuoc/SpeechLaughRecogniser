#!/bin/bash
#SBATCH --job-name=eval_swb_test-wav2vec2-finetuned                  # Job name
#SBATCH -c 8                                                   # Number of cores
#SBATCH --mem=12G                                                # Request 12GB memory (8GB swb_test + 4GB for the model)
#SBATCH --gres=gpu:ampere:1                                     # Request 1 GPU
#SBATCH --time=24:00:00                                          # Set a walltime limit
#SBATCH --mail-type=BEGIN,END,FAIL                              # Email status changes
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl             # Your email address

# Load modules (adjust versions as needed)
module purge
module load nvidia/cuda-11.8
module load nvidia/nvtop


# Print some useful information
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)" # log hostname
echo "Working Directory = $(pwd)"
echo "Name of nodes used          : "$SLURM_JOB_NODELIST
echo "Gpu devices                 : "$CUDA_VISIBLE_DEVICES
echo "Starting worker: "

# Activate your environment (if applicable)
source activate .venv


#===================================================================================================
#                                   EVALUATE THE DATASET
#===================================================================================================

# WHISPER
# python evaluate.py \
#     --dataset_dir ../datasets/switchboard/whisper/swb_test \
#     --model_name openai/whisper-large-v2 \
#     --model_type whisper \
#     --pretrained_model_dir ../ref_models/pre_trained \
#     --output_dir ../alignment_transcripts/whisper \

# ==========================================================================================

# # WAV2VEC2
# python evaluate.py \
#     --dataset_dir ../datasets/switchboard/whisper/swb_test \
#     --model_name facebook/wav2vec2-large-960h-lv60 \
#     --model_type wav2vec2 \
#     --pretrained_model_dir ../ref_models/pre_trained \
#     --output_dir ../alignment_transcripts/wav2vec2-new \

# ==========================================================================================

# Fine-tuned Wav2Vec2
# python evaluate.py \
#     --dataset_dir ../datasets/switchboard/whisper/swb_test \
#     --model_name  finetuned-wav2vec2-checkpoint-20500\
#     --model_type wav2vec2 \
#     --pretrained_model_dir ../checkpoints/wav2vec2-batch32 \
#     --output_dir ../alignment_transcripts/finetuned_wav2vec2 \

#===================================================================================================

# Fine-tuned Whisper
python evaluate.py \
    --dataset_dir ../datasets/switchboard/whisper/swb_test \
    --model_name  finetuned-whisper-checkpoint-8000\
    --model_type whisper \
    --pretrained_model_dir ../checkpoints/whisper-batch4-8000steps \
    --output_dir ../alignment_transcripts/finetuned_whisper/whisper_batch4-8000steps \
    
#----------------------------------------+
# TO PRODUCE ALIGNMENT PLOTS
#----------------------------------------+
# python align_dtw.py