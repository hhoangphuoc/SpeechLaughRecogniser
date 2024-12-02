#!/bin/bash
#SBATCH --job-name=align_swb_laugh_intext                              # Job name
#SBATCH -c 8                                                   # Number of cores
#SBATCH --mem=8G                                                # Request 8GB memory
#SBATCH --gres=gpu:ampere:1                                     # Request 1 GPU
#SBATCH --time=2:00:00                                          # Set a walltime limit
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


#==========================================================================
#               EVALUATE WHISPER SMALL WITH HIT-MISS RATE 
#========================================================================== 
# Using both token_speechlaugh and word_speechlaugh datasets
# and filter to SPEECH_LAUGH and LAUGHTER datasets
#
# python evaluate_whisper.py \
#     --source_dataset_path ../datasets/switchboard/token_speechlaugh/switchboard_dataset\
#     --target_dataset_path ../datasets/switchboard/word_speechlaugh/word_speechlaugh/switchboard_dataset\
#     --model_name openai/whisper-small \
#     --output_file ../alignment_transcripts/whisper_small_laughing_word_hit_miss_summary_fixed.txt \
# ==========================================================================================


#==========================================================================
#               EVALUATE WHISPER SMALL WITH WORD_SPEECHLAUGH
#==========================================================================
# word_speechlaugh alignment
# i.e. [laughter-word] -> WORD
#
# python evaluate_whisper.py \
#     --dataset_path ./datasets/switchboard/word_speechlaugh/word_speechlaugh/switchboard_dataset\
#     --model_name openai/whisper-small \
#     --output_file ./alignment_transcripts/whisper_small_word_speechlaugh.txt \
# ==========================================================================================



#===========================================================================================
#                       TRY WITH SHORT_PADDED_SWITCHBOARD 
#===========================================================================================
# - SPEECHLAUGH
# python evaluate_whisper.py \
#     --dataset_dir ../datasets/switchboard \
#     --dataset_type speechlaugh \
#     --model_name openai/whisper-small \
#     --pretrained_model_dir ../ref_models/pre_trained \
#     --output_dir ../alignment_transcripts \

# # - LAUGH
# python evaluate_whisper.py \
#     --dataset_dir ../datasets/switchboard \
#     --dataset_type laugh \
#     --model_name openai/whisper-small \
#     --pretrained_model_dir ../ref_models/pre_trained \
#     --output_dir ../alignment_transcripts \

# - In-text LAUGH
python evaluate_whisper.py \
    --dataset_dir ../datasets/switchboard \
    --dataset_type laugh_intext \
    --model_name openai/whisper-small \
    --pretrained_model_dir ../ref_models/pre_trained \
    --output_dir ../alignment_transcripts \

# - SPEECH
# python evaluate_whisper.py \
#     --dataset_dir ../datasets/switchboard \
#     --dataset_type speech \
#     --model_name openai/whisper-small \
#     --pretrained_model_dir ../ref_models/pre_trained \
#     --output_dir ../alignment_transcripts \

#=======================================================================================================



#=========================================
# TO PRODUCE ALIGNMENT PLOTS
#========================================+
# python align_dtw.py