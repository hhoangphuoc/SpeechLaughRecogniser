#!/bin/bash
#SBATCH --job-name=align_hit_miss_short_padded_segments          # Job name
#SBATCH -c 16                                                   # Number of cores
#SBATCH --mem=8G                                                # Request 8GB memory
#SBATCH --gres=gpu:ampere:1                                     # Request 1 GPU
#SBATCH --time=5:00:00                                          # Set a walltime limit
#SBATCH --mail-type=BEGIN,END,FAIL                              # Email status changes
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl             # Your email address

# Load modules (adjust versions as needed)
module purge # clean the environment before loading new modules
module load nvidia/cuda-11.7
module load nvidia/cuda-11.x_cudnn-8.6
module load nvidia/cuda-11.x_tensorrt-8.6
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



#=============================================
# TRY WITH SHORT_PADDED_SWITCHBOARD
#=============================================
# python evaluate_whisper.py \
#     --source_dataset_path ../datasets/short_padded_switchboard/token_speechlaugh/switchboard_dataset\
#     --target_dataset_path ../datasets/short_padded_switchboard/word_speechlaugh/switchboard_dataset\
#     --model_name openai/whisper-small \
#     --pretrained_model_dir ../ref_models/pre_trained \
#     --output_file ../alignment_transcripts/alignment_laughing_word_hit_miss_short_padded.txt \
#=======================================================================================================



#=========================================
# TO PRODUCE ALIGNMENT PLOTS
#========================================+
python align_dtw.py