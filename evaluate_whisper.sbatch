#!/bin/bash
#SBATCH --job-name=evaluate-whisper_only_laughing_words          # Job name
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


# EVALUATE WHISPER SMALL WITH -----------------------
# token_speechlaugh alignment
# i.e. [laughter-word] -> [SPEECH_LAUGH]
#------------------------------------------------------
python evaluate_whisper.py \
    --source_dataset_path ./datasets/switchboard/token_speechlaugh/switchboard_dataset\
    --target_dataset_path ./datasets/switchboard/word_speechlaugh/word_speechlaugh/switchboard_dataset\
    --model_name openai/whisper-small \
    --output_file ./alignment_transcripts/whisper_small_only_laughing_tokens_summary.txt \
#------------------------------------------------------



# EVALUATE WHISPER SMALL WITH -----------------------
# word_speechlaugh alignment
# i.e. [laughter-word] -> WORD
#------------------------------------------------------
# python evaluate_whisper.py \
#     --dataset_path ./datasets/switchboard/word_speechlaugh/word_speechlaugh/switchboard_dataset\
#     --model_name openai/whisper-small \
#     --output_file ./alignment_transcripts/whisper_small_word_speechlaugh.txt \
#------------------------------------------------------