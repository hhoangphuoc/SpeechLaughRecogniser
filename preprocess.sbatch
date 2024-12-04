#!/bin/bash
#SBATCH --job-name=preprocess_swb_laugh_intext         # Job name
#SBATCH -c 16                                               # Number of cores
#SBATCH --mem=10G                                            # Request 10GB memory
#SBATCH --gres=gpu:ampere:1                                 # Request 1 GPU
#SBATCH --time=10:00:00                                      # Set a walltime limit
#SBATCH --mail-type=BEGIN,END,FAIL                          # Email status changes
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl  # Your email address

# Load modules (adjust versions as needed)
module purge # clean the environment before loading new modules
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

# Run your script with the appropriate arguments

#---------------------------------------------#
# PREPROCESS WITH SPEECH_LAUGH TOKENIZATION
#---------------------------------------------#
# for token [SPEECH_LAUGH]
# python preprocess.py \
#     --data_names switchboard \
#     --csv_dir ./datasets/short_padded_switchboard/ \
#     --to_dataset True \
#     --to_csv True \
#     --tokenize_speechlaugh True \
#---------------------------------------------#

#---------------------------------------------#
#                   PREPROCESS
#---------------------------------------------#
# for special token for speech-laugh: WORD
# python data_preprocess.py \
#     --data_names switchboard \
#     --dataset_dir ./datasets/switchboard/ \
#     --to_dataset True \
#     --to_csv True \
#     --retokenize_type speechlaugh \
#     --audio_segment_name swb_speechlaugh \
    # --push_to_hf True \
#---------------------------------------------#
# for special token for laugh: [LAUGH]
# python data_preprocess.py \
#     --data_names switchboard \
#     --dataset_dir ./datasets/switchboard/ \
#     --to_dataset True \
#     --to_csv True \
#     --retokenize_type laugh \
#     --audio_segment_name swb_laugh \
#     # --push_to_hf True \

# For [LAUGH] in text
python data_preprocess.py \
    --data_names switchboard \
    --dataset_dir ./datasets/switchboard/ \
    --to_dataset True \
    --to_csv True \
    --retokenize_type laugh\
    --audio_segment_name swb_laugh_intext \

#---------------------------------------------#
# for normal speech: no special token
# python data_preprocess.py \
#     --data_names switchboard \
#     --dataset_dir ./datasets/switchboard/ \
#     --to_dataset True \
#     --to_csv True \
#     --retokenize_type speech \
#     --audio_segment_name swb_speech \
    # --push_to_hf True \
#---------------------------------------------#
