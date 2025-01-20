#!/bin/bash
#SBATCH --job-name=preprocess_swb_speech         # Job name
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

#===============================================================================================================#
#                   PREPROCESS FOR SPECIFIC SPEECH EVENTS TYPE IN SWITCHBOARD DATASET
#===============================================================================================================#

#----------------
# SPEECHLAUGH
#----------------
# python data_preprocess.py \
#     --data_names switchboard \
#     --dataset_dir ./datasets/switchboard/ \
#     --to_dataset True \
#     --to_csv True \
#     --retokenize_type speechlaugh \
#     --audio_segment_name swb_speechlaugh \
    # --push_to_hf True \
#---------------------------------------------#
#----------------
#     LAUGH
#----------------
# python data_preprocess.py \
#     --data_names switchboard \
#     --dataset_dir ./datasets/switchboard/ \
#     --to_dataset True \
#     --to_csv True \
#     --retokenize_type laugh \
#     --audio_segment_name swb_laugh \
#     # --push_to_hf True \

#----------------
# LAUGH-INTEXT
#----------------
# python data_preprocess.py \
#     --data_names switchboard \
#     --dataset_dir ./datasets/switchboard/ \
#     --to_dataset True \
#     --to_csv True \
#     --retokenize_type laugh\
#     --audio_segment_name swb_laugh_intext \

#---------------------------------------------#
#----------------
# SPEECH-ONLY
#----------------
# python data_preprocess.py \
#     --data_names switchboard \
#     --dataset_dir ./datasets/switchboard/ \
#     --to_dataset True \
#     --to_csv True \
#     --retokenize_type speech \
#     --audio_segment_name swb_speech \
#---------------------------------------------#

#===============================================================================================================#
#                               PREPROCESS FULL SWITCHBOARD DATASET
#===============================================================================================================#
# python data_preprocess.py \
#     --data_names switchboard \
#     --dataset_dir ./datasets/switchboard/ \
#     --to_dataset True \
#     --to_csv True \
#     --audio_segment_name swb_all \

#===============================================================================================================#
#                               PREPROCESS BUCKEYE DATASET
#===============================================================================================================#
python data_preprocess.py \
    --data_names buckeye \
    --dataset_dir ./datasets/buckeye/ \
    --to_dataset True \
    --to_csv True \


