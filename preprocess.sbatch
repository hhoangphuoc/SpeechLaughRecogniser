#!/bin/bash
#SBATCH --job-name=preprocess_swb_token_speechlaugh         # Job name
#SBATCH -c 16                                               # Number of cores
#SBATCH --mem=6G                                            # Request 6GB memory
#SBATCH --gres=gpu:ampere:1                                 # Request 1 GPU
#SBATCH --time=5:00:00                                      # Set a walltime limit
#SBATCH --mail-type=BEGIN,END,FAIL                          # Email status changes
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl  # Your email address

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

# Run your script with the appropriate arguments

#---------------------------------------------#
# PREPROCESS WITH SPEECH_LAUGH TOKENIZATION
#---------------------------------------------#
# for token [SPEECH_LAUGH]
# python preprocess.py \
#     --data_names switchboard \
#     --csv_dir ./datasets/switchboard/ \
#     --to_dataset True \
#     --to_csv True \
#     --tokenize_speechlaugh True \
#---------------------------------------------#

#---------------------------------------------#
# PREPROCESS WITH LAUGHING WORD TOKENIZATION
#---------------------------------------------#
# for token 'word' in [laughter-word]
python preprocess.py \
    --data_names switchboard \
    --csv_dir ./datasets/switchboard/word_speechlaugh \
    --to_dataset True \
    --to_csv True \