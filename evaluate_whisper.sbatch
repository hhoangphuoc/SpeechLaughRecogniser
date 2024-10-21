#!/bin/bash
#SBATCH --job-name=evaluate-whisper      # Job name
#SBATCH -c 16                     # Number of cores
#SBATCH --gres=gpu:ampere:1                 # Request 1 GPU
#SBATCH --time=24:00:00              # Set a walltime limit
#SBATCH --mail-type=BEGIN,END,FAIL    # Email status changes
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

# Evaluate whisper-small
python evaluate_whisper.py \
    --csv_file ./datasets/combined/val_switchboard.csv \n
    --model_path openai/whisper-small \n
    --output_file ./alignment_transcripts/alignment_whisper_small \n

# Evaluate whisper-large
python evaluate_whisper.py \
    --csv_file ./datasets/combined/val_switchboard.csv \n
    --model_path openai/whisper-large \n
    --output_file ./alignment_transcripts/alignment_whisper_large \n

