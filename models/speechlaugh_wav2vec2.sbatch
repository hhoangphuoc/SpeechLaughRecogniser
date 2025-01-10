#!/bin/bash
#SBATCH --job-name=speechlaugh-wav2vec2-large-lv60-batch64     # Job name
#SBATCH -c 8                                      # Number of cores
#SBATCH --gres=gpu:ampere:1                        # Request 1 GPU
#SBATCH --constraint=a40
#SBATCH --mem=64G                                   # Request 64GB of memory (8GB whisper-large, 8GB optimizer, 4GB batch, 4GB overhead)
#SBATCH --time=120:00:00                             # Set a walltime limit
#SBATCH --mail-type=BEGIN,END,FAIL                  # Email status changes
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl  # Your email address

# Load modules (adjust versions as needed)
module purge # clean the environment before loading new modules
module load nvidia/cuda-11.8
module load nvidia/nvtop

export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"

#============================================================
#                       PRINT INFO
#============================================================

# Print some useful information
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)" # log hostname
echo "Working Directory = $(pwd)"
echo "Name of nodes used          : "$SLURM_JOB_NODELIST
echo "Gpu devices                 : "$CUDA_VISIBLE_DEVICES
echo "Starting worker: "

echo "Number of CPU cores = $(nproc)"
echo "SLURM_CPUS_PER_TASK:          "$SLURM_CPUS_PER_TASK

# Activate your environment (if applicable)
source activate .venv

# Run your script with the appropriate arguments
python SpeechLaughWav2Vec2.py 


