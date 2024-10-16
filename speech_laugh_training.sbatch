#!/bin/bash
#SBATCH --job-name=speechlaugh-whisper      # Job name
#SBATCH -c 16                     # Number of cores
#SBATCH --gres=gpu:ampere:1                 # Request 1 GPU
#SBATCH --mem=24G                # Request 24GB of memory (8GB whisper-large, 8GB optimizer, 4GB batch, 4GB overhead)
#SBATCH --time=24:00:00              # Set a walltime limit
#SBATCH --mail-type=BEGIN,END,FAIL    # Email status changes
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl  # Your email address

# Load modules (adjust versions as needed)
module load microsoft/vscode/1.78.2
module load nvidia/cuda-11.8
module load nvidia/cuda-11.8_nccl-2.16
module load nvidia/cuda-11.x_cudnn-8.6
module load nvidia/cuda-11.x_tensorrt-8.6
module load nvidia/nvtop

#if we use laplace GPU instead
# module load nvidia/cuda-12.4
# module load nvidia/nvhpc-byo-compiler/23.3
# module load nvidia/nvhpc-hpcx/23.3
# module load nvidia/nvhpc-nompi/23.3

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
python SpeechLaughRecognition.py \
    --train_file_path ./datasets/train_switchboard.csv \
    --eval_file_path ./datasets/val_switchboard.csv \
    --model_output_dir ./vocalwhisper/speechlaugh-whisper-large/ \
    --log_dir ./checkpoints \


