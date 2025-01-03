#!/bin/bash
#SBATCH --job-name=laughtokens-whisper-subset10-a100-no-dataloader     # Job name
#SBATCH -c 32                                      # Number of cores
#SBATCH --cpus-per-task=16                         # Allocate 16 cores per task
#SBATCH --gres=gpu:ampere:1                        # Request 1 GPU
#SBATCH --constraint=a100
#SBATCH --mem=64G                                   # Request 80GB of memory (8GB whisper-large, 8GB optimizer, 4GB batch, 4GB overhead)
#SBATCH --time=72:00:00                             # Set a walltime limit
#SBATCH --mail-type=BEGIN,END,FAIL                  # Email status changes
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl  # Your email address

# Load modules (adjust versions as needed)
module purge # clean the environment before loading new modules
module load nvidia/cuda-11.8
module load nvidia/nvtop

export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"


#===========================================================
#               MULTIPROCESSING SETTINGS
#===========================================================
# export MKL_NUM_THREADS=8
# export NUMEXPR_NUM_THREADS=8
# export OMP_NUM_THREADS=8
# export TOKENIZERS_PARALLELISM=false #disable parallel tokenization when using multiprocessing

#============================================================
#                       CUDA SETTINGS
#============================================================
# Memory optimization for CUDA A40
export NVIDIA_TF32_OVERRIDE=1 # enable TF32 for A40
export CUDA_LAUNCH_BLOCKING=0
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

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
python SpeechLaughRecognition.py \
    --processed_as_dataset True \
    --dataset_dir ./datasets/switchboard \
    --model_path openai/whisper-large-v2 \
    --model_output_dir ./vocalwhisper/speechlaugh-whisper-a100/ \
    --log_dir ./logs \
    --evaluate_dir ./evaluate


