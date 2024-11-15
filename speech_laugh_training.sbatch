#!/bin/bash
#SBATCH --job-name=laughtokens-whisper-subset-10-16c-40G-1000-save       # Job name
#SBATCH -c 16                                       # Number of cores
#SBATCH --gres=gpu:ampere:1                        # Request 1 GPU
#SBATCH --mem=40G                                   # Request 24GB of memory (8GB whisper-large, 8GB optimizer, 4GB batch, 4GB overhead)
#SBATCH --time=64:00:00                             # Set a walltime limit
#SBATCH --mail-type=BEGIN,END,FAIL                  # Email status changes
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl  # Your email address

# Load modules (adjust versions as needed)
module purge # clean the environment before loading new modules
# module load nvidia/cuda-11.7
# module load nvidia/cuda-11.x_cudnn-8.6
# module load nvidia/cuda-11.x_tensorrt-8.6
module load nvidia/cuda-11.8
module load nvidia/nvtop

# Add GPU monitoring
# nvidia-smi dmon -s pucvt -o TD &

# Print some useful information
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)" # log hostname
echo "Working Directory = $(pwd)"
echo "Name of nodes used          : "$SLURM_JOB_NODELIST
echo "Gpu devices                 : "$CUDA_VISIBLE_DEVICES
echo "Starting worker: "

# Activate your environment (if applicable)
source activate .venv

export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Run your script with the appropriate arguments
python SpeechLaughRecognition.py \
    --processed_as_dataset True \
    --processed_file_path ./datasets/switchboard/token_speechlaugh/switchboard_dataset \
    --model_output_dir ./vocalwhisper/speechlaughwhisper-subset-10/ \
    --log_dir ./checkpoints \
    --evaluate_dir ./evaluate \


