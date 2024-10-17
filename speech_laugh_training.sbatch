#!/bin/bash
#SBATCH --job-name=speechlaugh-whisper      # Job name
#SBATCH -c 16                     # Number of cores
#SBATCH --gres=gpu:ampere:2                 # Request 1 GPU
#SBATCH --mem=24G                # Request 24GB of memory (8GB whisper-large, 8GB optimizer, 4GB batch, 4GB overhead)
#SBATCH --time=24:00:00              # Set a walltime limit
#SBATCH --mail-type=BEGIN,END,FAIL    # Email status changes
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl  # Your email address

# Load modules (adjust versions as needed)
module purge # clean the environment before loading new modules
module load nvidia/cuda-11.7
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

# If preprocess 
# python preprocess.py \
#     --data_names switchboard \
#     --csv_dir ./datasets/combined/ \
#     --to_csv True \
#     --do_combine True \
#     --train_val_split True \

python SpeechLaughRecognition.py \
    --train_file_path ./datasets/combined/train_switchboard.csv \
    --eval_file_path ./datasets/combined/val_switchboard.csv \
    --processed_dataset True \
    --processed_file_path ./datasets/processed/ \
    --model_output_dir ./vocalwhisper/speechlaugh-whisper-small/ \
    --log_dir ./checkpoints \
    --evaluate_dir ./evaluate \


