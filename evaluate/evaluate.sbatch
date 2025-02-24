#!/bin/bash
#SBATCH --job-name=eval-finetuned-whisper-(buckeye/swb)       # Job name
#SBATCH -c 8                                                   # Number of cores
#SBATCH --mem=12G                                                # Request 12GB memory (8GB swb_test + 4GB for the model)
#SBATCH --gres=gpu:ampere:1                                     # Request 1 GPU
#SBATCH --time=24:00:00                                          # Set a walltime limit
#SBATCH --mail-type=BEGIN,END,FAIL                              # Email status changes
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl             # Your email address

# Load modules (adjust versions as needed)
module purge
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


#===================================================================================================
#                                   FINETUNED-MODELS ON SWB_TEST
#===================================================================================================

# WHISPER [DONE]
# python evaluate.py \
#     --dataset_dir ../datasets/switchboard/whisper/swb_test \
#     --model_name openai/whisper-large-v2 \
#     --model_type whisper \
#     --pretrained_model_dir ../ref_models/pre_trained \
#     --output_dir ../alignment_transcripts/buckeye2/whisper \

# # WAV2VEC2 [DONE]
# python evaluate.py \
#     --dataset_dir ../datasets/switchboard/whisper/swb_test \
#     --model_name facebook/wav2vec2-large-960h-lv60 \
#     --model_type wav2vec2 \
#     --pretrained_model_dir ../ref_models/pre_trained \
#     --output_dir ../alignment_transcripts/buckeye2/wav2vec2 \



# #=======================================================
# #               FINETUNED WHISPER
# #=======================================================

# Finetuned-whisper-nolaugh-swb [DONE]
# FINETUNED VERSION WITHOUT <LAUGH> TOKEN (finetuned-whisper-nolaugh)
# python evaluate.py \
#     --dataset_dir ../datasets/switchboard/whisper/swb_test \
#     --model_name  finetuned-whisper-nolaugh-10epochs\
#     --model_type whisper \
#     --pretrained_model_dir ../fine-tuned/whisper-nolaugh \
#     --output_dir ../alignment_transcripts/switchboard/finetuned_whisper_nolaugh \


# Finetuned-whisper-swb [DONE]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_dataset \
#     --model_name  finetuned-whisper-10epochs\
#     --model_type whisper \
#     --pretrained_model_dir ../fine-tuned/whisper/ \
#     --output_dir ../alignment_transcripts/buckeye3/finetuned_whisper \


# Finetuned-whisper-buckeye [WAIT FOR MODEL FINETUNED]
# python evaluate.py \
#     --dataset_dir ../datasets/switchboard/whisper/swb_test \
#     --model_name  finetuned-whisper-buckeye\
#     --model_type whisper \
#     --pretrained_model_dir ../fine-tuned/whisper \
#     --output_dir ../alignment_transcripts/switchboard/finetuned_whisper-buckeye\


# Finetuned-whisper-nolaugh-buckeye  [WAIT FOR MODEL FINETUNED]
# python evaluate.py \
#     --dataset_dir ../datasets/switchboard/whisper/swb_test \
#     --model_name  finetuned-whisper-nolaugh-buckeye\
#     --model_type whisper \
#     --pretrained_model_dir ../fine-tuned/whisper-nolaugh \
#     --output_dir ../alignment_transcripts/switchboard/finetuned_whisper_nolaugh-buckeye\

#=======================================================
#               FINETUNED WAV2VEC2
#=======================================================

# Fine-tuned wav2vec2-swb [DONE]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye2/buckeye_dataset \
#     --model_name  finetuned-wav2vec2-checkpoint-20500\
#     --model_type wav2vec2 \
#     --pretrained_model_dir ../checkpoints/wav2vec2-batch32 \
#     --output_dir ../alignment_transcripts/buckeye2/finetuned_wav2vec2 \


# Fine-tuned Wav2Vec2-nolaugh-swb [DONE]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye2/buckeye_dataset\
#     --model_name  finetuned-wav2vec2-checkpoint-20100\
#     --model_type wav2vec2 \
#     --pretrained_model_dir ../fine-tuned/wav2vec2-nolaugh \
#     --output_dir ../alignment_transcripts/buckeye2/finetuned_wav2vec2_nolaugh \


# Fine-tuned wav2vec2-buckeye


# Fine-tuned wav2vec2-nolaugh-buckeye [RUNNING]
# python evaluate.py \
#     --dataset_dir ../datasets/switchboard/whisper/swb_test \
#     --model_name  finetuned-wav2vec2-nolaugh-buckeye\
#     --model_type wav2vec2 \
#     --pretrained_model_dir ../fine-tuned/wav2vec2-nolaugh \
#     --output_dir ../alignment_transcripts/switchboard/finetuned_wav2vec2_nolaugh-buckeye\

#===============================================================================================================





# ================================================================================================================
#                                       FINETUNED MODELS ON BUCKEYE 
# ================================================================================================================


# WHISPER [RUNNING]
python evaluate.py \
    --dataset_dir ../datasets/buckeye3/buckeye_dataset \
    --model_name openai/whisper-large-v2 \
    --model_type whisper \
    --pretrained_model_dir ../ref_models/pre_trained \
    --output_dir ../alignment_transcripts/buckeye/whisper-FT \

#WAV2VEC2 [RUNNING]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_dataset \
#     --model_name facebook/wav2vec2-large-960h-lv60 \
#     --model_type wav2vec2 \
#     --pretrained_model_dir ../ref_models/pre_trained \
#     --output_dir ../alignment_transcripts/buckeye/wav2vec2-FT \



#=====================================================
#               FINETUNED-WHISPER
#=====================================================

# Finetune-whisper-swb [RUNNING]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_dataset\
#     --model_name  finetuned-whisper-10epochs\
#     --model_type whisper \
#     --pretrained_model_dir ../fine-tuned/whisper \
#     --output_dir ../alignment_transcripts/buckeye/whisper+FT+L\

# Finetuned-whisper-nolaugh-swb [RUNNING]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_dataset \
#     --model_name  finetuned-whisper-nolaugh-10epochs\
#     --model_type whisper \
#     --pretrained_model_dir ../fine-tuned/whisper-nolaugh \
#     --output_dir ../alignment_transcripts/buckeye/whisper+FT-L\


#=====================================================
#               FINETUNED-WAV2VEC2
#=====================================================

# Finetuned-wav2vec2-swb [RUNNING]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_dataset \
#     --model_name  finetuned-wav2vec2-swb\
#     --model_type wav2vec2 \
#     --pretrained_model_dir ../fine-tuned/wav2vec2 \
#     --output_dir ../alignment_transcripts/buckeye/wav2vec2+FT+L\


# Finetuned-wav2vec2-nolaugh-swb [RUNNING]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_dataset \
#     --model_name  finetuned-wav2vec2-nolaugh-swb\
#     --model_type wav2vec2 \
#     --pretrained_model_dir ../fine-tuned/wav2vec2-nolaugh \
#     --output_dir ../alignment_transcripts/buckeye/wav2vec2+FT-L\

# ===============================================================================================================



# ================================================================================================================
#                                       FINETUNED MODELS ON BUCKEYE_TEST 
# ================================================================================================================


# WHISPER [RUNNING]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_test \
#     --model_name openai/whisper-large-v2 \
#     --model_type whisper \
#     --pretrained_model_dir ../ref_models/pre_trained \
#     --output_dir ../alignment_transcripts/buckeye_test_v2/whisper-FT \

# WAV2VEC2 [RUNNING]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_test \
#     --model_name facebook/wav2vec2-large-960h-lv60 \
#     --model_type wav2vec2 \
#     --pretrained_model_dir ../ref_models/pre_trained \
#     --output_dir ../alignment_transcripts/buckeye_test_v2/wav2vec2-FT \



#=====================================================
#               FINETUNED-WHISPER
#=====================================================

# Finetuned-whisper-buckeye [WAIT FOR MODEL FINETUNED]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_test \
#     --model_name  finetuned-whisper-buckeye\
#     --model_type whisper \
#     --pretrained_model_dir ../fine-tuned/whisper \
#     --output_dir ../alignment_transcripts/buckeye_test_v2/whisper+FT+L_buckeye\


# Finetuned-whisper-nolaugh-buckeye [WAIT FOR MODEL FINETUNED]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_test \
#     --model_name  finetuned-whisper-nolaugh-buckeye\
#     --model_type whisper \
#     --pretrained_model_dir ../fine-tuned/whisper-nolaugh \
#     --output_dir ../alignment_transcripts/buckeye_test_v2/whisper+FT-L_buckeye\


# Finetune-whisper-swb [RUNNING]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_test\
#     --model_name  finetuned-whisper-10epochs\
#     --model_type whisper \
#     --pretrained_model_dir ../fine-tuned/whisper \
#     --output_dir ../alignment_transcripts/buckeye_test_v2/whisper+FT+L_swb\

# Finetuned-whisper-nolaugh-swb [RUNNING]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_test \
#     --model_name  finetuned-whisper-nolaugh-10epochs\
#     --model_type whisper \
#     --pretrained_model_dir ../fine-tuned/whisper-nolaugh \
#     --output_dir ../alignment_transcripts/buckeye_test_v2/whisper+FT-L_swb\


#=====================================================
#               FINETUNED-WAV2VEC2
#=====================================================

# Finetuned-wav2vec2-swb [RUNNING]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_test \
#     --model_name  finetuned-wav2vec2-swb\
#     --model_type wav2vec2 \
#     --pretrained_model_dir ../fine-tuned/wav2vec2 \
#     --output_dir ../alignment_transcripts/buckeye_test_v2/wav2vec2+FT+L_swb\


# Finetuned-wav2vec2-nolaugh-swb [RUNNING]
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_test \
#     --model_name  finetuned-wav2vec2-nolaugh-swb\
#     --model_type wav2vec2 \
#     --pretrained_model_dir ../fine-tuned/wav2vec2-nolaugh \
#     --output_dir ../alignment_transcripts/buckeye_test_v2/wav2vec2+FT-L_swb\



# Finetuned-wav2vec2-buckeye [RUNNING] //Current eval_wer=0.35
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_test \
#     --model_name  finetuned-wav2vec2-buckeye\
#     --model_type wav2vec2 \
#     --pretrained_model_dir ../fine-tuned/wav2vec2 \
#     --output_dir ../alignment_transcripts/buckeye_test_v2/wav2vec2+FT+L_buckeye\

# Finetuned-wav2vec2-nolaugh-buckeye [RUNNING] //Current eval_wer=0.33
# python evaluate.py \
#     --dataset_dir ../datasets/buckeye3/buckeye_test \
#     --model_name  finetuned-wav2vec2-nolaugh-buckeye\
#     --model_type wav2vec2 \
#     --pretrained_model_dir ../fine-tuned/wav2vec2-nolaugh \
#     --output_dir ../alignment_transcripts/buckeye_test_v2/wav2vec2+FT-L_buckeye\

# ===============================================================================================================
    
#----------------------------------------+
# TO PRODUCE ALIGNMENT PLOTS
#----------------------------------------+
# python align_dtw.py