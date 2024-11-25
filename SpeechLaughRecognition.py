import argparse
import numpy as np
import os
import warnings
import torch
import multiprocessing
import torch.multiprocessing as mp
import gc
from dotenv import load_dotenv

# For Fine-tuned Model--------------------------------
from transformers import Trainer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datasets import load_from_disk
from huggingface_hub import login
#-----------------------------------------------------------------

# Custom Modules
from modules import (
    DataCollatorSpeechSeq2SeqWithPadding,
    # MemoryEfficientCallback,
    MetricsCallback,
    MultiprocessingCallback,
)
from preprocess import split_dataset
from utils import (
    # for training process
    init_multiprocessing,
    gpu_config,
    save_model_components,
    prepare_dataset_with_a40,
    load_metrics,
)

# Evaluation Metrics------
import pandas as pd
#====================================================================================================================================================


#===================================================================
# REMOVE TF WARNINGS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false" #disable parallel tokenization when using multiprocessing
#===================================================================

#==========================================================================================================
#           HELPER FUNCTIONS TO LOAD AND SAVE MODEL COMPONENTS
#========================================================================================================== 
def initialize_model_config(model_path, cache_dir):
    """
    Initialize the model configuration specified for A40 GPU
    with low memory usage and not using use_cache
    """
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, 
        cache_dir=cache_dir,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    model.config.use_cache = False
    return model


#==========================================================================================================
#                               SPEECH LAUGH RECOGNITION USING WHISPER
#==========================================================================================================
def SpeechLaughWhisper(args):
    """
    This is the fine-tuning Whisper model on `Switchboard dataset`
    used to recognize speech-laugh and laughter in conversational speech.

    """
    print("CUDA device name:", torch.cuda.get_device_name(0))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Device: ", device)

    #==========================================================================================================
    #                                       ENABLE MULTIPROCESSING
    #==========================================================================================================
    n_proc = init_multiprocessing()


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")

        gpu_config() #to enable faster training: CuDNN enabled, CuDNN allow_tf32 enabled, CuDNN matmul allow_tf32 enabled
    else:
        device = torch.device("cpu")
    
    print("Device: ", device)
    #=======================================================================
    #               MODEL CONFIGURATION 
    #=======================================================================
    
    #               MODELS
    #=====================================================================
    model = initialize_model_config(
        model_path=args.model_path, 
        cache_dir=args.pretrained_model_dir
    )

    #               Processor
    #=====================================================================  
    processor = WhisperProcessor.from_pretrained(
        args.model_path, 
        cache_dir=args.pretrained_model_dir
    ) # processor - combination of feature extractor and tokenizer

    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    # special_tokens = ["[LAUGHTER]", "[SPEECHLAUGH]"]
    special_tokens = ["[LAUGHTER]"] #add only laughter token to tokenizer
    tokenizer.add_tokens(special_tokens)

    model.resize_token_embeddings(len(tokenizer))


    #       DATA COLLATOR
    #=====================================================================
    speech_laugh_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, 
        decoder_start_token_id=model.config.decoder_start_token_id,
        device=device
    )

    # LOAD METRICS
    wer_metric, f1_metric = load_metrics()


    #===============================================================================================
    #           LOAD DATASET AND PROCESSING 
    #===============================================================================================
    if args.processed_as_dataset:
        print("Loading the Dataset as HuggingFace Dataset...")
        switchboard_dataset = load_from_disk(args.processed_file_path)
        # Split the dataset into train and validation
        train_dataset, eval_dataset = split_dataset(
            switchboard_dataset, 
            subset_ratio=0.1, #TODO: Given subset ration < 1 to get smaller dataset for testing
            split_ratio=0.9, 
            split="both"
        )
    print(f"Train Dataset: {train_dataset}")
    print(f"Validation Dataset: {eval_dataset}")
    print("------------------------------------------------------")
    
    #===============================================================================================
    #        DATASET MAPPING TO TENSORS
    #===============================================================================================
    train_dataset = prepare_dataset_with_a40(
        train_dataset,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        num_proc=n_proc # Use 16 CPU cores for multiprocessing
    )
    eval_dataset = prepare_dataset_with_a40(
        eval_dataset, 
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        num_proc=n_proc # Use 16 CPU cores for multiprocessing
    )
    # Also verify the dataset format
    print("Dataset Processed for Training!....\n")
    #===============================================================================================

    #===============================================================================================
    #                           TRAINING CONFIGURATION 
    #===============================================================================================

        # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_output_dir,
        overwrite_output_dir=True, #using for save model to checkpoint
        logging_dir=args.log_dir,
        save_total_limit=10,

        do_train=True,
        do_eval=True,

        per_device_train_batch_size=4,  # Reduced batch size
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=4,

        evaluation_strategy="steps",
        save_strategy="steps",
        max_steps=5000,
        eval_steps=1000,
        save_steps=1000,
        logging_steps=100,
        warmup_steps=500,

        #=============================================================
        # IF USING EPOCHS
        #=============================================================
        # num_train_epochs=50,
        # eval_strategy="epoch",
        # save_strategy="epoch",
        #=============================================================

        learning_rate=1e-5,
        fp16=True,
        tf32=True,
        gradient_checkpointing=True,     # Enable checkpointing
        torch_empty_cache_steps=500, #empty cache every 500 steps
        torch_compile=True,

        metric_for_best_model="wer",
        greater_is_better=False,
        load_best_model_at_end=True,
        report_to=["tensorboard"],

        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=True,

    )


    #================================================================
    #                   CALLBACKS FUNCTIONS
    #================================================================
    writer = SummaryWriter(log_dir=args.log_dir)

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3, #stop training if the model is not improving
        early_stopping_threshold=0.01 #consider improve if WER decrease by 0.01
    )
    metrics_callback = MetricsCallback(
        output_dir=args.log_dir, # ./logs
        save_steps=1000, #save the model every 1000 steps - SAVE LESS FREQUENTLY
        model_name="speechlaugh_subset10" #model name to saved to corresponding checkpoint
    )
    multiprocessing_callback = MultiprocessingCallback(num_proc=n_proc)
    #--------------------------------------------------------------------------------------------
    
    #===============================================================================================    
    #               DATA LOADERS
    #===============================================================================================
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=4,  # Adjust batch size as needed
    #     shuffle=True,
    #     num_workers=16,  # Utilize multiple CPU cores for parallel loading
    # )
    # eval_dataloader = DataLoader(
    #     eval_dataset,
    #     batch_size=8,  # Adjust batch size as needed
    #     num_workers=16,
    # )



    #===============================================================================================
    #                           TRAINING 
    #===============================================================================================

    # Define your Trainer with the optimized data loaders
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=speech_laugh_collator,
        # train_dataloader=train_dataloader,  # Use the optimized data loader
        # eval_dataloader=eval_dataloader,  # Use the optimized data loader
        callbacks=[
            early_stopping, 
            metrics_callback, #Save metrics to tensorboard
            multiprocessing_callback #Enable multiprocessing when training starts
        ],
    )

    trainer.train()

    output_dir = args.model_output_dir + "fine-tuned-1"
    save_model_components(
        model=model, 
        tokenizer=tokenizer, 
        # generation_config=generation_config, 
        output_dir=output_dir
    )

    #-----------------------------------------end of training ------------------------------


    # PUSH MODEL TO HUB ----------------------------------------------------------
    # kwargs = {
    #     "dataset_tags": "hhoangphuoc/switchboard",
    #     "dataset": "Switchboard",
    #     # "dataset_args": "config: hi, split: test",
    #     "model_name": "Speech Laugh Whisper - Phuoc Ho",
    #     "finetuned_from": "openai/whisper-large-v2",
    #     "tasks": "automatic-speech-recognition",
    # }
    # trainer.push_to_hub(**kwargs)
    #-------------------------------------------------------------------------------------------


# MAIN FUNCTION ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Laugh Recognition")
    
    # Data Configs
    parser.add_argument("--processed_as_dataset", default=False, type=bool, required=False, help="Whether or not process as Huggingface dataset")
    parser.add_argument("--processed_file_path", default="./datasets/processed_dataset/", type=str, required=False, help="Path to the test.csv file")
    
    #FIXME: IF LOAD FROM CSV FILE, NEED THESE FILE PATH ----------------------------------
    parser.add_argument("--train_file_path", default="./datasets/train_switchboard.csv", type=str, required=False, help="Path to the train.csv file")
    parser.add_argument("--eval_file_path", default="./datasets/val_switchboard.csv", type=str, required=False, help="Path to the val.csv file")
    #-------------------------------------------------------------------------------------

    # Model Configs
    parser.add_argument("--model_path", default="openai/whisper-small", type=str, required=False, help="Select pretrained model")
    parser.add_argument("--pretrained_model_dir", default="./ref_models/pre_trained/", type=str, required=False, help="Name of the model")
    parser.add_argument("--model_output_dir", default="./vocalwhisper/speechlaugh-whisper-small/", type=str, required=False, help="Path to the output directory")
    parser.add_argument("--log_dir", default="./logs", type=str, required=False, help="Path to the log directory")
    parser.add_argument("--evaluate_dir", default="./evaluate", type=str, required=False, help="Path to the evaluation directory")
    #------------------------------------------------------------------------------
    args = parser.parse_args()

    load_dotenv()

    try:
        SpeechLaughWhisper(args)
    except OSError as error:
        print(error)

