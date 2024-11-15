import argparse
import numpy as np
import os
import torch
import multiprocessing
import gc
import psutil
from dotenv import load_dotenv
import time

# For Fine-tuned Model--------------------------------
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from transformers.trainer_callback import EarlyStoppingCallback
#import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel   #for distributed training
from datasets import load_from_disk
from huggingface_hub import login
from modules.SpeechLaughDataCollator import DataCollatorSpeechSeq2SeqWithPadding
from modules.TrainerCallbacks import MemoryCallback, MetricsCallback
#----------------------------------------------------------

from preprocess import split_dataset
import utils.params as prs

# Evaluation Metrics------
import pandas as pd
import evaluate
import jiwer
#====================================================================================================================================================

output_transform = jiwer.Compose([
    # jiwer.RemovePunctuation(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemoveEmptyStrings(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])

"""
This is the fine-tuning Whisper model 
for the specific task of transcribing recognizing speech laughter, fillers, 
pauses, and other non-speech sounds in conversations.
"""

# Initialise Multiprocessing------------------------------
multiprocessing.set_start_method("spawn", force=True)


# ----------------------------------------------------
#                   MEMORY CALLBACK 
# This callback is used to clear the CUDA memory cache 
# for every 100 steps while training
# ----------------------------------------------------

def SpeechLaughWhisper(args):
    """
    This function is used to recognize speech laughter, fillers, 
    pauses, and other non-speech sounds in conversations.
    Args:
        df: pandas dataframe - contains all audios and transcripts

    """
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    #-------------------------------------------------------------------------------------------------
    #                                           MODEL CONFIGURATION 
    #-------------------------------------------------------------------------------------------------  
    
    #Processor and Tokenizer
    processor = WhisperProcessor.from_pretrained(args.model_path, cache_dir=args.pretrained_model_dir) # processor - combination of feature extractor and tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_path, cache_dir=args.pretrained_model_dir) #feature extractor
    tokenizer = WhisperTokenizer.from_pretrained(args.model_path, cache_dir=args.pretrained_model_dir) #tokenizer
    # special_tokens = ["[LAUGHTER]", "[COUGH]", "[SNEEZE]", "[THROAT-CLEARING]", "[SIGH]", "[SNIFF]", "[UH]", "[UM]", "[MM]", "[YEAH]", "[MM-HMM]"]
    special_tokens = ["[LAUGHTER]", "[SPEECH_LAUGH]"] #FIXME: Currently, only laughter and speech_laugh are used
    tokenizer.add_tokens(special_tokens)
    

    # Pre-trained Model ----------------
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path, cache_dir=args.pretrained_model_dir)
    model.resize_token_embeddings(len(tokenizer))
    model.generation_config.forced_decoder_ids = None
    model.to(device) # move model to GPUs
    #-----------------------------------------------------------------------------------------------------------------------


    #---------------------------------------------------------------------
    #                               DEVICE CONFIGS
    #---------------------------------------------------------------------
    # clear GPU cache
    torch.cuda.empty_cache()

    model.config.use_cache = False # disable caching
    #---------------------------------------------------------


    #Data Collator for padding 
    speech_laugh_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, 
        decoder_start_token_id=model.config.decoder_start_token_id,
        device=device
    )
   
    

    #=================================================================================================
    #                           PREPARE DATASET
    #=================================================================================================

    # Prepare Dataset with Batch_size > 1
    # def prepare_dataset(examples):
    #     total_examples = len(examples["audio"])
    #     with torch.no_grad():  # disable gradient computation when processing the data
    #         for i in range(total_examples):
    #             try:
    #                 example_audio = examples["audio"][i]
    #                 example_transcript = examples["transcript"][i]

    #                 if example_audio is None or example_transcript is None:
    #                     print(f"Error in processing example {i}: None audio or transcript")
    #                     continue
                
                
    #                 # remove the batch dimension such that the input_features is a 2D array:
    #                 # (n_mels, time_steps)
    #                 input_features = feature_extractor( 
    #                     raw_speech=example_audio["array"],
    #                     sampling_rate=example_audio["sampling_rate"],
    #                     padding=True,
    #                     return_tensors="pt",
    #                 ).input_features#(batch_size, n_mels, time_steps) - remove batch dimension
                    
    #                 input_features = input_features.squeeze(0) #(n_mels, time_steps) = (80, audio_time_steps)
    #                 # Add shape checking
    #                 print(f"Example {i} - Input features shape: {input_features.shape}") 

    #                 #---------LABELS ----------------------------------------
    #                 labels = tokenizer(
    #                     example_transcript, 
    #                     padding=True,
    #                     return_tensors="pt",
    #                 ).input_ids.squeeze(0) #(sequence_length) - remove batch dimension

    #                 # Add shape checking
    #                 print(f"Example {i} - Labels shape: {labels.shape}")

    #                 batch["input_features"].append(input_features) # each input_features is (80, audio_time_steps) -> List[torch.Tensor(80, audio_time_steps)] or List[List[float]]
    #                 batch["labels"].append(labels) # each labels is (sequence_length) -> List[torch.Tensor(sequence_length)]

    #             except Exception as e:
    #                 print(f"Error in processing example {i}: {e}")
    #                 continue
    #     # print(f"Successfully processed {success_examples} out of {total_examples} examples")
    #     # if success_examples == 0:
    #     #     raise ValueError("No examples were successfully processed!")

    #     return batch

    # Prepare Dataset with Batch_size = 1 (single example)\
    # TODO - The problem with `batch_size > 1` right now is that the input_features and labels are list of tensors, not tensors
    def prepare_dataset(batch):
        audio = batch["audio"]
        transcript = batch["transcript"]

        batch["input_features"] = feature_extractor(
            raw_speech=audio["array"],
            sampling_rate=audio["sampling_rate"],
            # padding=True,
            return_tensors="pt",
        ).input_features[0] #(n_mels, time_steps)

        batch["labels"] = tokenizer(
            transcript,
            # padding=True,
            return_tensors="pt",
        ).input_ids[0] #(sequence_length)

        return batch

    #=================================================================================================
    #                       COMPUTE METRICS 
    #=================================================================================================   
    
    # Evaluation Metrics -----------
    wer_metric = evaluate.load("wer", cache_dir=args.evaluate_dir) #Word Error Rate between the hypothesis and the reference transcript
    f1_metric = evaluate.load("f1", cache_dir=args.evaluate_dir) #F1 score between the hypothesis and the reference transcript
    # exact_match_metric = evaluate.load("exact_match", cache_dir=args.evaluate_dir) #compare the exact match between the hypothesis and the reference transcript
    
    def compute_metrics(pred):
        print("Computing Metrics....")

        pred_ids = pred.predictions.cpu() if isinstance(pred.predictions, torch.Tensor) else pred.predictions
        label_ids = pred.label_ids.cpu() if isinstance(pred.label_ids, torch.Tensor) else pred.label_ids

        # FIXME: IF NEEDED: Ensure the proper shape is produced before decoding
        # if pred_ids.dim() == 3:
        #     pred_ids = pred_ids.argmax(axis=-1)

        # pred_ids = pred_ids.reshape(-1, pred_ids.shape[-1])
        # label_ids = label_ids.reshape(-1, label_ids.shape[-1])
        #-----------------------------------------------------------------------

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_transcripts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True) #HYP transcript
        ref_transcripts = tokenizer.batch_decode(label_ids, skip_special_tokens=True) #REF transcript

        # Transform the transcripts so that they are in the correct format
        # when computing the metrics
        pred_transcripts = output_transform(pred_transcripts)
        ref_transcripts = output_transform(ref_transcripts)

        # METRICS TO CALCULATE -------------------------------------------------
        wer = 100 * wer_metric.compute(predictions=pred_transcripts, references=ref_transcripts)
        f1 = 100 * f1_metric.compute(predictions=pred_transcripts, references=ref_transcripts)

        # Visualise the alignment between the HYP and REF transcript
        alignments = jiwer.process_words(
            reference=ref_transcripts, 
            hypothesis=pred_transcripts
        )
        
        #write the alignment to the text file
        with open("alignment_transcripts/alignment_speechlaugh_whisper_subset10.txt", "w") as f:
            report_alignment = jiwer.visualize_alignment(
                alignments,
                show_measures=True, 
                skip_correct=False
                )
            f.write(report_alignment)

        return {
            "wer": wer,
            "f1": f1,
            }
    #--------------------------------------------------------------------------------------------


    #===============================================================================================
    #                           LOAD DATASET AND PROCESSING 
    #===============================================================================================
    if args.processed_as_dataset:
        # Load the dataset
        print("Loading the dataset as HuggingFace Dataset...")
        switchboard_dataset = load_from_disk(args.processed_file_path)
        # Split the dataset into train and validation
        train_dataset, test_dataset = split_dataset(
            switchboard_dataset, 
            subset_ratio=0.1, #TODO: Given subset ration < 1 to get smaller dataset for testing
            split_ratio=0.9, 
            split="both"
        )
    print("Dataset Loaded....\n")
    print(f"Train Dataset: {train_dataset}")
    print(f"Validation Dataset: {test_dataset}")
    print("------------------------------------------------------")
    
    #===============================================================================================
    #                       DATASET MAPPING TO TENSORS
    #===============================================================================================
    train_dataset = train_dataset.map(
        prepare_dataset,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=True,
        desc="Preparing Training Dataset",
        # batched=True,
        # batch_size=4, #4 - default= 16, small to save memory, could add up to 256 based on the GPU max memory
    
    )
    test_dataset = test_dataset.map(
        prepare_dataset,
        remove_columns=test_dataset.column_names,
        load_from_cache_file=True,
        desc="Preparing Validation Dataset",
        # batched=True,
        # batch_size=4,
    )

    # Verify dataset size
    print(f"Processed training dataset size: {len(train_dataset)}")
    # Also verify the dataset format
    print("Dataset features:", train_dataset.features)

    # ---------------------------------------------------- end of prepare dataset --------------------------------------------


    #===============================================================================================
    #                           TRAINING CONFIGURATION 
    #===============================================================================================
    # Data Parallel for distributed training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = DistributedDataParallel(model)

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_output_dir,

        #Training Configs--------------------------------
        per_device_train_batch_size=4, #8 - default batch size = 16, could add up to 256 based on the GPU max memory
        gradient_accumulation_steps=8, #8 - default = 8 increase the batch size by accumulating gradients
        learning_rate=args.lr, #1e-5
        weight_decay=0.01,
        max_steps=args.max_steps, #default = 5000 - change based on the number of epochs
        warmup_steps=args.warmup_steps, #800
        logging_dir=args.log_dir,

        # Evaluation Configs--------------------------------
        eval_strategy="steps",
        per_device_eval_batch_size=2,
        eval_steps=1000,
        report_to=["tensorboard"], #enable tensorboard for logging
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        resume_from_checkpoint=None,
        #-----------------------------------------------------

        # Computations efficiency--------------------------------
        gradient_checkpointing=True,
        fp16=True, #use mixed precision training
        #-----------------------------------------------------

        # Dataloader Configs--------------------------------
        dataloader_num_workers=4, #default = 4 - can change larger if possible
        dataloader_pin_memory=True, #use pinned memory for faster data transfer from CPUs to GPUs
        dataloader_persistent_workers=False, #keep the workers alive for multiple training loops
        dataloader_prefetch_factor=1, #number of batches to prefetch from the dataloader (1 for reduce memory usage)
        dataloader_drop_last=True, #drop the last incomplete batch
        
        #-----------------------------------------------------
        remove_unused_columns=False,
        logging_steps=100,
        save_steps=1000,
        save_strategy="steps",
        save_total_limit=5, #save the last 5 checkpoints
        # push_to_hub=True,   
    )

    writer = SummaryWriter(log_dir=args.log_dir)

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3, #stop training if the model is not improving
        early_stopping_threshold=0.01 #consider improve if WER decrease by 0.01
    )
    memory_callback = MemoryCallback()
    metrics_callback = MetricsCallback(
        output_dir=args.log_dir,
        save_steps=1000
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=speech_laugh_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            early_stopping, 
            memory_callback,
            metrics_callback
            ], #FIXME: Add MemoryCallback() to clear the CUDA memory cache for every 100 steps
    )


    #===============================================================================================
    #                                 MONITOR GPU MEMORY
    #===============================================================================================

    def cleanup_workers():
        """Cleanup function for workers"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize() #wait for all operations to complete (finished cache clear)
        gc.collect()

    #-------------------------------------------------------------------------------------------------


    #===============================================================================================
    #                           TRAINING 
    #===============================================================================================
    # Create DataFrames to store the training and evaluation metrics
    # training_metrics = pd.DataFrame(columns=["step", "training_loss", "epoch", "validation_loss","wer", "f1"])

    #================================
    # TRAINING THE MODEL
    #================================
    try:
        trainer.train()
    except Exception as e:
        print(f"Error in training: {e}")
        cleanup_workers() #clear the CUDA memory cache
    finally:
        cleanup_workers() #clear the CUDA memory cache

    # Save the final model
    model.save_pretrained(args.model_output_dir + "fine-tuned")
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
    parser.add_argument("--log_dir", default="./checkpoints", type=str, required=False, help="Path to the log directory")
    parser.add_argument("--evaluate_dir", default="./evaluate", type=str, required=False, help="Path to the evaluation directory")

    # Training Configs
    parser.add_argument("--batch_size", default=8, type=int, required=False, help="Batch size for training")
    parser.add_argument("--grad_steps", default=8, type=int, required=False, help="Number of gradient accumulation steps, which increase the batch size without extend the memory usage")
    # parser.add_argument("--num_train_epochs", default=2, type=int, required=False, help="Number of training epochs")
    parser.add_argument("--num_workers", default=4, type=int, required=False, help="number of workers to use for data loading, can change based on the number of cores")
    parser.add_argument("--warmup_steps", default=800, type=int, required=False, help="Number of warmup steps")
    parser.add_argument("--save_steps", default=1000, type=int, required=False, help="Number of steps to save the model")
    parser.add_argument("--max_steps", default=5000, type=int, required=False, help="Maximum number of training steps")
    parser.add_argument("--lr", default=1e-5, type=float, required=False, help="Learning rate for training")
    #------------------------------------------------------------------------------
    args = parser.parse_args()

    load_dotenv()

    try:
        # login(token=os.getenv("HUGGINGFACE_TOKEN"))
        SpeechLaughWhisper(args)
    except OSError as error:
        print(error)

    # torch.backends.cudnn.benchmark = True


