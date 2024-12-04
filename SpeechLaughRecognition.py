import argparse
import numpy as np
import os
import warnings
import torch
import multiprocessing
import gc
from dotenv import load_dotenv

# For Fine-tuned Model--------------------------------
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from transformers.trainer_callback import EarlyStoppingCallback
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel   #for distributed training
import torch.multiprocessing as mp
from datasets import load_from_disk
from huggingface_hub import login
#-----------------------------------------------------------------

# Custom Modules
from modules.SpeechLaughDataCollator import DataCollatorSpeechSeq2SeqWithPadding
from modules.TrainerCallbacks import MemoryEfficientCallback, MetricsCallback, MultiprocessingCallback
from preprocess import split_dataset, transform_number_words, transform_alignment_sentence
import utils.params as prs

# For metrics and transcript transformation before computing the metrics
from utils import track_laugh_word_alignments
import pandas as pd
import evaluate
import jiwer
#====================================================================================================================================================

#===================================================================
# REMOVE TF WARNINGS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)

# Enable TF32 for faster computation with A40 GPUs=================
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
#===================================================================

"""
This is the fine-tuning Whisper model 
for the specific task of transcribing recognizing speech laughter, fillers, 
pauses, and other non-speech sounds in conversations.
"""

#=======================Multiprocessing==============================
# Initialise Multiprocessing
multiprocessing.set_start_method("spawn", force=True)
# Set multiprocessing for Pytorch
#===================================================================

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
    # special_tokens = ["[LAUGHTER]", "[COUGH]", "[SNEEZE]", "[THROAT-CLEARING]", "[SIGH]", "[SNIFF]"]
    special_tokens = ["[LAUGH]"]
    tokenizer.add_tokens(special_tokens)
    

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_path, 
        cache_dir=args.pretrained_model_dir,
        #parameters for memory optimization
        # torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        )
    #load the model from the checkpoint
    # model = WhisperForConditionalGeneration.from_pretrained(
    #     args.model_output_dir + "fine-tuned-1000steps",
    #     device_map="auto",
    # )
    
    model.resize_token_embeddings(len(tokenizer))
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False # disable caching
    model.to(device) # move model to GPUs
    #-----------------------------------------------------------------------------------------------------------------------


    #---------------------------------------------------------------------
    #                               DEVICE CONFIGS
    #---------------------------------------------------------------------
    # clear GPU cache
    torch.cuda.empty_cache()

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

    def compute_metrics(pred):
        """
        This function is used to compute the metrics for the model
        on every evaluation step (`eval_steps`) and pass the metrics to the 
        `metrics_callback` for logging to tensorboard and saving to a csv file
        Args:
            pred: predictions from the model
        Returns:
            metrics: dictionary of metrics
        """

        pred_ids = pred.predictions.cpu() if isinstance(pred.predictions, torch.Tensor) else pred.predictions
        label_ids = pred.label_ids.cpu() if isinstance(pred.label_ids, torch.Tensor) else pred.label_ids

        #-----------------------------------------------------------------------
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        # Reconstruct the REF and HYP transcripts at Decoder

        ref_transcripts = tokenizer.batch_decode(label_ids, skip_special_tokens=True) #REF transcript, contains laughter tokens [LAUGHTER] and [SPEECH_LAUGH]
        pred_transcripts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True) #HYP transcript
        
        # Transform the transcripts so that they are in the correct format
        ref_transcripts = transform_alignment_sentence(ref_transcripts) #NOT LOWERCASE

        # when computing the metrics
        pred_transcripts = transform_number_words(pred_transcripts, reverse=True) #change eg. two zero to twenty
        pred_transcripts = transform_alignment_sentence(pred_transcripts) #LOWERCASE


        # METRICS TO CALCULATE -------------------------------------------------
        wer = 100 * wer_metric.compute(predictions=pred_transcripts, references=ref_transcripts)
        f1 = 100 * f1_metric.compute(predictions=pred_transcripts, references=ref_transcripts)

        # TODO:TRY WITH OTHER METRICS ================================================
        # Track laugh metrics for each transcript pair
        # laugh_metrics = {
        #     # 'wer': wer,
        #     # 'f1': f1,
        #     'lwhr': 0, #Laugh Word Hit Rate
        #     'lthr': 0, #Laughter Token Hit Rate
        #     'lwsr': 0, #Laugh Word Substitution Rate
        #     'ltsr': 0, #Laughter Token Substitution Rate
        #     'lwdr': 0, #Laugh Word Deletion Rate
        #     'ltdr': 0, #Laughter Token Deletion Rate
        #     'lwir': 0, #Laugh Word Insertion Rate
        #     'ltir': 0 #Laughter Token Insertion Rate
        # }

        
        # alignments = jiwer.process_words(
        #     reference=ref_transcripts, 
        #     hypothesis=pred_transcripts
        # )
        
        # Calculate average laugh metrics across batch
        # for ref, hyp, align in zip(ref_transcripts, pred_transcripts, alignments.alignments):
        #     laugh_stats = track_laugh_word_alignments(ref, hyp, align) # will return "lwhr", "lthr", etc.
        #     for metric in laugh_metrics.keys():
        #         laugh_metrics[metric] += laugh_stats[metric]
        
        # Average the metrics
        # batch_size = len(ref_transcripts)
        # for metric in laugh_metrics:
        #     laugh_metrics[metric] = laugh_metrics[metric] / batch_size * 100  # Convert to percentage
        
        #==============================================================================
        
        # Combine with existing metrics
        return {
            "wer": wer,
            "f1": f1,
            # **laugh_metrics  # Add all laugh metrics
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
    ).with_format("torch", device=device) #load dataset as Tensor on GPUs
    test_dataset = test_dataset.map(
        prepare_dataset,
        remove_columns=test_dataset.column_names,
        load_from_cache_file=True,
        desc="Preparing Validation Dataset",
    ).with_format("torch", device=device)

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
        logging_dir=args.log_dir,

        
        #Training Configs--------------------------------
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4, 
        learning_rate=1e-4, #1e-5
        weight_decay=0.01,
        max_steps=5000, 
        warmup_steps=800,


        # Evaluation Configs--------------------------------
        eval_strategy="steps",
        per_device_eval_batch_size=2,
        eval_steps=500, #evaluate the model every 1000 steps - Executed compute_metrics()
        save_steps=500,
        save_strategy="steps",
        logging_steps=50,
        save_total_limit=10, #save the last 10 checkpoints
        
        report_to=["tensorboard"], #enable tensorboard for logging
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        resume_from_checkpoint=None,
        #-----------------------------------------------------

        # Computations efficiency--------------------------------
        gradient_checkpointing=True,
        fp16=True, #use mixed precision training
        tf32=True, #use TensorFloat32 for faster computation
        torch_empty_cache_steps=500, #clear CUDA memory cache at checkpoints
        #-----------------------------------------------------

        # Dataloader Configs--------------------------------
        # dataloader_num_workers=4, #default = 4 - can change larger if possible
        # dataloader_pin_memory=True, #use pinned memory for faster data transfer from CPUs to GPUs
        # dataloader_persistent_workers=True, #keep the workers alive for multiple training loops
        # dataloader_drop_last=True, #drop the last incomplete batch
        

        # push_to_hub=True,   
    )

    #================================================================
    #                   CALLBACKS FUNCTIONS
    #================================================================
    writer = SummaryWriter(log_dir=args.log_dir)

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3, #stop training if the model is not improving
        early_stopping_threshold=0.01 #consider improve if WER decrease by 0.01
    )
    memory_callback = MemoryEfficientCallback()
    metrics_callback = MetricsCallback(
        output_dir=args.log_dir, # ./checkpoints
        save_steps=1000,
        model_name="speechlaugh_subset10" #model name to saved to corresponding checkpoint
    )
    # multiprocessing_callback = MultiprocessingCallback(num_proc=4)
    #--------------------------------------------------------------------------------------------

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
    model.save_pretrained(args.model_output_dir + "fine-tuned-from-original")
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
    # parser.add_argument("--batch_size", default=8, type=int, required=False, help="Batch size for training")
    # parser.add_argument("--grad_steps", default=8, type=int, required=False, help="Number of gradient accumulation steps, which increase the batch size without extend the memory usage")
    # # parser.add_argument("--num_train_epochs", default=2, type=int, required=False, help="Number of training epochs")
    # parser.add_argument("--num_workers", default=4, type=int, required=False, help="number of workers to use for data loading, can change based on the number of cores")
    # parser.add_argument("--warmup_steps", default=800, type=int, required=False, help="Number of warmup steps")
    # parser.add_argument("--save_steps", default=1000, type=int, required=False, help="Number of steps to save the model")
    # parser.add_argument("--max_steps", default=5000, type=int, required=False, help="Maximum number of training steps")
    # parser.add_argument("--lr", default=1e-5, type=float, required=False, help="Learning rate for training")
    #------------------------------------------------------------------------------
    args = parser.parse_args()

    load_dotenv()

    try:
        # login(token=os.getenv("HUGGINGFACE_TOKEN"))
        SpeechLaughWhisper(args)
    except OSError as error:
        print(error)

    # torch.backends.cudnn.benchmark = True


