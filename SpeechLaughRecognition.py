import argparse
import numpy as np
import os
import warnings
import torch
import multiprocessing
import gc
from dotenv import load_dotenv

# For Fine-tuned Model--------------------------------
from transformers import (
    WhisperProcessor, 
    WhisperTokenizer, 
    WhisperFeatureExtractor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    TrainerCallback
)
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
from preprocess import (
    split_dataset, 
    transform_number_words, 
    transform_alignment_sentence,
    combined_dataset
)
# For metrics and transcript transformation before computing the metrics
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
    # special_tokens = ["[LAUGH]"]
    special_tokens = ["[laugh]"] #FIXME-SPECIAL TOKEN FOR LAUGHTER is [laugh] since we want to lowercase the laughter events  
    tokenizer.add_tokens(special_tokens)
    

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_path, 
        cache_dir=args.pretrained_model_dir,
        torch_dtype=torch.float16, #use mixed precision for faster computation
        )
    
    model.resize_token_embeddings(len(tokenizer))
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False # disable caching
    model.to(device) # move model to GPUs
    #-----------------------------------------------------------------------------------------------------------------------

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
        """
        This function is used to prepare the dataset for the model
        Args:
            batch: batch of data contains:
                - audio: audio data
                - transcript: transcript data
                (with lowercase the laugh events, so that the transcript will be lower and the laughter events will be [laugh],
                and speechlaugh is a normal word as speech variation) - FIXME - THIS PART WAS DONE IN PREPROCESSING?
        Returns:
            batch: batch of data
        """
        audio = batch["audio"]
        transcript = batch["transcript"].lower()


        batch["input_features"] = feature_extractor(
            raw_speech=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).input_features[0] #[0] #(n_mels, time_steps)

        batch["labels"] = tokenizer(
            transcript,
            return_tensors="pt",
        ).input_ids[0] #[0] #(sequence_length)

        return batch

    #=================================================================================================
    #                       COMPUTE METRICS 
    #=================================================================================================   
    
    # Evaluation Metrics -----------
    # wer_metric = evaluate.load("wer", cache_dir=args.evaluate_dir) #Word Error Rate between the hypothesis and the reference transcript
    # f1_metric = evaluate.load("f1", cache_dir=args.evaluate_dir) #F1 score between the hypothesis and the reference transcript

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
        # label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        # Reconstruct the REF and HYP transcripts at Decoder
        ref_transcripts = tokenizer.batch_decode(label_ids, skip_special_tokens=True) #REF transcript, contains laughter tokens [LAUGHTER] and [SPEECH_LAUGH]
        pred_transcripts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True) #HYP transcript
        
        # NORMALISED THE TRANSCRIPT
        ref_transcripts = transform_alignment_sentence(ref_transcripts) #LOWERCASE  

        # when computing the metrics
        pred_transcripts = transform_number_words(pred_transcripts, reverse=True) #change eg. two zero to twenty
        pred_transcripts = transform_alignment_sentence(pred_transcripts) #LOWERCASE


        # METRICS TO CALCULATE -------------------------------------------------

        alignment = jiwer.process_words(
            reference=ref_transcripts, 
            hypothesis=pred_transcripts,
        )
        
        #-----------------------------------------------------------------------------------------
        # wer = 100 * wer_metric.compute(predictions=pred_transcripts, references=ref_transcripts)
        # f1 = 100 * f1_metric.compute(predictions=pred_transcripts, references=ref_transcripts)
        #-----------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------
        #               CALCULATE F1 AND TOKEN RATE FOR [laugh] TOKEN MATCH
        #               GO THROUGH EACH PAIR OF SENTENCE AND CALCULATE THE METRICS
        #================================================================================== 

        ref_words = ref_transcripts.split()
        hyp_words = pred_transcripts.split()

        # Get the laughter indices
        eval_laugh_indices = {
            i: {
                'word': word,
                'type': 'laugh', #'laugh' or 'speechlaugh' or 'laugh_intext'
                'lower': word.lower()
            }
            for i, word in enumerate(ref_words)
            if word == '[laugh]'  #either speech-laugh (word.upper) or laugh (word = [LAUGH])
        }
        token_stat_summary = {
            'total_TH': 0,
            'total_TS': 0,
            'total_TD': 0,
            'total_TI': 0,
            'total_token_operations': 0, # total number of token operations in alignment process
        }
        for alignment in alignment.alignments:
            for chunk in alignment:
                # Get the aligning  words from reference and hypothesis
                ref_start, ref_end = chunk.ref_start_idx, chunk.ref_end_idx
                hyp_start, hyp_end = chunk.hyp_start_idx, chunk.hyp_end_idx

                #==================================================================================
                #                           ALIGNMENT CHUNK BY TYPE
                #==================================================================================
                if chunk.type == "equal":
                    # If the index of the word 
                    for i, (ref_idx, hyp_idx) in enumerate(zip(range(ref_start, ref_end), 
                                                            range(hyp_start, hyp_end))):
                        if ref_idx in eval_laugh_indices:
                            token_stat_summary['total_TH'] += 1
                elif chunk.type == "substitute":
                    # Check for substitutions
                    for i, ref_idx in enumerate(range(ref_start, ref_end)):
                        if ref_idx in eval_laugh_indices:
                            token_stat_summary['total_TS'] += 1
                elif chunk.type == "delete":
                    # Check for deletions
                    for ref_idx in range(ref_start, ref_end):
                        if ref_idx in eval_laugh_indices:
                            token_stat_summary['total_TD'] += 1
                elif chunk.type == "insert":
                    # Check for insertions
                    for hyp_idx in range(hyp_start, hyp_end):
                        if hyp_idx in eval_laugh_indices:
                            token_stat_summary['total_TI'] += 1

            #------------------------------------------------------------------------------------------
        
        # CALCULATE F1 AND TOKEN RATE FOR [laugh] TOKEN MATCH
        total_token_operations = token_stat_summary['total_TH'] + token_stat_summary['total_TS'] + token_stat_summary['total_TD']
        
        th_rate = token_stat_summary['total_TH'] / total_token_operations if total_token_operations > 0 else 0
        ts_rate = token_stat_summary['total_TS'] / total_token_operations if total_token_operations > 0 else 0
        td_rate = token_stat_summary['total_TD'] / total_token_operations if total_token_operations > 0 else 0
        ti_rate = token_stat_summary['total_TI'] / total_token_operations if total_token_operations > 0 else 0
        #------------------------------------------------------------------------------------------

        TP = token_stat_summary['total_TH']
        FP = token_stat_summary['total_TS'] + token_stat_summary['total_TI']
        FN = token_stat_summary['total_TD'] + token_stat_summary["total_TS"]
        
        # COMPUTE OVERALL WER, F1
        wer = alignment.wer #FIXME - USING THIS IF WER IS NOT WORKING
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return {
            "wer": wer,
            "f1": f1,
            "th": th_rate,
            "ts": ts_rate,
            "td": td_rate,
            "ti": ti_rate,
        }
    #--------------------------------------------------------------------------------------------


    #===============================================================================================
    #                           LOAD DATASET AND PROCESSING 
    #===============================================================================================
    if args.processed_as_dataset:
        # Load the dataset
        print("Loading the dataset as HuggingFace Dataset...")
        switchboard_dataset = load_from_disk(os.path.join(args.dataset_dir, "swb_full"))
        print(f"Switchboard Dataset: {switchboard_dataset}")
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
        load_from_cache_file=False,
        desc="Preparing Training Dataset",
    ).with_format("torch") #load dataset as Tensor on GPUs

    test_dataset = test_dataset.map(
        prepare_dataset,
        remove_columns=test_dataset.column_names,
        load_from_cache_file=False,
        desc="Preparing Validation Dataset",
    ).with_format("torch")

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
        per_device_train_batch_size=16, #4
        gradient_accumulation_steps=2, #4
        learning_rate=5e-5, #or 1e-4
        weight_decay=0.01,
        max_steps=6000, 
        warmup_steps=800,


        # Evaluation Configs--------------------------------
        eval_strategy="steps",
        per_device_eval_batch_size=16,
        eval_steps=1000, #evaluate the model every 1000 steps - Executed compute_metrics()
        save_steps=1000,
        save_strategy="steps",
        logging_steps=100,
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
        torch_empty_cache_steps=1000, #clear CUDA memory cache at checkpoints
        #-----------------------------------------------------

        # Dataloader Configs--------------------------------
        # dataloader_num_workers=4, #default = 4 - can change larger if possible
        dataloader_pin_memory=True, #use pinned memory for faster data transfer from CPUs to GPUs
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
        model_name="speechlaugh_10_percent" #model name to saved to corresponding checkpoint
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
        print("Cleaning CUDA memory cache...")
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
        cleanup_workers() #clear the CUDA memory cache
        trainer.train()
        log_history = trainer.state.log_history
        # save the log history to txt file
        with open(os.path.join("./logs", "log_history.txt"), "w") as f:
            for entry in log_history:
                f.write(str(entry) + "\n")
    except Exception as e:
        print(f"Error in training: {e}")
        cleanup_workers() #clear the CUDA memory cache
    finally:
        cleanup_workers() #clear the CUDA memory cache

    # Save the final model
    model.save_pretrained(args.model_output_dir + "speechlaugh-fine_tuned")
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
    parser.add_argument("--dataset_dir", default="./datasets/switchboard", type=str, required=False, help="Path to the dataset directory")
    #-------------------------------------------------------------------------------------

    # Model Configs
    parser.add_argument("--model_path", default="openai/whisper-large-v2", type=str, required=False, help="Select pretrained model")
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


