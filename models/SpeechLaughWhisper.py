import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import warnings
import torch
import gc
from dotenv import load_dotenv
from huggingface_hub import login

# For Fine-tuned Model--------------------------------
from transformers import (
    WhisperProcessor, 
    WhisperTokenizer, 
    WhisperFeatureExtractor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    EvalPrediction,
    TrainerCallback
)
from transformers.trainer_callback import EarlyStoppingCallback
from torch.utils.tensorboard import SummaryWriter
from datasets import load_from_disk
#-----------------------------------------------------------------

# Custom Modules
from modules import (
    DataCollatorSpeechSeq2SeqWithPadding,
    MemoryEfficientCallback,
    MetricsCallback,
    CustomSeq2SeqTrainer
)
from preprocess import (
    split_dataset, 
    transform_number_words,
    find_total_laughter_speechlaugh
)

# Evaluation
import evaluate
import jiwer

#===================================================================
# REMOVE TF WARNINGS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)
# Enable TF32 for faster computation with A40 GPUs=================
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cuda.allow_tf32 = True
# torch.backends.cuda.matmul.allow_tf32 = True
#===================================================================================

#===================================================================================

"""
This is the fine-tuning Whisper model 
for the specific task of transcribing recognizing speech laughter, fillers, 
pauses, and other non-speech sounds in conversations.
"""

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
    model_path = args.model_path
    print(f"Model Path: {model_path}")
    model_cache_dir = args.pretrained_model_dir
    print(f"Pretrained Model Directory: {args.pretrained_model_dir}")
    #Processor and Tokenizer
    processor = WhisperProcessor.from_pretrained(model_path, cache_dir=model_cache_dir) # processor - combination of feature extractor and tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path, cache_dir=model_cache_dir) #feature extractor
    
    
    tokenizer = WhisperTokenizer.from_pretrained(model_path, cache_dir=model_cache_dir) #tokenizer
    special_tokens = ["<laugh>"]
    tokenizer.add_tokens(special_tokens)
    

    model = WhisperForConditionalGeneration.from_pretrained(model_path, cache_dir=model_cache_dir,)
    model.resize_token_embeddings(len(tokenizer))
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False # disable caching


    model.to(device) # move model to GPUs
    #-----------------------------------------------------------------------------------------------------------------------

    #Data Collator for padding 
    speech_laugh_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, 
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    # #--------------------------------------------------------------------------------------------


    #===============================================================================================
    #                           LOAD DATASET AND PROCESSING 
    #===============================================================================================
    # Load the dataset
    print("Loading the dataset as HuggingFace Dataset...")
    switchboard_dataset = load_from_disk(os.path.join(args.dataset_dir, "swb"))
    print(f"Switchboard Dataset: {switchboard_dataset}")
    # Split the dataset into train and validation
    swb_train, swb_eval, swb_test = split_dataset(
        switchboard_dataset, 
        split_ratio=0.8, 
        split="both",
        train_val_split=True,
        val_split_ratio=0.1
    )
    print("Dataset Loaded....\n")
    print(f"Train Dataset (70%): {swb_train}")
    print(f"Validation Dataset (10%): {swb_eval}")
    print(f"Test Dataset (20%): {swb_test}")
    print("------------------------------------------------------")

    """
    Datasets containing the laughter token as <LAUGH> and speechlaugh as uppercase words
    Therefore we need to lowercase all sentences in the dataset, so that:
    - laughter token become <laugh>
    speechlaugh token is lowercase and existing as speech variation

    THIS STEP WOULD BE DONE IN `prepare_dataset`
    """

    # FIND TOTAL LAUGHTER SPEECHLAUGH IN THE SPLITTED DATASET
    total_laugh_train = find_total_laughter_speechlaugh(swb_train)
    total_laugh_val = find_total_laughter_speechlaugh(swb_eval)
    total_laugh_test = find_total_laughter_speechlaugh(swb_test)

    laughter_ratio = (total_laugh_train["laughter"] + total_laugh_val["laughter"]) / total_laugh_test["laughter"]
    speechlaugh_ratio = (total_laugh_train["speechlaugh"] + total_laugh_val["speechlaugh"]) / total_laugh_test["speechlaugh"]
    print(f"Laughter Train/Test ratio: {laughter_ratio}")
    print(f"Speechlaugh Train/Test ratio: {speechlaugh_ratio}")
    print("------------------------------------------------------")

    #===============================================================================================
    #                       DATASET MAPPING TO TENSORS
    #===============================================================================================
    """
    Datasets containing the laughter token as <LAUGH> and speechlaugh as uppercase words
    Therefore we need to lowercase all sentences in the dataset, so that:
    - laughter token become <laugh>
    speechlaugh token is lowercase and existing as speech variation

    THIS STEP WOULD BE DONE IN `prepare_dataset`
    """
    # Prepare Dataset with Batch_size = 1 (single example)
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
        ).input_features[0] #[0] #(n_mels, time_steps)

        batch["labels"] = tokenizer(
            transcript,
        ).input_ids

        return batch

    swb_train = swb_train.map(
        prepare_dataset,
        remove_columns=swb_train.column_names,
        load_from_cache_file=True,
        desc="Preparing Training Dataset",
    )

    swb_eval = swb_eval.map(
        prepare_dataset,
        remove_columns=swb_eval.column_names,
        load_from_cache_file=True,
        desc="Preparing Validation Dataset",
    )

    # Verify dataset size
    print(f"Processed training dataset size: {len(swb_eval)}")
    # Also verify the dataset format
    print("Dataset features:", swb_train.features)

    # ---------------------------------------------------- end of prepare dataset --------------------------------------------


    #===============================================================================================
    #                           COMPUTE METRICS
    #===============================================================================================
    #=================================================================================================
    #                       COMPUTE METRICS 
    #=================================================================================================   
    # Load the WER and F1 metrics
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred, val_loss):
        """
        This function is used to compute the metrics for the model
        on every evaluation step (`eval_steps`) and pass the metrics to the 
        `metrics_callback` for logging to tensorboard and saving to a csv file
        Args:
            pred: predictions from the model
        Returns:
            metrics: dictionary of metrics
        """
        print("Computing metrics...")

        label_ids = pred.label_ids
        pred_ids = pred.predictions

        assert type(label_ids) == type(pred_ids), "Predictions and labels should be the same type"
        assert type(pred_ids) == np.ndarray, "Predictions and labels should be numpy arrays"
        assert type(pred_ids[0]) == np.ndarray, "Predictions and labels should be numpy arrays"


        # Reconstruct the REF and HYP transcripts at Decoder
        ref_transcripts = tokenizer.batch_decode(label_ids, skip_special_tokens=True) #REF transcript, contains laughter tokens [LAUGHTER] and [SPEECH_LAUGH]
        pred_decoded = tokenizer.batch_decode(pred_ids, skip_special_tokens=True) #HYP transcript
        
        pred_transcripts = [transform_number_words(transcript, reverse=True) for transcript in pred_decoded]

        eval_transformation = jiwer.Compose([
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ToLowerCase(),
            jiwer.SubstituteWords({
                "uhhuh": "uh-huh",
                "mmhmm": "um-hum",
                "umhum": "um-hum",
            })
        ])

        # NORMALISED THE TRANSCRIPT
        ref_transcripts = eval_transformation(ref_transcripts) #lowercase
        pred_transcripts = eval_transformation(pred_transcripts) #lowercase

        #-----------------------------------------------------------------------------------------
        wer = 100 * wer_metric.compute(predictions=pred_transcripts, references=ref_transcripts)
        #-----------------------------------------------------------------------------------------


    #     # # METRICS TO CALCULATE -------------------------------------------------

    #     # # alignment = jiwer.process_words(
    #     # #     reference=ref_transcripts, 
    #     # #     hypothesis=pred_transcripts,
    #     # # )
        

    #     # #-----------------------------------------------------------------------------------------
    #     # #               CALCULATE F1 AND TOKEN RATE FOR [laugh] TOKEN MATCH
    #     # #               GO THROUGH EACH PAIR OF SENTENCE AND CALCULATE THE METRICS
    #     # #================================================================================== 

    #     # ref_words = ref_transcripts.split()
    #     # hyp_words = pred_transcripts.split()

    #     # # Get the laughter indices
    #     # eval_laugh_indices = {
    #     #     i: {
    #     #         'word': word,
    #     #         'type': 'laugh', #'laugh' or 'speechlaugh' or 'laugh_intext'
    #     #         'lower': word.lower()
    #     #     }
    #     #     for i, word in enumerate(ref_words)
    #     #     if word == '[laugh]'  #either speech-laugh (word.upper) or laugh (word = [LAUGH])
    #     # }
    #     # token_stat_summary = {
    #     #     'total_TH': 0,
    #     #     'total_TS': 0,
    #     #     'total_TD': 0,
    #     #     'total_TI': 0,
    #     #     'total_token_operations': 0, # total number of token operations in alignment process
    #     # }
    #     # for alignment in alignment.alignments:
    #     #     for chunk in alignment:
    #     #         # Get the aligning  words from reference and hypothesis
    #     #         ref_start, ref_end = chunk.ref_start_idx, chunk.ref_end_idx
    #     #         hyp_start, hyp_end = chunk.hyp_start_idx, chunk.hyp_end_idx

    #     #         #==================================================================================
    #     #         #                           ALIGNMENT CHUNK BY TYPE
    #     #         #==================================================================================
    #     #         if chunk.type == "equal":
    #     #             # If the index of the word 
    #     #             for i, (ref_idx, hyp_idx) in enumerate(zip(range(ref_start, ref_end), 
    #     #                                                     range(hyp_start, hyp_end))):
    #     #                 if ref_idx in eval_laugh_indices:
    #     #                     token_stat_summary['total_TH'] += 1
    #     #         elif chunk.type == "substitute":
    #     #             # Check for substitutions
    #     #             for i, ref_idx in enumerate(range(ref_start, ref_end)):
    #     #                 if ref_idx in eval_laugh_indices:
    #     #                     token_stat_summary['total_TS'] += 1
    #     #         elif chunk.type == "delete":
    #     #             # Check for deletions
    #     #             for ref_idx in range(ref_start, ref_end):
    #     #                 if ref_idx in eval_laugh_indices:
    #     #                     token_stat_summary['total_TD'] += 1
    #     #         elif chunk.type == "insert":
    #     #             # Check for insertions
    #     #             for hyp_idx in range(hyp_start, hyp_end):
    #     #                 if hyp_idx in eval_laugh_indices:
    #     #                     token_stat_summary['total_TI'] += 1

    #     #     #------------------------------------------------------------------------------------------
        
    #     # # CALCULATE F1 AND TOKEN RATE FOR [laugh] TOKEN MATCH
    #     # total_token_operations = token_stat_summary['total_TH'] + token_stat_summary['total_TS'] + token_stat_summary['total_TD']
        
    #     # th_rate = token_stat_summary['total_TH'] / total_token_operations if total_token_operations > 0 else 0
    #     # ts_rate = token_stat_summary['total_TS'] / total_token_operations if total_token_operations > 0 else 0
    #     # td_rate = token_stat_summary['total_TD'] / total_token_operations if total_token_operations > 0 else 0
    #     # ti_rate = token_stat_summary['total_TI'] / total_token_operations if total_token_operations > 0 else 0
    #     # #------------------------------------------------------------------------------------------

    #     # TP = token_stat_summary['total_TH']
    #     # FP = token_stat_summary['total_TS'] + token_stat_summary['total_TI']
    #     # FN = token_stat_summary['total_TD'] + token_stat_summary["total_TS"]
        
    #     # # COMPUTE OVERALL WER, F1
    #     # wer = alignment.wer #FIXME - USING THIS IF WER IS NOT WORKING
    #     # precision = TP / (TP + FP) if TP + FP > 0 else 0
    #     # recall = TP / (TP + FN) if TP + FN > 0 else 0
    #     # f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        metrics = {
            "wer": wer,
            "loss": val_loss,
            # "th": th_rate,
            # "ts": ts_rate,
            # "td": td_rate,
            # "ti": ti_rate,
        }
        print("Evaluated Metrics: ", metrics)

        return metrics

    #===============================================================================================
    #                           TRAINING CONFIGURATION 
    #===============================================================================================

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_output_dir,
        logging_dir=args.log_dir,
        do_train=True,
        do_eval=True,
        
        #Training Configs--------------------------------
        per_device_train_batch_size=4, #4
        gradient_accumulation_steps=8, #4
        learning_rate=5e-5, #or 1e-4
        weight_decay=0.01,
        max_steps=6000, 
        warmup_steps=800,


        # Evaluation Configs--------------------------------
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        eval_accumulation_steps=8,
        eval_steps=1000, #evaluate the model every 1000 steps - Executed compute_metrics()

        # Saving Configs--------------------------------    
        save_steps=1000,
        save_strategy="steps",
        save_total_limit=10, #save the last 10 checkpoints

        # Logging Configs--------------------------------
        logging_steps=100,
        report_to=["tensorboard"], #enable tensorboard for logging
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        resume_from_checkpoint=None,
        
        # Prediction Configs--------------------------------    
        predict_with_generate=True,
        generation_max_length=225,
        #-----------------------------------------------------

        # Computations efficiency--------------------------------
        gradient_checkpointing=True,
        fp16=True, #use mixed precision training
        torch_empty_cache_steps=500,
        #-----------------------------------------------------

        # Dataloader Configs--------------------------------
        dataloader_num_workers=4, #default = 4 - can change larger if possible
        dataloader_pin_memory=True, #use pinned memory for faster data transfer from CPUs to GPUs
        dataloader_persistent_workers=True, #keep the workers alive for multiple training loops
        # push_to_hub=True,   
    ) 
    
    #================================================================
    #                   CALLBACKS FUNCTIONS
    #================================================================
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3, #stop training if the model is not improving
        early_stopping_threshold=0.01 #consider improve if WER decrease by 0.01
    )
    memory_callback = MemoryEfficientCallback()
    metrics_callback = MetricsCallback(
        file_path=os.path.join(args.log_dir, "metrics", "compute_metrics.csv")
    )
    #--------------------------------------------------------------------------------------------
    
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     tokenizer=tokenizer,
    #     train_dataset=train_dataset,
    #     # eval_dataset=test_dataset,
    #     data_collator=speech_laugh_collator,
    #     # compute_metrics=compute_metrics,
    #     callbacks=[
    #         early_stopping, 
    #         memory_callback,
    #         metrics_callback
    #     ]
    # )
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=swb_train,
        eval_dataset=swb_eval, #test_dataset
        data_collator=speech_laugh_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            early_stopping, 
            memory_callback,
            metrics_callback
        ]
    )

    #===============================================================================================
    #                                 MONITOR GPU MEMORY
    #===============================================================================================

    def cleanup_workers():
        """Cleanup function for workers"""
        print("Cleaning CUDA memory cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    #-------------------------------------------------------------------------------------------------


    #===============================================================================================
    #                           TRAINING 
    #===============================================================================================
    try:
        print("Training the model...")
        trainer.train()

    except Exception as e:
        print(f"Error in training: {e}")
        cleanup_workers() #clear the CUDA memory cache
    finally:
        log_history = trainer.state.log_history
        # save the log history to txt file
        with open(os.path.join("../logs", "log_history.txt"), "w") as f:
            for entry in log_history:
                f.write(str(entry) + "\n")
        cleanup_workers() #clear the CUDA memory cache

    # Save the final model
    trainer.save_model(os.path.join(args.model_output_dir, "speechlaugh-fine_tuned", "model"))
    model.save_pretrained(os.path.join(args.model_output_dir, "speechlaugh-fine_tuned"))
    
    print("-----------------------------------------end of training ------------------------------")


    # PUSH MODEL TO HUB ----------------------------------------------------------
    # kwargs = {
    #     "dataset_tags": "hhoangphuoc/switchboard",
    #     "dataset": "Switchboard",
    #     "model_tags": "speechlaugh-whisper-small",
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
    parser.add_argument("--dataset_dir", default="../datasets/switchboard", type=str, required=False, help="Path to the dataset directory")
    #-------------------------------------------------------------------------------------

    # Model Configs
    parser.add_argument("--model_path", default="openai/whisper-large-v2", type=str, required=False, help="Select pretrained model")
    parser.add_argument("--pretrained_model_dir", default="../ref_models/pre_trained/", type=str, required=False, help="Name of the model")
    parser.add_argument("--model_output_dir", default="../vocalwhisper/speechlaugh-whisper-large-v2/", type=str, required=False, help="Path to the output directory")
    parser.add_argument("--log_dir", default="../logs", type=str, required=False, help="Path to the log directory")
    parser.add_argument("--evaluate_dir", default="../evaluate", type=str, required=False, help="Path to the evaluation directory")
    #------------------------------------------------------------------------------
    args = parser.parse_args()
    try:
        SpeechLaughWhisper(args)
    except OSError as error:
        print(error)


