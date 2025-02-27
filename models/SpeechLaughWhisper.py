from collections import Counter
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
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
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
    #                                           PATH CONFIGURATION 
    #-------------------------------------------------------------------------------------------------  
    model_path = args.model_path
    print(f"Model Path: {model_path}")
    model_cache_dir = args.pretrained_model_dir
    print(f"Pretrained Model Directory: {args.pretrained_model_dir}\n")

    output_dir = os.path.join(args.checkpoint_dir, args.checkpoint_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Training Checkpoint Directory: {output_dir} \n")


    #--------------------------------------------------------------------------------------------------------------------------------
    #                                           MODEL CONFIGURATION
    #--------------------------------------------------------------------------------------------------------------------------------
    #Processor and Tokenizer
    processor = WhisperProcessor.from_pretrained(model_path, cache_dir=model_cache_dir) # processor - combination of feature extractor and tokenizer
    # feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path, cache_dir=model_cache_dir) #feature extractor
    
    model = WhisperForConditionalGeneration.from_pretrained(model_path, cache_dir=model_cache_dir)
    
    # tokenizer = WhisperTokenizer.from_pretrained(model_path, cache_dir=model_cache_dir) #tokenizer
    # FIXME: Declare the Feature Extractor and Tokenizer from the Processor class------------------------
    feature_extractor = processor.feature_extractor

    tokenizer = processor.tokenizer
    #-----------------------------------------------------------------------------------------------------
    # Add special tokens for laughter 
    # - FIXME: REMOVED THIS PART IF WORKING WITH FINETUNED NO-LAUGH

    new_tokens = ['<laugh>']
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    laugh_token_id = tokenizer.convert_tokens_to_ids('<laugh>')
    print(f"The token ID for '<laugh>' is: {laugh_token_id}") # WE FOUND THAT <laugh> TOKEN ID is 51865
    #-----------------------------------------------------------------------------------------------------

    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_length = 448
    model.generation_config.suppress_tokens = [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362]
    model.generation_config.begin_suppress_tokens = [220, 50257]
    model.generation_config.save_pretrained(os.path.join(output_dir, "generation_config"))

    # TODO: Save processor directly to output directory as it not change over time
    processor.save_pretrained(os.path.join(output_dir, "processor"))

    model.config.use_cache = False # disable caching

    model.to(device) # move model to GPUs

    #-------------------------------------------------------------------------------------------------------------------------------------------------
    
    #Data Collator for padding 
    speech_laugh_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, 
        decoder_start_token_id=model.config.decoder_start_token_id,
    )


    #===============================================================================================
    #                           LOAD DATASET AND PROCESSING 
    #===============================================================================================
    
    # ================= Load from splitted dataset ==========================
    data_train = load_from_disk(os.path.join(args.dataset_dir, "buckeye_train"))
    data_eval = load_from_disk(os.path.join(args.dataset_dir, "buckeye_eval"))
    data_test = load_from_disk(os.path.join(args.dataset_dir, "buckeye_test"))

    print("Dataset Loaded....\n")
    print(f"Train Dataset (70%): {data_train}")
    print(f"Validation Dataset (10%): {data_eval}")
    print(f"Test Dataset (20%): {data_test}")
    print("------------------------------------------------------")

    #--------------------------------------------------------------------------------------------------------------------------
    # Remove empty transcripts
    data_train = data_train.filter(lambda x: len(x["transcript"]) > 0, desc="Filtering out empty transcript in Train dataset")
    print("Train Dataset (after filtered):", data_train)
    data_eval = data_eval.filter(lambda x: len(x["transcript"]) > 0, desc="Filtering out empty transcript in Eval dataset")
    print("Validation Dataset (after filtered):", data_eval)
    #--------------------------------------------------------------------------------------------------------------------------



    #--------------------------------------------------------------------------------------------------------------------------
    # FIXME: UNCOMMENT THIS PART IF WORKING WITH FINETUNED NOLAUGH

    #=========================================================================================================================
    #REMOVE LAUGHTER IN TRAIN
    data_train = data_train.map(lambda x: {'transcript': x['transcript'].replace('<LAUGH>', '')}, desc="Removing <LAUGH> for NOLAUGH finetuning in Train dataset")
    data_train = data_train.filter(lambda x: len(x["transcript"]) > 0 and x["transcript"] !="<LAUGH>", desc="Filtering out <LAUGH only> and empty transcript in Eval dataset")
    print("Train Dataset (after filtered):", data_train)

    #REMOVE LAUGHTER IN EVAL
    data_eval = data_eval.map(lambda x: {'transcript': x['transcript'].replace('<LAUGH>', '').strip()}, desc="Removing <LAUGH> for NOLAUGH finetuning in Eval dataset")
    data_eval = data_eval.filter(lambda x: len(x["transcript"]) > 0 and x["transcript"] !="<LAUGH>", desc="Filtering out <LAUGH only> and empty transcript in Eval dataset")
    print("Validation Dataset (after filtered):", data_eval)
    print("------------------------------------------------------")
    #=========================================================================================================================



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

        labels = tokenizer(
            transcript,
        ).input_ids

        batch["labels"] = labels

        return batch

    data_train = data_train.map(
        prepare_dataset,
        remove_columns=data_train.column_names,
        # load_from_cache_file=True,
        desc="Preparing Training Dataset",
    )

    data_eval = data_eval.map(
        prepare_dataset,
        remove_columns=data_eval.column_names,
        # load_from_cache_file=True,
        desc="Preparing Validation Dataset",
    )

    # Verify dataset size
    print(f"Processed training dataset size: {len(data_eval)}")
    # Also verify the dataset format
    print("Dataset features:", data_train.features)

    # ---------------------------------------------------- end of prepare dataset --------------------------------------------


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

        # check if the certain label_id of <laugh> token is in the label_ids

        pred_ids = pred.predictions

        # Reconstruct the REF and HYP transcripts at Decoder
        ref_decoded = tokenizer.batch_decode(label_ids, skip_special_tokens=True) #REF transcript, contains laughter tokens [LAUGHTER] and [SPEECH_LAUGH]
        pred_decoded = tokenizer.batch_decode(pred_ids, skip_special_tokens=True) #HYP transcript
        
        #-----------------------------------------------------------------------------------------
        # THIS CODE TO ENSURE  THE TRANSCRIPTS ARE NOT EMPTY
        # ref_transcripts, pred_transcripts = [], []

        # for i, (ref, pred) in enumerate(zip(ref_decoded, pred_decoded)):
        #     if not pred.strip() or not ref.strip():
        #         print("Empty transcript! Skipping...")
        #         continue
        #     ref_transcripts.append(ref)
        #     pred_transcripts.append(pred)

        # assert len(ref_transcripts) == len(pred_transcripts), "The number of transcripts should be equal!"
        #--------------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------
        #                   NORMALISED THE TRANSCRIPT
        #-----------------------------------------------------------------------------------------
        eval_transformation = jiwer.Compose([
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemovePunctuation(),
            jiwer.SubstituteWords({
                "uhhuh": "uh-huh",
                "uh huh": "uh-huh",
                "mmhmm": "um-hum",
                "mm hmm": "um-hum",
                "mmhum": "um-hum",
                "mm hum": "um-hum",
                "umhum": "um-hum",
                "um hum": "um-hum",
                "umhmm": "um-hum",
            }),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ToLowerCase()
        ])

        # NORMALISED THE TRANSCRIPT
        ref_decoded = eval_transformation(ref_decoded) #lowercase
        pred_decoded = eval_transformation(pred_decoded) #lowercase

        #-----------------------------------------------------------------------------------------
        wer = wer_metric.compute(predictions=pred_decoded, references=ref_decoded) #*100 - NOT USE x100, observing range from 0-1

        # wer = jiwer.wer(reference=ref_transcripts, hypothesis=pred_transcripts)
        #-----------------------------------------------------------------------------------------
        metrics = {
            "wer": wer,
            "loss": val_loss,
        }
        print("Evaluated Metrics: ", metrics)

        return metrics

    #===============================================================================================
    #                           TRAINING CONFIGURATION 
    #===============================================================================================

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,

        # max_steps=6000, #6000 steps shows good results - try  8000 steps with larger batch size, smaller lr (LONGER TRAINING :(()))
        # warmup_steps=1000, #warmup at longer steps for effectively learning the existence of <laugh> token
        num_train_epochs=10, #10 epochs - FIXME: (10 epochs for switchboard) || 3 epochs for buckeye
        warmup_ratio=0.15, #USE 10-15% of the total steps for warmup to better learn the <laugh> token

        
        #Training Configs--------------------------------
        per_device_train_batch_size=16, #16 - FIXME: or change to 16 and `accumulate the gradients` to 16
        gradient_accumulation_steps=4, #8


        # Evaluation Configs--------------------------------
        # eval_strategy="steps",
        # eval_steps=500, #evaluate the model every 1000 steps - Executed compute_metrics()
        
        eval_strategy="epoch", #evaluate the model every epoch
        per_device_eval_batch_size=8, #8 FIXME: Use larger batch size for evaluation (improved WER, loss)
        eval_accumulation_steps=4, #try with 8 to faster evaluation
        


        learning_rate=1e-4, #or 5e-5 - FIXME: `1e-5` is TOO SMALL, MAKE WHISPER UNABLE TO CONVERGE -> Try: `5e-5` or `1e-4`
        lr_scheduler_type="linear", # Use linear scheduler for learning rate
        weight_decay=0.005,#TRY SMALLER WEIGHT DECAY (0.005) instead of `0.01`- FIXME:Larger weight decay (0.05) making it harder to converge, TRY: smaller weight decay (0.001)


        # Saving Configs--------------------------------    
        # save_steps=500,
        # save_strategy="steps",
        save_strategy="epoch", #save the model every epoch
        save_total_limit=3, #save the last 2 checkpoints

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
        generation_max_length=448,
        #-----------------------------------------------------

        # Computations efficiency--------------------------------
        gradient_checkpointing=True,
        fp16=True, #use mixed precision training
        adam_beta2=0.98,
        torch_empty_cache_steps=1000,
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
        early_stopping_patience=3, #stop training if the model is not improving when 3 times validation loss not decrease
        early_stopping_threshold=0.01 #consider improve if WER decrease by 0.01
    )
    memory_callback = MemoryEfficientCallback()

    # ==================================================================================================================
    # SET UP `OPTIMIZER` AND `LEARNING RATE SCHEDULER` FOR BETTER LEARNING <laugh> TOKEN
    # ==================================================================================================================
    print(f"Training arguments:\n learning_rate = {training_args.learning_rate};\n weight_decay = {training_args.weight_decay}; \n Epochs: {training_args.num_train_epochs}; \n Batch Size: {training_args.per_device_train_batch_size} (accumulated with: {training_args.gradient_accumulation_steps}); \n Warmup Ratio: {training_args.warmup_ratio}")

    # optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    # lr_scheduler = get_linear_schedule_with_warmup( # or get_cosine_schedule_with_warmup
    #     optimizer=optimizer,
    #     num_warmup_steps=training_args.warmup_steps,
    #     num_training_steps=num_training_steps,
    # )
    #===================================================================================================================

    
    # TRAINER
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=data_train,
        eval_dataset=data_eval, #test_dataset
        data_collator=speech_laugh_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            early_stopping, 
            memory_callback,
            MetricsCallback(
                file_path=f"../logs/whisper/buckeye/validation_{args.checkpoint_id}.csv"
            )
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
        with open(os.path.join("../logs/whisper/buckeye", f"train_log_{args.checkpoint_id}.txt"), "w") as f:
            for entry in log_history:
                f.write(str(entry) + "\n")
        cleanup_workers() #clear the CUDA memory cache

    # Save the final model, processor, and tokenizer
    model.save_pretrained(os.path.join(args.model_output_dir, "model"))
    processor.save_pretrained(os.path.join(args.model_output_dir, "processor"))
    
    print("-----------------------------------------end of training ------------------------------")


    # PUSH MODEL TO HUB ----------------------------------------------------------
    # kwargs = {
    #     "dataset_tags": "hhoangphuoc/switchboard",
    #     "dataset": "Switchboard",
    #     "model_tags": "hhoangphuoc/speechlaugh-whisper-large-v2",
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
    parser.add_argument("--model_output_dir", default="../fine-tuned/speechlaugh-whisper-large-v2/", type=str, required=False, help="Directory to the checkpoints")
    parser.add_argument("--checkpoint_dir", default="../checkpoints/whisper", type=str, required=False, help="Directory to all the saved checkpoints during training of specific configuration")
    parser.add_argument("--checkpoint_id", default="speechlaugh-whisper-10epochs", type=str, required=False, help="Checkpoint ID - Name of model configs")
    parser.add_argument("--log_dir", default="../logs", type=str, required=False, help="Path to the log directory")
    parser.add_argument("--evaluate_dir", default="../evaluate", type=str, required=False, help="Path to the evaluation directory")
    #------------------------------------------------------------------------------
    args = parser.parse_args()
    try:
        SpeechLaughWhisper(args)
    except OSError as error:
        print(error)


