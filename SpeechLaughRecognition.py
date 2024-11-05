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
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
#import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel   #for distributed training
from datasets import load_from_disk
from huggingface_hub import login
from modules.SpeechLaughDataCollator import DataCollatorSpeechSeq2SeqWithPadding
#----------------------------------------------------------

from preprocess import split_dataset
import utils.params as prs

# Evaluation Metrics------
import pandas as pd
import evaluate
import jiwer
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
#----------------------------------------------------------

# Set the path for pre-trained model

#----------------------------------------------------------

def SpeechLaughWhisper(args):
    """
    This function is used to recognize speech laughter, fillers, 
    pauses, and other non-speech sounds in conversations.
    Args:
        df: pandas dataframe - contains all audios and transcripts

    """
    # MODEL CONFIGS ----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    #----------Processor and Tokenizer-----------
    processor = WhisperProcessor.from_pretrained(args.model_path, cache_dir=args.pretrained_model_dir) # processor - combination of feature extractor and tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_path, cache_dir=args.pretrained_model_dir) #feature extractor
    tokenizer = WhisperTokenizer.from_pretrained(args.model_path, cache_dir=args.pretrained_model_dir) #tokenizer
    # special_tokens = ["[LAUGHTER]", "[COUGH]", "[SNEEZE]", "[THROAT-CLEARING]", "[SIGH]", "[SNIFF]", "[UH]", "[UM]", "[MM]", "[YEAH]", "[MM-HMM]"]
    special_tokens = ["[LAUGHTER]", "[SPEECH_LAUGH]"] #FIXME: Currently, only laughter and speech_laugh are used
    tokenizer.add_tokens(special_tokens)
    #-------------------------------------------

    # Pre-trained Model Loading ----------------
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path, cache_dir=args.pretrained_model_dir)
    model.resize_token_embeddings(len(tokenizer))
    model.generation_config.forced_decoder_ids = None
    model.to(device) # move model to GPUs
    #--------------------------------------------

    #Data Collator for random noise --------------
    speech_laugh_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, 
        decoder_start_token_id=model.config.decoder_start_token_id,
        device=device,
        padding=True,
    )
    #----------------------------------------------------------
 
    def prepare_dataset(examples):
        # ---------------------------------------------------------------------------------
        # NOTES: IN THIS PREPARE_DATASET, THE BATCH IS COMPUTED AND STORED IN CPU
        #---------------------------------------------------------------------------------
        """
        Batched Dataset Format
        Examples:  {'audio': [
        {'path': '/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data/audio_segments/sw03111A_196534375_200339875.wav', 'array': array([-0.00018135, -0.00039364, -0.00035248, ..., -0.00202596,
        0.00115121,  0.00038809]), 'sampling_rate': 16000}, 
        {'path': '/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data/audio_segments/sw03111A_200339875_204145375.wav', 'array': array([-0.00018135, -0.00039364, -0.00035248, ..., -0.00202596,
        0.00115121,  0.00038809]), 'sampling_rate': 16000},
        ...
        ],
        'transcript': [
        'there is [A] lot in the society where things have changed', 
        "now [I] guess that's that that's become less so [I] noted that [UM] [UH] the last couple times that [I] had to go by jury they've actually selected intelligent jurors"
        ...
        ]}

        RETURN:
        Batched Dataset Format
        Examples: {
        'input_features': [
        array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ...,
        ...),
        array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ...,
        ...),
        ],
        'labels': [
        array([  0,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,..),
        array([  0,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,...),
        ]}

        Single Example format:
        {
            "audio": {
                "path": "path/to/audio",
                "array": [1, 2, 3, ...]
            },
            "sampling_rate": 16000,
            "transcript": "..."
        }

        RETURN:
        Example format: {
            "input_features": [1, 2, 3, ...],
            "labels": [1, 2, 3, ...]
        }
        """
        batch = {
            "input_features": [],
            "labels": []
        }
        # chunk_size = 32 # FIXME: Processing in chunks to reduce memory usage
        with torch.no_grad():  # disable gradient computation when processing the data
            # Pre-allocate the audio arrays for entire batch
            # audio_arrays = [
            #     np.array(audio["array"]).squeeze()
            #     for audio in examples["audio"]
            #     if audio is not None
            # ]
            # for i in range(0, len(audio_arrays), chunk_size):
                # if i % (chunk_size * 2) == 0:
                #     memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                #     print(f"Memory usage at chunk {i}: {memory:.2f} MB")
                #     gc.collect()  # Force garbage collection
            #     # example_audio = examples["audio"][i:i+chunk_size]
            #     # example_transcript = examples["transcript"][i:i+chunk_size]
            #     example_audio = audio_arrays[i:i+chunk_size]
            #     example_transcript = examples["transcript"][i:i+chunk_size]

            #     if example_audio is None or example_transcript is None:
            #         continue
            for i in range(len(examples["audio"])):
                example_audio = examples["audio"][i]
                example_transcript = examples["transcript"][i]

                if example_audio is None or example_transcript is None:
                    continue
            
                audio = np.array(example_audio["array"]).squeeze()
                
                # sampling_rate = example_audio["sampling_rate"]

                # Calculate the input features - FIXME: NOT USING GPUs due to inefficiency
                try:
                    input_features = feature_extractor( 
                        audio,
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True,
                    ).input_features
            
                    labels = tokenizer(
                        example_transcript, 
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).input_ids

                    batch["input_features"].extend(input_features.numpy())
                    batch["labels"].extend(labels.numpy())
                except Exception as e:
                    print(f"Error in processing example {i}: {e}")
                    continue
        return batch

    
    #COMPUTE METRICS -------------------------------------------------- 
    # Evaluation Metrics --------------------------------------------------
    # Load the WER metric
    wer_metric = evaluate.load("wer", cache_dir=args.evaluate_dir) #Word Error Rate between the hypothesis and the reference transcript
    f1_metric = evaluate.load("f1", cache_dir=args.evaluate_dir) #F1 score between the hypothesis and the reference transcript
    exact_match_metric = evaluate.load("exact_match", cache_dir=args.evaluate_dir) #compare the exact match between the hypothesis and the reference transcript
    #----------------------------------------------------------   
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
        exact_match = 100 * exact_match_metric.compute(predictions=pred_transcripts, references=ref_transcripts)

        # Visualise the alignment between the HYP and REF transcript
        alignments = jiwer.process_words(
            reference=ref_transcripts, 
            hypothesis=pred_transcripts, 
            reference_transform=output_transform,
            hypothesis_transform=output_transform
        )
        
        #write the alignment to the text file
        with open("alignment_transcripts/alignment_speechlaugh_whisper.txt", "w") as f:
            report_alignment = jiwer.visualize_alignment(
                alignments,
                show_measures=True, 
                skip_correct=False
                )
            f.write(report_alignment)

        return {
            "wer": wer,
            "f1": f1,
            "exact_match": exact_match
            }
    
    #-----------------------------------------------------END OF MODEL CONFIG ---------------------------------------------

    # --------------------------------------------------------------------------------------------
    #                           LOAD DATASET AND PROCESSING 
    # --------------------------------------------------------------------------------------------
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
    
    #--------------------------------------------------------
    #               PREPARE DATASET 
    #--------------------------------------------------------
    train_dataset = train_dataset.map(
        prepare_dataset,
        batched=True,
        batch_size=16,
        remove_columns=train_dataset.column_names,
        # load_from_cache_file=False,
        load_from_cache_file=True,
        desc="Preparing Training Dataset",
    )
    test_dataset = test_dataset.map(
        prepare_dataset,
        batched=True,
        batch_size= 4,
        remove_columns=test_dataset.column_names,
        # load_from_cache_file=False,
        load_from_cache_file=True,
        desc="Preparing Validation Dataset",
    )
    # ---------------------------------------------------- end of prepare dataset --------------------------------------------


    # ---------------------------------------------------------
    #  TRAINING CONFIGURATION 
    # ---------------------------------------------------------
    # Data Parallel for distributed training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = DistributedDataParallel(model)

    
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_output_dir,
        per_device_train_batch_size=args.batch_size, #16 - default batch size = 16, could add up to 256 based on the GPU max memory
        gradient_accumulation_steps=args.grad_steps, # - default = 4 increase the batch size by accumulating gradients
        learning_rate=args.lr, #1e-5
        weight_decay=0.01,
        # num_train_epochs=args.num_train_epochs, #default = 2 - change between 2 -5 based on overfitting
        max_steps=args.max_steps, #default = 5000 - change based on the number of epochs
        warmup_steps=args.warmup_steps, #800
        logging_dir=args.log_dir,
        gradient_checkpointing=True,
        fp16=True, #use mixed precision training
        eval_strategy="steps",
        per_device_eval_batch_size=4,
        dataloader_num_workers=args.num_workers, #default = 16 - can change max to 32
        dataloader_pin_memory=True, #use pinned memory for faster data transfer from CPUs to GPUs
        # dataloader_prefetch_factor=2, #FIXME: added for better GPU utilisation-
        logging_steps=25,
        save_steps=1000,
        save_strategy="steps",
        save_total_limit=5, #save the last 5 checkpoints
        eval_steps=1000,
        report_to=["tensorboard"], #enable tensorboard for logging
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        resume_from_checkpoint=None,
        # push_to_hub=True,   
    )

    writer = SummaryWriter(log_dir=args.log_dir)

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3, #stop training if the model is not improving
        early_stopping_threshold=0.01 #consider improve if WER decrease by 0.01
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=speech_laugh_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )


    #----------------------------------------------------------------------
    #                           TRAINING 
    #----------------------------------------------------------------------
    # Create DataFrames to store the training and evaluation metrics
    training_metrics = pd.DataFrame(columns=["step", "training_loss", "epoch", "validation_loss","wer", "f1", "exact_match"])

    # Training loop with TensorBoard logging and saving model per 1000 steps
    for step in range(training_args.max_steps):
        trainer.train()
        if step % args.save_steps == 0:
            trainer.save_model(args.model_output_dir + f"speechlaugh_whisper_{str(step)}.bin")
            
            metrics = trainer.evaluate() #return evaluation metrics: {"wer": wer, "f1": f1, "exact_match": exact_match}
            
            training_metrics = training_metrics.append({
                "step": step,
                "training_loss": trainer.state.loss if trainer.state.loss is not None else 0,
                "epoch": trainer.state.epoch,
                "validation_loss": trainer.state.eval_loss if trainer.state.eval_loss is not None else 0,
                "wer": metrics["wer"],
                "f1": metrics["f1"],
                "exact_match": metrics["exact_match"]
            }, ignore_index=True)
            # Log metrics to TensorBoard after each epoch
            for key, value in metrics.items():
                writer.add_scalar(f"eval/{key}", value, step)
            writer.add_scalar("training_loss", trainer.state.global_step, trainer.state.loss)
            writer.add_scalar("validation_loss", trainer.state.global_step, trainer.state.eval_loss)
            writer.add_scalar("WER", trainer.state.global_step, trainer.state.eval_metrics["wer"])
            writer.add_scalar("F1", trainer.state.global_step, trainer.state.eval_metrics["f1"])
            writer.flush()
            #---------- end of logging -------------------------

        if trainer.state.global_step >= training_args.max_steps:
            break

    # Save the training metrics to a CSV file
    training_metrics.to_csv(args.model_output_dir + "training_metrics.csv", index=False)

    writer.close()  # Close the TensorBoard writer

    # Save the final model
    model.save_pretrained(args.model_output_dir + "model")
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
    parser.add_argument("--batch_size", default=16, type=int, required=False, help="Batch size for training")
    parser.add_argument("--grad_steps", default=2, type=int, required=False, help="Number of gradient accumulation steps, which increase the batch size without extend the memory usage")
    # parser.add_argument("--num_train_epochs", default=2, type=int, required=False, help="Number of training epochs")
    parser.add_argument("--num_workers", default=16, type=int, required=False, help="number of workers to use for data loading, can change based on the number of cores")
    parser.add_argument("--warmup_steps", default=800, type=int, required=False, help="Number of warmup steps")
    parser.add_argument("--save_steps", default=1000, type=int, required=False, help="Number of steps to save the model")
    parser.add_argument("--max_steps", default=5000, type=int, required=False, help="Maximum number of training steps")
    parser.add_argument("--lr", default=1e-5, type=float, required=False, help="Learning rate for training")
    #------------------------------------------------------------------------------
    args = parser.parse_args()

    load_dotenv()

    try:
        # login(token=os.getenv("HUGGINGFACE_TOKEN"))
        start_time = time.time()
        SpeechLaughWhisper(args)
        print(f"Total Processing time: {time.time() - start_time:.2f} seconds")
    except OSError as error:
        print(error)


