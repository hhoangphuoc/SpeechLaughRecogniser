import argparse
import os
import sys
import torch
import torchaudio
import pandas as pd
import librosa
import numpy as np
from huggingface_hub import notebook_login
from datasets import Dataset, load_dataset, DatasetDict
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback

from model.SpeechLaughDataCollator import DataCollatorSpeechSeq2SeqWithPadding
from eval import compute_metrics
#----------------------------------------------------------

"""
This is the fine-tuning Whisper model 
for the specific task of transcribing recognizing speech laughter, fillers, 
pauses, and other non-speech sounds in conversations.
"""
notebook_login()

def SpeechLaughWhisper(args):
    """
    This function is used to recognize speech laughter, fillers, 
    pauses, and other non-speech sounds in conversations.
    Args:
        df: pandas dataframe - contains all audios and transcripts

    """
    # MODEL CONFIGS
    #----------------------------------------------------------
    #Processor and Tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_path) #feature extractor
    processor = WhisperProcessor.from_pretrained(args.model_path) # processor - combination of feature extractor and tokenizer

   

    tokenizer = WhisperTokenizer.from_pretrained(args.model_path) #tokenizer
    special_tokens = ["[LAUGHTER]", "[COUGH]", "[SNEEZE]", "[THROAT-CLEARING]", "[SIGH]", "[SNIFF]", "[UH]", "[UM]", "[MM]", "[YEAH]", "[MM-HMM]"]
    tokenizer.add_tokens(special_tokens)
    # model.config.decoder_start_token_id = tokenizer.get_vocab()["<|startoftext|>"]
    # model.config.decoder_end_token_id = tokenizer.get_vocab()["<|endoftext|>"]

    # Load the fine-tuned Whisper model
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path)

    model.resize_token_embeddings(len(tokenizer))
    #----------------------------------------------------------
    #Data collator for random noise
    speech_laugh_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, padding=True)


    def prepare_dataset(batch):
        audio = batch["audio"]
        transcript = batch["transcript"] #this audio and transcript are merged from all the dataset

        #TODO: Add the transcript to the batch
        batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features.numpy()
        batch["labels"] = processor(text=transcript, return_tensors="pt").input_ids
        # batch["input_features"] = input_features
        # batch["labels"] = input_ids
        return batch

    #----------------------------------------------------------
    df = pd.read_csv(args.input_file_path) #datasets/train.csv
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
    
    
    #----------------------------------------------------------
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_output_dir,
        per_device_train_batch_size=args.batch_size, #8 - default batch size = 8, could add up to 256 based on the GPU max memory
        gradient_accumulation_steps=4, #increase the batch size by accumulating gradients (add 4 per cycle)
        learning_rate=args.lr, #1e-5
        num_train_epochs=args.num_train_epochs, #default = 3 - change between 2 -5 based on overfitting
        warmup_steps=args.warmup_steps, #800
        fp16=True, #use mixed precision training
        eval_strategy="steps",
        data_loader_num_workers=24, #default = 24 - can change to 16 if the GPU has less memory, now compatible with 72 cores GPU
        logging_steps=25,
        save_steps=1000,
        eval_steps=1000,
        report_to=["tensorboard"], #enable tensorboard for logging
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
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
        train_dataset=dataset,
        eval_dataset=dataset, #TODO- create validation dataset for evaluation instead
        data_collator=speech_laugh_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )
    
    # trainer.train()

    # Training loop with TensorBoard logging
    for epoch in range(int(training_args.num_train_epochs)):
        trainer.train()
        # Log metrics to TensorBoard after each epoch
        metrics = trainer.evaluate()
        for key, value in metrics.items():
            writer.add_scalar(f"eval/{key}", value, epoch)
    writer.close()  # Close the TensorBoard writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Laugh Recognition")
    parser.add_argument("--input_file_path", default="../datasets/train.csv", type=str, required=False, help="Path to the train.csv file")
    parser.add_argument("--model_path", default="openai/whisper-medium", type=str, required=False, help="Select pretrained model")
    parser.add_argument("--model_output_dir", default="../vocalwhisper/vocalspeech-whisper-medium", type=str, required=False, help="Path to the output directory")
    parser.add_argument("--log_dir", default="./log", type=str, required=False, help="Path to the log directory")
    parser.add_argument("--batch_size", default=8, type=int, required=False, help="Batch size for training")
    parser.add_argument("--num_train_epochs", default=3, type=int, required=False, help="Number of training epochs")
    parser.add_argument("--num_workers", default=24, type=int, required=False, help="number of workers to use for data loading, can change based on the number of cores")
    parser.add_argument("--warmup_steps", default=800, type=int, required=False, help="Number of warmup steps")
    parser.add_argument("--lr", default=1e-5, type=float, required=False, help="Learning rate for training")
    #----------------------------------------------------------
    args = parser.parse_args()

    result = SpeechLaughWhisper(args)

