import argparse
import os
import sys
import torch
import torchaudio
from datasets import Dataset
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import pandas as pd
import librosa

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
    # MODEL CONFIGS
    # Load the fine-tuned Whisper model and processor
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
    processor = WhisperProcessor.from_pretrained(args.model_path)

    tokenizer = WhisperTokenizer.from_pretrained(args.model_path)
    special_tokens = ["[LAUGHTER]", "[COUGH]", "[SNEEZE]", "[THROAT-CLEARING]", "[SIGH]", "[SNIFF]", "[UH]", "[UM]", "[MM]", "[YEAH]", "[MM-HMM]"]
    tokenizer.add_tokens(special_tokens)
    # model.config.decoder_start_token_id = tokenizer.get_vocab()["<|startoftext|>"]
    # model.config.decoder_end_token_id = tokenizer.get_vocab()["<|endoftext|>"]

    model.resize_token_embeddings(len(tokenizer))
    #----------------------------------------------------------


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
        per_device_train_batch_size=args.batch_size, #16
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        fp16=True,
        eval_strategy="steps",
        data_loader_num_workers=8,
        logging_steps=25,
        save_steps=1000,
        report_to="tensorboard"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    pass
    parser = argparse.ArgumentParser(description="Speech Laugh Recognition")
    parser.add_argument("--input_file_path", default="../datasets/train.csv", type=str, required=False, help="Path to the train.csv file")
    parser.add_argument("--model_path", default="openai/whisper-large-v3", type=str, required=False, help="Select pretrained model")
    parser.add_argument("--model_output_dir", default="./speechlaugh-whisper-fine-tuned", type=str, required=False, help="Path to the output directory")
    parser.add_argument("--batch_size", default=256, type=int, required=False, help="Batch size for training")
    parser.add_argument("--num_train_epochs", default=3, type=int, required=False, help="Number of training epochs")
    parser.add_argument("--warmup_steps", default=800, type=int, required=False, help="Number of warmup steps")
    parser.add_argument("--lr", default=1e-5, type=float, required=False, help="Learning rate for training")
    #----------------------------------------------------------
    args = parser.parse_args()

    result = SpeechLaughWhisper(args)

