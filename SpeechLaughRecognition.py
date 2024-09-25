import argparse
import os
import sys
import torch
# import torchaudio
import librosa
import soundfile as sf
import pandas as pd

import numpy as np
# from huggingface_hub import notebook_login
from datasets import Dataset
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
#import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from modules.SpeechLaughDataCollator import DataCollatorSpeechSeq2SeqWithPadding

import evaluate
#----------------------------------------------------------

"""
This is the fine-tuning Whisper model 
for the specific task of transcribing recognizing speech laughter, fillers, 
pauses, and other non-speech sounds in conversations.
"""
# notebook_login()

metric = evaluate.load("wer")

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

    # Load the fine-tuned Whisper model
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))
    #----------------------------------------------------------
    #Data collator for random noise
    speech_laugh_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, padding=True)

    def load_dataset(csv_input_path):
        train_df = pd.read_csv(csv_input_path)

        train_df["sampling_rate"] = train_df["sampling_rate"].apply(lambda x: int(x))

        train_df = train_df[train_df["audio"].apply(lambda x: len(x) > 0)]
        train_df = train_df[train_df["transcript"].apply(lambda x: len(x) > 0)]
        
        #shuffle the dataframe
        train_df = train_df.sample(frac=1).reset_index(drop=True)

        train_dataset = Dataset.from_pandas(train_df)
        return train_dataset
        
    def prepare_dataset(example):
        #TODO: FIX THIS TO CONVERT THE STRING FORMAT OF ARRAY TO ACTUAL NUMPY ARRAY
        
        audio_path = os.path.abspath(example["audio"]) #using os to open the relative path

        # audio, sampling_rate = sf.read(audio_path, dtype='float32', samplerate=16000) #load the audio file and resample to 16kHz
        audio, sampling_rate = librosa.load(audio_path, sr=16000)
        # 2. Resample if necessary
        # if sampling_rate != 16000:
        #     audio, sampling_rate = librosa.resample(y=audio, orig_sr=sampling_rate, target_sr=16000)

        example["audio"] = audio.squeeze() #convert to suitable audio array
        example["sampling_rate"] = sampling_rate

        # #TODO: Add the transcript to the batch
        example["input_features"] = processor(example["audio"], sampling_rate=example["sampling_rate"], return_tensors="pt").input_features.numpy()
        example["labels"] = processor(text=example["transcript"], return_tensors="pt").input_ids
        return example
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    #----------------------------------------------------------
    # train_df = pd.read_csv(args.input_file_path) #datasets/train.csv
    train_dataset = load_dataset(args.input_file_path)
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
    
    # if args.eval_file_path is not None :
    eval_dataset = load_dataset(args.eval_file_path)
    eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names)
    #----------------------------------------------------------

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_output_dir,
        per_device_train_batch_size=args.batch_size, #4 - default batch size = 4, could add up to 16 based on the GPU max memory
        gradient_accumulation_steps=args.grad_steps, #increase the batch size by accumulating gradients (add 8 per cycle), could change to 2
        learning_rate=args.lr, #1e-5
        num_train_epochs=args.num_train_epochs, #default = 3 - change between 2 -5 based on overfitting
        warmup_steps=args.warmup_steps, #800
        fp16=True, #use mixed precision training
        eval_strategy="steps",
        data_loader_num_workers=args.num_workers, #default = 16 - can change to 24 if the GPU has less memory, now compatible with 72 cores GPU
        logging_steps=25,
        save_steps=1000,
        eval_steps=1000,
        report_to=["tensorboard"], #enable tensorboard for logging
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
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
        # eval_dataset=dataset, #TODO- create validation dataset for evaluation instead
        eval_dataset=eval_dataset,
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
    parser.add_argument("--input_file_path", default="./datasets/train.csv", type=str, required=False, help="Path to the train.csv file")
    parser.add_argument("--eval_file_path", default="./datasets/val.csv", type=str, required=False, help="Path to the val.csv file")
    parser.add_argument("--model_path", default="openai/whisper-medium", type=str, required=False, help="Select pretrained model")
    parser.add_argument("--model_output_dir", default="./vocalwhisper/vocalspeech-whisper-medium", type=str, required=False, help="Path to the output directory")
    parser.add_argument("--log_dir", default="./log", type=str, required=False, help="Path to the log directory")
    parser.add_argument("--batch_size", default=2, type=int, required=False, help="Batch size for training")
    parser.add_argument("--grad_steps", default=8, type=int, required=False, help="Number of gradient accumulation steps, which increase the batch size without extend the memory usage")
    parser.add_argument("--num_train_epochs", default=3, type=int, required=False, help="Number of training epochs")
    parser.add_argument("--num_workers", default=16, type=int, required=False, help="number of workers to use for data loading, can change based on the number of cores")
    parser.add_argument("--warmup_steps", default=800, type=int, required=False, help="Number of warmup steps")
    parser.add_argument("--lr", default=1e-5, type=float, required=False, help="Learning rate for training")
    #----------------------------------------------------------
    args = parser.parse_args()

    result = SpeechLaughWhisper(args)
