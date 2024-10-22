import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from datasets import load_from_disk
import jiwer
import argparse
import torch
import os
import librosa

from preprocess import split_dataset

# TRANFORMATION FOR HYP TRANSCRIPT BEFORE ALIGNMENT
alignment_transformation = jiwer.Compose([
    # jiwer.RemovePunctuation(), # FIXME: NOT USING THIS TO REMOVE PUNCTUATION BECAUSE IT ALSO REMOVE - AND ' WHICH ARE IMPORTANT FOR RETOKENIZATION
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemoveEmptyStrings(),
    jiwer.ToLowerCase(),
    jiwer.ReduceToSingleSentence() # incase the HYP transcript is not a single sentence
])

def evaluate_whisper(
        dataset_path="./datasets/processed/switchboard_datasets", 
        model_name="openai/whisper-small",
        pretrained_model_dir="./ref_models/pre_trained",
        evaluate_dir="./evaluate",
        output_file="./alignment_transcripts/alignment_whisper_small.txt"
        ):
    """
    Evaluates the Whisper model on a dataset provided in a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing audio paths and transcript (Evaluation Dataset)
        model_name (str): Name of the Whisper model to use (default: "openai/whisper-small").
    """
    print(f"Evaluate Whisper Model - {model_name} \n")

    # check GPU availability    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #---------------------------------------
    # LOAD PRETRAINED WHISPER MODEL + PROCESSOR
    #---------------------------------------
    processor = WhisperProcessor.from_pretrained(model_name, cache_dir=pretrained_model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, cache_dir=pretrained_model_dir)
    model.to(device)  # Move to GPUs

    #--------------------------------------
    #       LOAD DATASET
    #--------------------------------------
    dataset = load_from_disk(dataset_path)
    print(f"Loaded Switchboard Dataset: \n{dataset}")

    test_dataset = split_dataset(dataset, split_ratio=0.9, split="test")
    print(f"Test dataset: \n{test_dataset}")

    with open(output_file, "w") as f:
        f.write(f"Evaluate model - {model_name} \n\n")
        for row in test_dataset:
            # audio_path = row['path']
            audio = row['audio']['array']
            sr = row['audio']['sampling_rate']

            if sr != 16000:
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000) # Resample the audio to 16kHz

            #-------------------------
            #       REF Transcript
            #-------------------------
            reference_transcript = row['transcript']
            # reference_transcript = alignment_transformation(reference_transcript) #FIXME: DO WE NEED TO DO THIS FOR REF TRANSCRIPT?

            # Load and preprocess the audio
            audio = processor.feature_extractor(raw_speech=audio, sampling_rate=16000,return_tensors="pt").input_features
            audio = audio.to(device) # Move audio data to GPUs

            # attention_mask = torch.ones(audio.shape, device=audio.device) # Attention mask

            #--------------------------------
            #       HYP Transcript 
            #--------------------------------
            # Generate the predicted transcript
            predicted_ids = model.generate(audio)
            predicted_transcript = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0] 
            predicted_transcript = alignment_transformation(predicted_transcript)

            #--------------------------------
            #   VISUALIZE THE ALIGNMENT
            #--------------------------------
            alignment = jiwer.process_words(reference_transcript, predicted_transcript)
            f.write(jiwer.visualize_alignment(alignment, show_measures=True, skip_correct=False) + "\n")


if __name__ == "__main__":
    # csv_file = "train_switchboard.csv"  # Replace with your actual CSV file path
    parser = argparse.ArgumentParser(description="Evaluate the Whisper model on Switchboard dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, default="./datasets/processed/switchboard_dataset", help="Path to the CSV file containing audio paths and transcript.")
    parser.add_argument("--model_name", type=str, required=True, default="openai/whisper-small", help="Name of the Whisper model to use.")
    parser.add_argument("--output_file", type=str, default="./alignment_transcripts/alignment_whisper_small.txt", help="File to write the alignment transcripts.")

    args = parser.parse_args()

    evaluate_whisper(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_file=args.output_file
    )