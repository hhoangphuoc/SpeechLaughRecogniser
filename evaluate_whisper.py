import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
import datasets
import jiwer
import argparse
import torch
import os
import librosa

# Processing output for evaluation
output_transform = jiwer.Compose([
    jiwer.ExpandCommonEnglishContractions(), # can't -> cannot, it's -> it is
    jiwer.RemoveEmptyStrings(), # remove empty strings
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(), 

])

def evaluate_whisper(
        csv_file, 
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

    # Load the Whisper processor and model
    processor = WhisperProcessor.from_pretrained(model_name, cache_dir=pretrained_model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, cache_dir=pretrained_model_dir)

    df = pd.read_csv(csv_file) # Load the dataset

    # wer_metric = evaluate.load("wer", cache_dir=evaluate_dir)# Load the WER metric

    # check GPU availability    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move to GPUs


    # Iterate over the dataset and calculate WER for each example
    # if not os.path.exists(output_file):
    #     os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(f"Evaluate model - {model_name} \n\n")
        for index, row in df.iterrows():
            audio_path = row['audio'] # Audio path

            audio, sr = librosa.load(audio_path) # Load the audio

            if sr != 16000:
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000) # Resample the audio to 16kHz

            reference_transcript = row['transcript'] # Reference transcript

            # Load and preprocess the audio
            audio = processor.feature_extractor(raw_speech=audio, sampling_rate=16000,return_tensors="pt").input_features
            audio = audio.to(device) # Move audio data to GPUs
            attention_mask = torch.ones(audio.shape, device=audio.device) # Attention mask

            # Generate the predicted transcript
            predicted_ids = model.generate(inputs=audio, attention_mask=attention_mask)
            predicted_transcript = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0] # Hypothesis Transcript

            # Visualise alignment with Jiwer
            alignment = jiwer.process_words(reference_transcript, predicted_transcript)
            f.write(jiwer.visualize_alignment(alignment, show_measures=True, skip_correct=False) + "\n")


if __name__ == "__main__":
    # csv_file = "train_switchboard.csv"  # Replace with your actual CSV file path
    parser = argparse.ArgumentParser(description="Evaluate the Whisper model on Switchboard dataset.")
    parser.add_argument("--csv_file", type=str, required=True, default="./datasets/combined/val_switchboard.csv", help="Path to the CSV file containing audio paths and transcript.")
    parser.add_argument("--model_name", type=str, default="openai/whisper-small", help="Name of the Whisper model to use.")
    parser.add_argument("--output_file", type=str, default="./alignment_transcripts/alignment_whisper_small.txt", help="File to write the alignment transcripts.")

    args = parser.parse_args()

    evaluate_whisper(
        csv_file=args.csv_file,
        model_name=args.model_name,
        output_file=args.output_file
    )