import pandas as pd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from datasets import load_from_disk
import jiwer
import argparse
import torch
import os
import librosa
import re
import time

from preprocess import split_dataset, filter_and_match_datasets
from utils.metrics import track_laugh_word_alignments
from utils.transcript_process import alignment_transformation, transform_number_words


#--------------------------------------------------
# EVALUATE WHISPER
#--------------------------------------------------
def evaluate_whisper(
        source_dataset_path="./datasets/switchboard/token_speechlaugh/switchboard_dataset", 
        target_dataset_path="./datasets/switchboard/word_speechlaugh/word_speechlaugh/switchboard_dataset",
        model_name="openai/whisper-small",
        pretrained_model_dir="./ref_models/pre_trained",
        evaluate_dir="./evaluate",
        output_file="./alignment_transcripts/whisper_small_only_laughing_words.txt",
        dataset_type="word" # type of dataset to evaluate: "word" or "token"
        ):
    """
    Evaluates the Whisper model on a dataset provided in HuggingFace Dataset / CSV file 
    and writes the alignment transcripts to a file.
    Expected file output format:
    ```
    Evaluate model - openai/whisper-small --------------------------

    Audio Segment: sw2005_0000_0005.wav
     - From Audio: sw2005.wav
     - Start Timestamp: 0000 - End Timestamp: 0005

    REF: hello how are you doing today
    HYP: hello how are you doing today

    ...
    -------------------------------------------------------------
    ...

    Total Summary:

    Average WER: ...
    Average IOU: ...
    Average F1: ...
    ```
    Args:
        csv_file (str): Path to the CSV file containing audio paths and transcript (Evaluation Dataset)
        model_name (str): Name of the Whisper model to use (default: "openai/whisper-small").
    """
    print(f"Evaluate Whisper Model - {model_name} \n")

    # check GPU availability    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_dataset = None

    #---------------------------------------
    # LOAD PRETRAINED WHISPER MODEL + PROCESSOR
    #---------------------------------------
    processor = WhisperProcessor.from_pretrained(model_name, cache_dir=pretrained_model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, cache_dir=pretrained_model_dir)
    model.to(device)  # Move to GPUs

    #=========================================================================
    #                           LOAD DATASET
    #=========================================================================
    token_dataset = load_from_disk(source_dataset_path) # dataset with [SPEECH_LAUGH] and [LAUGHTER] tokens
    word_dataset = load_from_disk(target_dataset_path) # dataset with laughing words (in UPPERCASE)
    print(f"Loaded Token Dataset: \n{token_dataset}")
    print(f"Loaded Word Dataset: \n{word_dataset}")

    #=========================================================================
    #                           FILTER DATASET
    #=========================================================================
    laughter_dataset, laughing_words_dataset = filter_and_match_datasets(token_dataset, word_dataset)
    print(f"Filtered Laughter Dataset: \n{laughter_dataset}")
    print(f"Example of Laughter Dataset: \n{laughter_dataset[0]}")

    print(f"Filtered Laughing Words Dataset: \n{laughing_words_dataset}")
    print(f"Example of Laughing Words Dataset: \n{laughing_words_dataset[0]}")


    if dataset_type == "word":
        evaluate_dataset = laughing_words_dataset
    elif dataset_type == "token":
        evaluate_dataset = laughter_dataset
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}. Please specify 'word' or 'token'.")

    #==============================================================================================
    #                       MAIN EVALUATION PROCESS
    #==============================================================================================
    # wers = []

    summary_stats = {
        'wers': [], # word error rates

        # "speech_laugh_accuracy": [], #FIXME - Are there differences speech laugh accuracy and laugh word hit rate?
        # "laughter_accuracy": [],
        #------------------LAUGH WORDS------------------
        "whrs": [], # laugh word hit rates
        "wrs": [], # laugh word substitution rates
        "wdr": [], # laugh word deletion rates
        "wir": [], # laugh word insertion rates

        #------------------LAUGHTER TOKENS------------------
        "thr": [], # laughter token hit rates
        "trs": [], # laughter token substitution rates
        "tdr": [], # laughter token deletion rates
        "tir": [], # laughter token insertion rates

    }

    special_tokens = ["[SPEECH_LAUGH]", "[LAUGHTER]"]

    with open(output_file, "w") as f:
        f.write(f"Evaluate model - {model_name} --------------- \n\n")

        for row in evaluate_dataset:

            #----------------------------------------------------------------
            # NOW PROCESSING AUDIO
            #----------------------------------------------------------------

            audio_pathname = row['audio']['path'].split("/")[-1]
            f.write(f"Audio Segment: {audio_pathname} \n\n")
            audio_name = audio_pathname.split(".")[0] # Get the audio name excluding the .wav extension

            name, start_timestamp, end_timestamp = audio_name.split("_")
            f.write(f" - From Audio: {name} \n")
            f.write(f" - Start at: {start_timestamp} - End at: {end_timestamp} \n\n")

            # Load the audio
            audio = row['audio']['array']
            sr = row['audio']['sampling_rate']

            if sr != 16000:
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000) # Resample the audio to 16kHz

            #==================================================================================================
            #                                   REF Transcript                                                #   
            #==================================================================================================
            reference_transcript = row['transcript']
            if not isinstance(reference_transcript, str) or not reference_transcript.strip():
                print(f"Skipping empty or invalid reference transcript for {audio_pathname}")
                print(f"REF skipped: {reference_transcript}")
                continue

            reference_transcript = jiwer.ToLowerCase()(reference_transcript)
            # FIXME: Transform all existing numbers to words - NOT DO IN REF, USE IN HYP WITH REVERSE INSTEAD
            # reference_transcript = transform_number_words(reference_transcript) 
            f.write(f"REF: {reference_transcript} \n")

            #==================================================================================================
            #                                       HYP Transcript                                             #
            #==================================================================================================
            # Load and preprocess the audio
            audio = processor.feature_extractor(raw_speech=audio, sampling_rate=16000,return_tensors="pt").input_features
            audio = audio.to(device) # Move audio data to GPUs
            # Generate the predicted transcript
            predicted_ids = model.generate(audio)
            predicted_transcript = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Transform all existing numbers to words in the HYP transcript: 
            # one nine -> nineteen  
            predicted_transcript = transform_number_words(predicted_transcript, reverse=True) 

            if not isinstance(predicted_transcript, str) or not predicted_transcript.strip():
                print(f"Skipping empty or invalid predicted transcript for {audio_pathname}")
                print(f"HYP skipped: {predicted_transcript}")
                continue 
            
            predicted_transcript = alignment_transformation(predicted_transcript)
            
            f.write(f"HYP: {predicted_transcript} \n")
            f.write("-------------------------------------------------------\n")

            #--------------------------------
            #   VISUALIZE THE ALIGNMENT
            #--------------------------------
            alignment = jiwer.process_words(
                reference=reference_transcript, 
                hypothesis=predicted_transcript,
            )
            # Calculate the metrics
            wer = alignment.wer
            summary_stats['wers'].append(wer)

            #=====================================================================================================
            #                               TRACK LAUGH WORD ALIGNMENTS
            #=====================================================================================================
            laugh_stats = track_laugh_word_alignments(
                original_reference=row['transcript'], #ORIGINAL WITH UPPERCASE LAUGH WORDS instead of Lowercase ones
                hypothesis=predicted_transcript, 
                alignment=alignment
            )

            #=====================================================================================================
            #                               WRITE SUMMARY STATISTICS
            #=====================================================================================================
            summary_stats['whrs'].append(laugh_stats['laugh_word_hit_rate'])
            summary_stats['wrs'].append(laugh_stats['laugh_word_substitution_rate'])
            summary_stats['wdr'].append(laugh_stats['laugh_word_deletion_rate'])
            summary_stats['wir'].append(laugh_stats['laugh_word_insertion_rate'])

            summary_stats['thr'].append(laugh_stats['laugh_token_hit_rate'])
            summary_stats['trs'].append(laugh_stats['laugh_token_substitution_rate'])
            summary_stats['tdr'].append(laugh_stats['laugh_token_deletion_rate'])
            summary_stats['tir'].append(laugh_stats['laugh_token_insertion_rate'])


            #===============================================================================================================================
            #                       WRITE ALIGNMENT DETAILS (HITS, SUBSTITUTIONS, DELETIONS, INSERTIONS)
            #===============================================================================================================================
            f.write("\n========================================== Laugh Word Alignment Details ============================================= \n")

            f.write(f"Total Laugh Words: {laugh_stats['total_laugh_words']} \n")
            f.write(f"Total Laughter Tokens: {laugh_stats['total_laughter_tokens']} \n")
            f.write("---------------------------------\n")
            f.write(f"SPEECH LAUGH WORDS: [{', '.join(laugh_stats['laugh_words'])}] \n")
            f.write(f"LAUGHTER TOKENS: [{', '.join(laugh_stats['laughter_tokens'])}] \n")
            f.write("---------------------------------\n")

            if laugh_stats['hits'] and len(laugh_stats['hits']) > 0:
                f.write("\n ====== Hits: ====== \n")
                for hit in laugh_stats['hits']:
                    f.write(f"- REF: {hit['word']} → HYP: {hit['hyp_word']} "
                            f"(type: {hit['type']}, ref pos: {hit['ref_pos']}, hyp pos: {hit['hyp_pos']})\n")
            if laugh_stats['substitutions'] and len(laugh_stats['substitutions']) > 0:
                f.write("\n ====== Substitutions: ====== \n")
                for sub in laugh_stats['substitutions']:
                    f.write(f"- REF: {sub['ref_word']} → HYP: {sub['hyp_word']} "
                            f"(type: {sub['type']}, ref pos: {sub['ref_pos']}, "
                            f"hyp pos: {sub['hyp_pos'] if sub['hyp_pos'] is not None else 'N/A'})\n")
            if laugh_stats['deletions'] and len(laugh_stats['deletions']) > 0:
                f.write("\n ====== Deletions: ====== \n")
                for deletion in laugh_stats['deletions']:
                    f.write(f"- REF: {deletion['word']} "
                            f"(type: {deletion['type']}, ref pos: {deletion['ref_pos']})\n")
            if laugh_stats['insertions'] and len(laugh_stats['insertions']) > 0:
                f.write("\n ====== Insertions: ====== \n")
                for insertion in laugh_stats['insertions']:
                    f.write(f"- HYP: {insertion['word']} "
                            f"(type: {insertion['type']}, hyp pos: {insertion['hyp_pos']})\n")

            f.write("\n ====== Detailed Alignment: ====== \n")
            f.write(jiwer.visualize_alignment(alignment, show_measures=False, skip_correct=False) + "\n")
            f.write("______________________________________________________________________________________________________________________\n\n")
        

        f.write("=========================== OVERALL METRICS SUMMARY =======================================\n")
        
        f.write(f"Average WER: {np.mean(summary_stats['wers']):.2f} \n\n")

        f.write("___________________________________________________________________________________________\n")
        f.write(f"Avg Laugh Word Detected Rate: {np.mean(summary_stats['whrs']):.2f} \n")
        f.write(f"Avg Laugh Word Substitution Rate: {np.mean(summary_stats['wrs']):.2f} \n")
        f.write(f"Avg Laugh Word Deletion Rate: {np.mean(summary_stats['wdr']):.2f} \n")
        f.write(f"Avg Laugh Word Insertion Rate: {np.mean(summary_stats['wir']):.2f} \n")
        f.write("___________________________________________________________________________________________\n\n")
        f.write(f"Avg Laughter Token Hit Rate: {np.mean(summary_stats['thr']):.2f} \n")
        f.write(f"Avg Laughter Token Substitution Rate: {np.mean(summary_stats['trs']):.2f} \n")
        f.write(f"Avg Laughter Token Deletion Rate: {np.mean(summary_stats['tdr']):.2f} \n")
        f.write(f"Avg Laughter Token Insertion Rate: {np.mean(summary_stats['tir']):.2f} \n")
        f.write("____________________________________________________________________________________________\n")

if __name__ == "__main__":
    # csv_file = "train_switchboard.csv"  # Replace with your actual CSV file path
    parser = argparse.ArgumentParser(description="Evaluate the Whisper model on Switchboard dataset.")
    parser.add_argument("--source_dataset_path", type=str, required=True, default="./datasets/switchboard/token_speechlaugh/switchboard_dataset", help="Path to the source dataset.")
    parser.add_argument("--target_dataset_path", type=str, required=True, default="./datasets/switchboard/word_speechlaugh/word_speechlaugh/switchboard_dataset", help="Path to the target dataset.")
    parser.add_argument("--model_name", type=str, required=True, default="openai/whisper-small", help="Name of the Whisper model to use.")
    parser.add_argument("--output_file", type=str, default="./alignment_transcripts/alignment_whisper_small_laughing_words.txt", help="File to write the alignment transcripts.")

    args = parser.parse_args()

    start_time = time.time()
    evaluate_whisper(
        source_dataset_path=args.source_dataset_path,
        target_dataset_path=args.target_dataset_path,
        model_name=args.model_name,
        output_file=args.output_file
    )
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")