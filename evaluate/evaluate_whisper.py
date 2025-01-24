import numpy as np
from transformers import (
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    )
from datasets import load_from_disk
import jiwer
import argparse
import torch
import librosa
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import transform_number_words, transform_alignment_sentence
from utils import evaluate_token_alignments


#--------------------------------------------------
# EVALUATE WHISPER
#--------------------------------------------------
def evaluate_whisper(
        # source_dataset_path="./datasets/switchboard/token_speechlaugh/switchboard_dataset", 
        # target_dataset_path="./datasets/switchboard/word_speechlaugh/word_speechlaugh/switchboard_dataset",
        dataset_dir="../datasets/switchboard",
        dataset_type="speechlaugh", # type of dataset to evaluate: "laugh" or "speechlaugh" or "speech"
        model_name="openai/whisper-small",
        pretrained_model_dir="../ref_models/pre_trained",
        output_dir="../alignment_transcripts"
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

    #---------------------------------------
    # LOAD PRETRAINED WHISPER MODEL + PROCESSOR
    #---------------------------------------
    processor = WhisperProcessor.from_pretrained(model_name, cache_dir=pretrained_model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, cache_dir=pretrained_model_dir)
    model.to(device)  # Move to GPUs


    #=========================================================================
    #                           LOAD DATASET
    #=========================================================================
    try:
        dataset_name = f"swb_{dataset_type}"
        dataset_path = os.path.join(dataset_dir, dataset_name, "switchboard_dataset") #../datasets/switchboard/swb_speechlaugh/switchboard_dataset  
        evaluate_dataset = load_from_disk(dataset_path)
        print(f"Loaded Dataset with type: {dataset_type}: \n{evaluate_dataset}")

        # FIXME:Get 6900 samples from the dataset
        evaluate_dataset = evaluate_dataset.select(range(6900))
    
    except FileNotFoundError:
        raise ValueError(f"Dataset not found: {dataset_path}. Please choose the dataset types of 'speechlaugh', 'laugh' or 'speech'.")

    #==============================================================================================
    #                       MAIN EVALUATION PROCESS
    #==============================================================================================
    # wers = []

    summary_stats = {
        "total_TH": 0,
        "total_TS": 0,
        "total_TD": 0,
        "total_TI": 0,
        "total_token_operations": 0,

    }

    # special_tokens = ["[LAUGH]"]
    output_file = os.path.join(output_dir, f"alignment_swb_{dataset_type}_large-v2.txt")

    with open(output_file, "w") as f:
        f.write(f"Evaluate model - {model_name} --------------- \n")
        f.write(f"Dataset: {dataset_name} \n\n")

        ref_transcripts = []
        hyp_transcripts = []
        
        for row in evaluate_dataset:
            #----------------------------------------------------------------
            #                   NOW PROCESSING AUDIO
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
            original_transcript = row['transcript'] #TODO: Original transcript with uppercase [LAUGH] for laughter or WORD for speech-laugh
            if not isinstance(original_transcript, str) or not original_transcript.strip():
                print(f"Skipping empty or invalid reference transcript for {audio_pathname}")
                print(f"ORIGINAL REF skipped: {original_transcript}")
                continue

            reference_transcript = jiwer.ToLowerCase()(original_transcript)
            f.write(f"REF: {reference_transcript} \n")
            ref_transcripts.append(reference_transcript)

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
            
            predicted_transcript = transform_alignment_sentence(predicted_transcript)
            
            f.write(f"HYP: {predicted_transcript} \n")
            hyp_transcripts.append(predicted_transcript)

            f.write("-------------------------------------------------------\n")

            

            # #=====================================================================================================  
            # #                           VISUALIZE THE PAIR ALIGNMENT
            # #=====================================================================================================
            pair_alignment = jiwer.process_words(
                reference=reference_transcript, 
                hypothesis=predicted_transcript,
            )

            #FIXME: CALCULATE THE ALIGNMENT OF TOKEN IN PAIR OF SENTENCES, 
            # TO GET THE NUMBER OF TOKENS HITS, SUBSTITUTIONS, INSERTIONS, DELETIONS
            # AND CALCULATE THE RATES. THESE ARE THE PERFORMANCE OF THE MODEL ON EACH DATASET   

            f.write(jiwer.visualize_alignment(pair_alignment, show_measures=False, skip_correct=False) + "\n")

            # #=====================================================================================================
            # #                               TRACK LAUGH WORD ALIGNMENTS
            # #=====================================================================================================
            
            if dataset_type == "speechlaugh" or dataset_type == "laugh" or dataset_type == "laugh_intext":
                laugh_stats = evaluate_token_alignments(
                    original_reference=original_transcript, #ORIGINAL WITH UPPERCASE LAUGH WORDS instead of Lowercase ones
                    hypothesis=predicted_transcript, 
                    alignment=pair_alignment,
                    dataset_type=dataset_type
                ) # return the number of hits, substitutions, deletions, insertions for each pair

                #===============================================================================================================================
                #                       VISUALIZE ALIGNMENT DETAILS (HITS, SUBSTITUTIONS, DELETIONS, INSERTIONS)
                #===============================================================================================================================
                f.write("\n========================================== Laugh Word Alignment Details ============================================= \n")
                f.write(f"{dataset_type.upper()} WORDS: [{', '.join(laugh_stats['laugh_words'])}] \n")
                f.write("-------------------------------------------------------------------------\n")

                # TH: NUMBER OF TOKENS HITS 
                th_count = len(laugh_stats['TH']) if laugh_stats['TH'] else 0
                summary_stats['total_TH'] += th_count #FIXME: ADD TO TOTAL HITS

                #Visualise Hits (TH)
                if th_count > 0:
                    f.write("\n ====== Hits: ====== \n")
                    for hit in laugh_stats['TH']:
                        f.write(f"- REF: {hit['word']} → HYP: {hit['hyp_word']} "
                                f"(type: {hit['type']}, ref pos: {hit['ref_pos']}, hyp pos: {hit['hyp_pos']})\n")

                #------------------------------------------------------------------------------------------------   
                
                # TS: NUMBER OF TOKENS SUBSTITUTIONS
                ts_count = len(laugh_stats['TS']) if laugh_stats['TS'] else 0
                summary_stats['total_TS'] += ts_count #FIXME: ADD TO TOTAL SUBSTITUTIONS

                #Visualise Substitutions (TS)
                if ts_count > 0:
                    f.write("\n ====== Substitutions: ====== \n")
                    for sub in laugh_stats['TS']:
                        f.write(f"- REF: {sub['ref_word']} → HYP: {sub['hyp_word']} "
                                f"(type: {sub['type']}, ref pos: {sub['ref_pos']}, "
                                f"hyp pos: {sub['hyp_pos'] if sub['hyp_pos'] is not None else 'N/A'})\n")

                #------------------------------------------------------------------------------------------------
    
                # TD: NUMBER OF TOKENS DELETIONS
                td_count = len(laugh_stats['TD']) if laugh_stats['TD'] else 0
                summary_stats['total_TD'] += td_count #FIXME: ADD TO TOTAL DELETIONS

                #Visualise Deletions (TD)
                if td_count > 0:
                    f.write("\n ====== Deletions: ====== \n")
                    for deletion in laugh_stats['TD']:
                        f.write(f"- REF: {deletion['word']} "
                                f"(type: {deletion['type']}, ref pos: {deletion['ref_pos']})\n")

                #------------------------------------------------------------------------------------------------   

                # TI: NUMBER OF TOKENS INSERTIONS
                ti_count = len(laugh_stats['TI']) if laugh_stats['TI'] else 0
                summary_stats['total_TI'] += ti_count #FIXME: ADD TO TOTAL INSERTIONS

                #Visualise Insertions (TI)
                if ti_count > 0:
                    f.write("\n ====== Insertions: ====== \n")
                    for insertion in laugh_stats['TI']:
                        f.write(f"- HYP: {insertion['word']} "
                                f"(type: {insertion['type']}, hyp pos: {insertion['hyp_pos']})\n")

                #------------------------------------------------------------------------------------------------ 
                # TOTAL TOKEN OPERATIONS
                total_token_operations = th_count + ts_count + td_count + ti_count
                summary_stats['total_token_operations'] += total_token_operations
            else:
                continue   
        f.write("______________________________________________________________________________________________________________________\n\n")

        alignment = jiwer.process_words(
            reference=ref_transcripts, 
            hypothesis=hyp_transcripts,
        )


        f.write("=========================== OVERALL METRICS SUMMARY =======================================\n")
        
        f.write(f"Percentage WER: {alignment.wer * 100:.2f} \n\n")
        f.write(f"Percentage MER: {alignment.mer * 100:.2f} \n\n") # Match Error Rate
        f.write("___________________________________________________________________________________________\n")
        # TOTAL OPERATIONS OF ALL DATASET
        hits, substitutions, deletions, insertions = alignment.hits, alignment.substitutions, alignment.deletions, alignment.insertions
        total_operations = hits + substitutions + deletions + insertions
        f.write(f"Total Operations: {total_operations} \n")

        f.write(f"Percentage Hits: {hits / total_operations * 100:.2f} \n")
        f.write(f"Percentage Substitutions: {substitutions / total_operations * 100:.2f} \n")
        f.write(f"Percentage Deletions: {deletions / total_operations * 100:.2f} \n")
        f.write(f"Percentage Insertions: {insertions / total_operations * 100:.2f} \n")

        f.write("___________________________________________________________________________________________\n")

        if dataset_type == "speechlaugh" or dataset_type == "laugh" or dataset_type == "laugh_intext":
            # TOTAL TOKEN OPERATIONS OF ALL DATASET
            f.write(f"Total Token Operations: {summary_stats['total_token_operations']} \n")
            f.write("___________________________________________________________________________________________\n")
            f.write(f"{dataset_type.upper()} Token Hit Rate: {summary_stats['total_TH'] / summary_stats['total_token_operations'] * 100:.2f} \n")
            f.write(f"{dataset_type.upper()} Token Substitution Rate: {summary_stats['total_TS'] / summary_stats['total_token_operations'] * 100:.2f} \n")
            f.write(f"{dataset_type.upper()} Token Deletion Rate: {summary_stats['total_TD'] / summary_stats['total_token_operations'] * 100:.2f} \n")
            f.write(f"{dataset_type.upper()} Token Insertion Rate: {summary_stats['total_TI'] / summary_stats['total_token_operations'] * 100:.2f} \n")
            f.write("___________________________________________________________________________________________\n\n")

if __name__ == "__main__":
    # # csv_file = "train_switchboard.csv"  # Replace with your actual CSV file path
    # parser = argparse.ArgumentParser(description="Evaluate the Whisper model on Switchboard dataset.")
    # parser.add_argument("--dataset_dir", type=str, required=True, default="./datasets/switchboard", help="Path to the dataset directory.")
    # parser.add_argument("--dataset_type", type=str, required=True, default="speechlaugh", help="Type of dataset to evaluate: 'speechlaugh' or 'laugh' or 'speech'.")
    # parser.add_argument("--model_name", type=str, required=True, default="openai/whisper-small", help="Name of the Whisper model to use.")
    # parser.add_argument("--pretrained_model_dir", type=str, required=True, default="../ref_models/pre_trained", help="Path to the pretrained model directory.")
    # parser.add_argument("--output_dir", type=str, default="./alignment_transcripts", help="Directory to write the alignment transcripts.")

    # args = parser.parse_args()

    # start_time = time.time()
    # evaluate_whisper(
    #     dataset_dir=args.dataset_dir,
    #     dataset_type=args.dataset_type,
    #     model_name=args.model_name,
    #     pretrained_model_dir=args.pretrained_model_dir,
    #     output_dir=args.output_dir
    # )
    # end_time = time.time()
    # print(f"Total runtime: {end_time - start_time} seconds")
    #===============================================================================================================

    #===================================================
    #       TEST EVALUATE WHISPER WITH 1 SAMPLE
    #===================================================
    pre_trained_path = "../fine-tuned/whisper/finetuned-whisper-10epochs"
    model = WhisperForConditionalGeneration.from_pretrained(pre_trained_path)
    # tokenizer = WhisperTokenizer.from_pretrained(pre_trained_path)
    
    processor = WhisperProcessor.from_pretrained(pre_trained_path)
    example_audio = librosa.load("../evaluate/example/sw02089A_2230385_226499875.wav", sr=16000)

    audio = processor.feature_extractor(raw_speech=example_audio[0], sampling_rate=16000, return_tensors="pt").input_features

    with torch.no_grad(): #FIXME: added `with torch.no_grad()` to avoid gradient computation
        predicted_ids = model.generate(audio)

    predicted_transcript = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    reference_transcript = "but he sold his boat too so <laugh> so we just kind of"

    pair_alignment = jiwer.process_words(
        reference=reference_transcript,
        hypothesis=predicted_transcript
    )
    print(jiwer.visualize_alignment(pair_alignment, show_measures=False, skip_correct=False))
    print("---------------------------end---------------------------------")