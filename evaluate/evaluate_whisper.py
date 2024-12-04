import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
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
from utils import track_laugh_word_alignments


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

        # Get 6900 samples from the dataset
        # evaluate_dataset = evaluate_dataset.select(range(6900))
    except FileNotFoundError:
        raise ValueError(f"Dataset not found: {dataset_path}. Please choose the dataset types of 'speechlaugh', 'laugh' or 'speech'.")

    #==============================================================================================
    #                       MAIN EVALUATION PROCESS
    #==============================================================================================
    # wers = []

    summary_stats = {
        'wers': [], # word error rates
        'mers': [], # match error rates
        'hit_rate': [], # sentence hit rate
        'substitution_rate': [], # sentence substitution rate
        'deletion_rate': [], # sentence deletion rate
        'insertion_rate': [], # sentence insertion rate
        'thr': [], # token hit rate
        'tsr': [], # token substitution rate
        'tdr': [], # token deletion rate
        'tir': [], # token insertion rate
    }

    # special_tokens = ["[LAUGH]"]
    output_file = os.path.join(output_dir, f"alignment_swb_{dataset_type}_large-v2.txt")

    with open(output_file, "w") as f:
        f.write(f"Evaluate model - {model_name} --------------- \n")
        f.write(f"Dataset: {dataset_name} \n\n")
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
            
            predicted_transcript = transform_alignment_sentence(predicted_transcript)
            
            f.write(f"HYP: {predicted_transcript} \n")
            f.write("-------------------------------------------------------\n")

            #=====================================================================================================  
            #                           VISUALIZE THE ALIGNMENT
            #=====================================================================================================
            alignment = jiwer.process_words(
                reference=reference_transcript, 
                hypothesis=predicted_transcript,
            )
            # Calculate the metrics
            wer = alignment.wer
            mer = alignment.mer
            summary_stats['wers'].append(wer)
            summary_stats['mers'].append(mer)

            #=====================================================================================================  
            #           GET THE NUMBER OF HITS, SUBSTITUTIONS, INSERTIONS, DELETIONS for each sentences
            #           AND CALCULATE THE RATES. THESE ARE THE PERFORMANCE OF THE MODEL ON EACH DATASET
            #=====================================================================================================
            hits = alignment.hits
            substitutions = alignment.substitutions
            insertions = alignment.insertions
            deletions = alignment.deletions

            total_operations = hits + substitutions + insertions + deletions

            summary_stats['hit_rate'].append(hits / total_operations)
            summary_stats['substitution_rate'].append(substitutions / total_operations)
            summary_stats['insertion_rate'].append(insertions / total_operations)
            summary_stats['deletion_rate'].append(deletions / total_operations)

            #=====================================================================================================
            #                               TRACK LAUGH WORD ALIGNMENTS
            #=====================================================================================================
            
            if dataset_type == "speechlaugh" or dataset_type == "laugh" or dataset_type == "laugh_intext":
                laugh_stats = track_laugh_word_alignments(
                    original_reference=original_transcript, #ORIGINAL WITH UPPERCASE LAUGH WORDS instead of Lowercase ones
                    hypothesis=predicted_transcript, 
                    alignment=alignment,
                    dataset_type=dataset_type
                )

                #=====================================================================================================
                #                               WRITE SUMMARY TOKENS STATISTICS 
                #                   HIT RATE, SUBSTITUTION RATE, DELETION RATE, INSERTION RATE    
                #=====================================================================================================
                summary_stats['thr'].append(laugh_stats['thr']) # token hit rate
                summary_stats['tsr'].append(laugh_stats['tsr']) # token substitution rate
                summary_stats['tdr'].append(laugh_stats['tdr']) # token deletion rate
                summary_stats['tir'].append(laugh_stats['tir']) # token insertion rate

                #===============================================================================================================================
                #                       WRITE ALIGNMENT DETAILS (HITS, SUBSTITUTIONS, DELETIONS, INSERTIONS)
                #===============================================================================================================================
                f.write("\n========================================== Laugh Word Alignment Details ============================================= \n")
                f.write(f"{dataset_type.upper()} WORDS: [{', '.join(laugh_stats['laugh_words'])}] \n")
                f.write("-------------------------------------------------------------------------\n")

                if laugh_stats['laugh_hits'] and len(laugh_stats['laugh_hits']) > 0:
                    f.write("\n ====== Hits: ====== \n")
                    for hit in laugh_stats['laugh_hits']:
                        f.write(f"- REF: {hit['word']} → HYP: {hit['hyp_word']} "
                                f"(type: {hit['type']}, ref pos: {hit['ref_pos']}, hyp pos: {hit['hyp_pos']})\n")
                if laugh_stats['laugh_substitutions'] and len(laugh_stats['laugh_substitutions']) > 0:
                    f.write("\n ====== Substitutions: ====== \n")
                    for sub in laugh_stats['laugh_substitutions']:
                        f.write(f"- REF: {sub['ref_word']} → HYP: {sub['hyp_word']} "
                                f"(type: {sub['type']}, ref pos: {sub['ref_pos']}, "
                                f"hyp pos: {sub['hyp_pos'] if sub['hyp_pos'] is not None else 'N/A'})\n")
                if laugh_stats['laugh_deletions'] and len(laugh_stats['laugh_deletions']) > 0:
                    f.write("\n ====== Deletions: ====== \n")
                    for deletion in laugh_stats['laugh_deletions']:
                        f.write(f"- REF: {deletion['word']} "
                                f"(type: {deletion['type']}, ref pos: {deletion['ref_pos']})\n")
                if laugh_stats['laugh_insertions'] and len(laugh_stats['laugh_insertions']) > 0:
                    f.write("\n ====== Insertions: ====== \n")
                    for insertion in laugh_stats['laugh_insertions']:
                        f.write(f"- HYP: {insertion['word']} "
                                f"(type: {insertion['type']}, hyp pos: {insertion['hyp_pos']})\n")

            f.write("\n ====== Detailed Alignment: ====== \n")
            f.write(jiwer.visualize_alignment(alignment, show_measures=True, skip_correct=False) + "\n")
            f.write("______________________________________________________________________________________________________________________\n\n")
    

        f.write("=========================== OVERALL METRICS SUMMARY =======================================\n")
        
        f.write(f"Average WER: {np.mean(summary_stats['wers']) * 100:.2f} \n\n")
        f.write(f"Average MER: {np.mean(summary_stats['mers']) * 100:.2f} \n\n") # Match Error Rate
        
        f.write("___________________________________________________________________________________________\n")

        f.write(f"Avg Sentence Hit Rate: {np.mean(summary_stats['hit_rate']) * 100:.2f} \n")
        f.write(f"Avg Sentence Substitution Rate: {np.mean(summary_stats['substitution_rate']) * 100:.2f} \n")
        f.write(f"Avg Sentence Deletion Rate: {np.mean(summary_stats['deletion_rate']) * 100:.2f} \n")
        f.write(f"Avg Sentence Insertion Rate: {np.mean(summary_stats['insertion_rate']) * 100:.2f} \n")
        f.write("___________________________________________________________________________________________\n\n")

        if dataset_type == "speechlaugh" or dataset_type == "laugh" or dataset_type == "laugh_intext":
            f.write("___________________________________________________________________________________________\n")
            f.write(f"{dataset_type.upper()} Token Hit Rate: {np.mean(summary_stats['thr']) * 100:.2f} \n")
            f.write(f"{dataset_type.upper()} Token Substitution Rate: {np.mean(summary_stats['tsr']) * 100:.2f} \n")
            f.write(f"{dataset_type.upper()} Token Deletion Rate: {np.mean(summary_stats['tdr']) * 100:.2f} \n")
            f.write(f"{dataset_type.upper()} Token Insertion Rate: {np.mean(summary_stats['tir']) * 100:.2f} \n")
            f.write("___________________________________________________________________________________________\n\n")

if __name__ == "__main__":
    # csv_file = "train_switchboard.csv"  # Replace with your actual CSV file path
    parser = argparse.ArgumentParser(description="Evaluate the Whisper model on Switchboard dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True, default="./datasets/switchboard", help="Path to the dataset directory.")
    parser.add_argument("--dataset_type", type=str, required=True, default="speechlaugh", help="Type of dataset to evaluate: 'speechlaugh' or 'laugh' or 'speech'.")
    parser.add_argument("--model_name", type=str, required=True, default="openai/whisper-small", help="Name of the Whisper model to use.")
    parser.add_argument("--pretrained_model_dir", type=str, required=True, default="../ref_models/pre_trained", help="Path to the pretrained model directory.")
    parser.add_argument("--output_dir", type=str, default="./alignment_transcripts", help="Directory to write the alignment transcripts.")

    args = parser.parse_args()

    start_time = time.time()
    evaluate_whisper(
        dataset_dir=args.dataset_dir,
        dataset_type=args.dataset_type,
        model_name=args.model_name,
        pretrained_model_dir=args.pretrained_model_dir,
        output_dir=args.output_dir
    )
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")