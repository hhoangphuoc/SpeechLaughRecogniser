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

#---------------------------------------
# TRANSCRIPT TRANFORMATION
#---------------------------------------
alignment_transformation = jiwer.Compose([
    # jiwer.RemovePunctuation(), # FIXME: NOT USING THIS TO REMOVE PUNCTUATION BECAUSE IT ALSO REMOVE - AND ' WHICH ARE IMPORTANT FOR RETOKENIZATION
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemoveEmptyStrings(),
    jiwer.ToLowerCase(),
    jiwer.ReduceToSingleSentence() # incase the HYP transcript is not a single sentence
])

# TRANSFORMATION FOR NUMBERS
def transform_number_to_word(sentence):
    number_words = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine"
    }

    # Use re.sub with a lambda function to replace digits
    pattern = r"\d"  # Matches any single digit
    return re.sub(pattern, lambda match: number_words[match.group(0)] + " ", sentence).strip()


#----------------------
# METRICS FUNCTIONS
#----------------------
# wer_metric = evaluate.load("wer") #WER

def calculate_iou(ref_sentence, hyp_sentence):
    """
    Calculate the Intersection Over Union (IOU) between reference and hypothesis sentence
    By this we can evaluate the similarity between the two sentences, and how accurate
    the model is in predicting the laughter words
    """
    # ref_segments = [segment.split() for segment in ref_segments]
    # hyp_segments = [segment.split() for segment in hyp_segments]

    ref_words = set(ref_sentence.split())
    hyp_words = set(hyp_sentence.split())

    intersection = ref_words.intersection(hyp_words)
    union = ref_words.union(hyp_words)

    if len(union) == 0:
        return 0
    
    similarity = len(intersection) / len(union)

    return similarity

def calculate_f1(ref_sentence, hyp_sentence):
    """
    Calculate the F1 score between reference and hypothesis segments.
    F1 score is the harmonic mean of precision and recall.
    This is used to evaluate the accuracy of the model in recognising the laughing words
    In this case, we only use it in the dataset that having [SPEECH_LAUGH] or [LAUGHTER] token
    
    Args:
        ref_segments: List of reference segments (ground truth)
        hyp_segments: List of hypothesis segments (predictions)
        
    Returns:
        float: F1 score between 0 and 1
    """
    # Convert segments to sets of words
    ref_words = set(ref_sentence.split())
    hyp_words = set(hyp_sentence.split())

    # Calculate true positives (intersection)
    true_positives = len(ref_words.intersection(hyp_words))
    
    # Handle edge case of empty sets
    if len(ref_words) == 0 and len(hyp_words) == 0:
        return 1.0
    if len(ref_words) == 0 or len(hyp_words) == 0:
        return 0.0
    
    # Calculate precision and recall
    precision = true_positives / len(hyp_words)
    recall = true_positives / len(ref_words)
    
    # Handle edge case where both precision and recall are 0
    if precision + recall == 0:
        return 0.0
        
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


#-----------------------------------------------------------------------------------
# TRACK LAUGH WORD ALIGNMENTS
# FIXME - Is that a better way to track?
#-----------------------------------------------------------------------------------
def track_laugh_word_alignments(reference, hypothesis, alignment):
    """
    Track the alignment status of laughing words (in uppercase) between reference and hypothesis.
    
    Args:
        reference (str): Original reference transcript with uppercase laugh words
        hypothesis (str): Predicted transcript
        alignment: JiWER alignment object
        
    Returns:
        dict: Statistics about laugh word alignments including hits, substitutions, deletions
    """
    # Split reference into words while preserving case
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Find indices of laugh words (uppercase words)
    laugh_word_indices = {
        i: word.lower() 
        for i, word in enumerate(ref_words) 
        if word.isupper()
    }

    # Initialize tracking dictionaries
    laugh_word_stats = {
        'hits': [],        # Correct alignments
        'substitutions': [],  # Wrong word at aligned position
        'deletions': [],   # Laugh word was deleted
        'total': len(laugh_word_indices)
    }

        # Helper function to extract word from alignment output
    def get_word(aligned_word):
        if isinstance(aligned_word, str):
            return aligned_word
        elif isinstance(aligned_word, list) and len(aligned_word) > 0:
            return aligned_word[0]
        else:
            return '*'
    
    # Get alignment information from JiWER output
    # ref_aligned = alignment.references
    # hyp_aligned = alignment.hypotheses
    ref_aligned = [get_word(word) for word in alignment.references]
    hyp_aligned = [get_word(word) for word in alignment.hypotheses]
    
    # Track current position in reference and hypothesis
    ref_pos = 0
    aligned_pos = 0
    ref_map = {}
    
    for word in ref_aligned:
        if word != '*':  # Skip insertions
            ref_map[ref_pos] = aligned_pos
            ref_pos += 1
        aligned_pos += 1
    for ref_pos, laugh_word in laugh_word_indices.items():
        if ref_pos not in ref_map:
            # Words was deleted
            laugh_word_stats['deletions'].append({
                'word': laugh_word,
                'ref_pos': ref_pos
            })
            continue
        
        aligned_pos = ref_map[ref_pos]

        if aligned_pos < len(hyp_aligned):
            hyp_word = hyp_aligned[aligned_pos]

            if hyp_word == "*":
                # Word was deleted
                laugh_word_stats['deletions'].append({
                    'word': laugh_word,
                    'ref_pos': ref_pos
                })

            elif hyp_word.lower() == laugh_word:
                # Mark a HITs match in laughter words
                laugh_word_stats['hits'].append({
                    'word': laugh_word,
                    'ref_pos': ref_pos,
                    'hyp_pos': aligned_pos
                })
            else:
                # Mark a SUBSTITUTION
                laugh_word_stats['substitutions'].append({
                    'ref_word': laugh_word,
                    'hyp_word': hyp_word,
                    'ref_pos': ref_pos,
                    'hyp_pos': aligned_pos
                })
        else:
            # Word was deleted
            laugh_word_stats['deletions'].append({
                'word': laugh_word,
                'ref_pos': ref_pos
            })
    
    # Calculate statistics
    laugh_word_stats['hit_rate'] = len(laugh_word_stats['hits']) / laugh_word_stats['total'] if laugh_word_stats['total'] > 0 else 0
    laugh_word_stats['substitution_rate'] = len(laugh_word_stats['substitutions']) / laugh_word_stats['total'] if laugh_word_stats['total'] > 0 else 0
    laugh_word_stats['deletion_rate'] = len(laugh_word_stats['deletions']) / laugh_word_stats['total'] if laugh_word_stats['total'] > 0 else 0
    
    return laugh_word_stats

#-----------------------------------------------------------------------------------
def evaluate_laughter_words(dataset):
    pass
def evaluate_laughter_tokens(dataset):
    pass



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

    #--------------------------------------
    #       LOAD DATASET
    #--------------------------------------
    token_dataset = load_from_disk(source_dataset_path) # dataset with [SPEECH_LAUGH] and [LAUGHTER] tokens
    word_dataset = load_from_disk(target_dataset_path) # dataset with laughing words (in UPPERCASE)
    print(f"Loaded Token Dataset: \n{token_dataset}")
    print(f"Loaded Word Dataset: \n{word_dataset}")

    #--------------------------------------
    #       FILTER DATASET
    #--------------------------------------
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

    # ----------------------------------- MAIN EVALUATION PROCESS -----------------------------------
    wers = []
    # ious = []
    # f1s = []

    total_laugh_words = 0
    total_matched_laugh_words = 0

    special_tokens = ["[SPEECH_LAUGH]", "[LAUGHTER]"]

    with open(output_file, "w") as f:
        f.write(f"Evaluate model - {model_name} --------------- \n\n")

        # for row in laughing_words_dataset:
        for row in evaluate_dataset:
        #----------------------------------------------------------------
        # NOW PROCESSING THE DATASET THAT CONTAINS [SPEECH_LAUGH] AND [LAUGHTER] TOKENS
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

            #--------------------------------------------------------------------------------------------------
            #                                   REF Transcript                                                #   
            #--------------------------------------------------------------------------------------------------
            reference_transcript = row['transcript']
            if not isinstance(reference_transcript, str) or not reference_transcript.strip():
                print(f"Skipping empty or invalid reference transcript for {audio_pathname}")
                print(f"REF skipped: {reference_transcript}")
                continue
            
            # laugh_words = set(word.lower() for word in reference_transcript.split() if word.isupper() and word not in special_tokens)
            if dataset_type == "word":
                laugh_words = set(word.lower() for word in reference_transcript.split() if word.isupper() and word not in special_tokens) #any UPPERCASE words that not [SPEECH_LAUGH] or [LAUGHTER]
            elif dataset_type == "token":
                laugh_words = set(word.lower() for word in reference_transcript.split() if word in special_tokens)
            
            current_laugh_words = len(laugh_words)
            total_laugh_words += current_laugh_words

            reference_transcript = jiwer.ToLowerCase()(reference_transcript)
            f.write(f"REF: {reference_transcript} \n")

            #---------------------------------------------------------------------------------------------------
            #                                       HYP Transcript                                             #
            #---------------------------------------------------------------------------------------------------
            # Load and preprocess the audio
            audio = processor.feature_extractor(raw_speech=audio, sampling_rate=16000,return_tensors="pt").input_features
            audio = audio.to(device) # Move audio data to GPUs
            # Generate the predicted transcript
            predicted_ids = model.generate(audio)
            predicted_transcript = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            predicted_transcript = transform_number_to_word(predicted_transcript) # First: Transform all existing numbers to words
            
            if not isinstance(predicted_transcript, str) or not predicted_transcript.strip():
                print(f"Skipping empty or invalid predicted transcript for {audio_pathname}")
                print(f"HYP skipped: {predicted_transcript}")
                continue 
            
            predicted_transcript = alignment_transformation(predicted_transcript)
            
            # Count the number of matched laugh words ------------------------------------------------------------
            predicted_words = predicted_transcript.split()
            matched_laugh_words = sum(
                sum(1 for match_word in predicted_words if match_word == laugh_word)
                for laugh_word in laugh_words
            )
            total_matched_laugh_words += matched_laugh_words
            #--------------------------------------------------------------------------------------------

            f.write(f"HYP: {predicted_transcript} \n")
            f.write("-----------------------------------\n")

            f.write(f"SPEECH LAUGH WORDS: [{', '.join(laugh_words)} ] \n")
            f.write(f"Matched Laugh Words: {matched_laugh_words}/{current_laugh_words} \n\n")


            #--------------------------------
            #   VISUALIZE THE ALIGNMENT
            #--------------------------------
            alignment = jiwer.process_words(
                reference=reference_transcript, 
                hypothesis=predicted_transcript,
            )
            # Calculate the metrics
            wer = alignment.wer
            wers.append(wer)

            # -----------------------------------------------------------------------------------
            # TRACK LAUGH WORD ALIGNMENTS
            # -----------------------------------------------------------------------------------
            laugh_word_stats = track_laugh_word_alignments(
                reference=row['transcript'], #ORIGINAL WITH UPPERCASE LAUGH WORDS instead of Lowercase ones
                hypothesis=predicted_transcript, 
                alignment=alignment
            )
            f.write("\n --------- Laugh Word Alignment Details -------- \n")
            f.write(f"Total Laugh Words: {laugh_word_stats['total']} \n")

            # -----------------------------------------------------------------------------------
            # WRITE ALIGNMENT OPERATIONS IF ANY (HITS, SUBSTITUTIONS, DELETIONS)
            # -----------------------------------------------------------------------------------
            if laugh_word_stats['hits']:
                f.write("\nCorrect alignments:\n")
                for hit in laugh_word_stats['hits']:
                    f.write(f"- {hit['word']} (ref pos: {hit['ref_pos']}, hyp pos: {hit['hyp_pos']})\n")
            
            if laugh_word_stats['substitutions']:
                f.write("\nSubstitutions:\n")
                for sub in laugh_word_stats['substitutions']:
                    f.write(f"- {sub['ref_word']} by {sub['hyp_word']} (ref pos: {sub['ref_pos']}, hyp pos: {sub['hyp_pos']})\n")
            
            if laugh_word_stats['deletions']:
                f.write("\nDeletions:\n")
                for deletion in laugh_word_stats['deletions']:
                    f.write(f"- {deletion['word']} (ref pos: {deletion['ref_pos']})\n")
            
            f.write("Rate Summary: ------------- \n")
            f.write(f"\nHit rate: {laugh_word_stats['hit_rate']:.2%}\n")
            f.write(f"Substitution rate: {laugh_word_stats['substitution_rate']:.2%}\n")
            f.write(f"Deletion rate: {laugh_word_stats['deletion_rate']:.2%} \n")
            f.write("----------------------------\n\n")


            f.write(f"------- Detailed Alignment ---------- \n")
            f.write(jiwer.visualize_alignment(alignment, show_measures=True, skip_correct=False) + "\n")
            f.write("______________________________________________________________________________________________________________________\n\n")
        

        #----------------------------------------------
        #   METRICS SUMMARY
        #---------------------------------------------
        f.write("--------------------- Laughing Words Summary ---------------------\n")
        f.write(f"Total Laughing Words in REF: {total_laugh_words} \n")
        f.write(f"Total Matched Laugh Words in HYP: {total_matched_laugh_words} \n")

        speech_laugh_accuracy = total_matched_laugh_words / total_laugh_words if total_laugh_words > 0 else 0
        f.write(f"Speech Laugh Accuracy: {speech_laugh_accuracy:.4f} \n")
        f.write("-------------------------------------------------- \n\n")

        f.write("--------------------- Metrics Summary ---------------------\n")
        f.write(f"Average WER: {np.mean(wers)} \n")
        # f.write(f"Average IOU: {np.mean(ious)} \n")
        # f.write(f"Average F1: {np.mean(f1s)} \n")
        f.write("-------------------------------------------------- \n")


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