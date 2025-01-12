import numpy as np
from transformers import (
        WhisperProcessor, 
        WhisperForConditionalGeneration, 
        Wav2Vec2Processor, 
        Wav2Vec2ForCTC,
        pipeline, 
        logging
    )
from datasets import load_from_disk
import jiwer
import argparse
import torch
import librosa
import time
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm

from preprocess import transform_number_words, transform_alignment_sentence
from utils import evaluate_token_alignments


# ============================================= WRITE TRANSCRIPTS FUNCTIONS =============================================

def write_alignment_transcript(
        alignment_file,
        model_type="whisper",
        alignment_ref=None,
        alignment_hyp=None,
        ):
    """
    Get the alignment transcripts for the Switchboard test set for specific model and write to the corresponding text file
    named `alignment_[model_type].txt`.
    """

    # ==========================================================================================
    #                           OUTPUT ALIGNMENT TRANSCRIPTS
    # ==========================================================================================

    with open(alignment_file, "w") as f2:
        f2.write(f"Alignment Model - {model_type} \n")

        if alignment_ref is None or alignment_hyp is None:
            raise ValueError("Normalised reference and hypothesis transcripts is None.")
        
        alignment = jiwer.process_words(
            reference=alignment_ref, #list of references
            hypothesis=alignment_hyp, #list of hypothesis
        )

        f2.write("========== OVERALL METRICS SUMMARY =================\n")
        
        f2.write(f"Percentage WER: {alignment.wer * 100:.2f} \n")
        f2.write(f"Percentage MER: {alignment.mer * 100:.2f} \n") # Match Error Rate
        f2.write("-----------------------------------------------\n\n")


        f2.write("========== ALIGNMENT TRANSCRIPTS =================\n")
        f2.write(jiwer.visualize_alignment(alignment, show_measures=False, skip_correct=False))

        f2.write("__________________________________________end of transcripts______________________________________________\n")
    print(f"Finished! Alignment transcript for {model_type} processed to: {alignment_file}")
#============================================================================================================


def write_transcript(
    transcript_file,
    transcripts,
    transcript_type="ref" #TODO: or "hyp" or "normalised ref" or "normalised hyp"
):
    """
    Write the transcripts with specified transcript_type to the corresponding text file, namely:

    - original_ref_[model_type].txt
    - original_hyp_[model_type].txt
    - normalised_ref_[model_type].txt
    - normalised_hyp_[model_type].txt

    """
    print(f"Writing the transcript file for {transcript_type.upper()} to {transcript_file} ...")
    try:
        if transcripts is None:
            raise ValueError(f"Transcripts is None. Unable to write the transcript file for {transcript_type}.")
        
        with open(transcript_file, "w") as f:
            f.write(f"Transcript of: {transcript_type.upper()} ------------------ \n\n")
            f.write("========== TRANSCRIPTS =================\n\n")
            num_recordings = len(transcripts)
            for i, transcript in enumerate(transcripts):
                f.write(f"Recording: ({i+1}/ {num_recordings}) \n")
                f.write(f"{transcript} \n\n")
            
            f.write("__________________________________________end of transcripts______________________________________________\n")
    except Exception as e:
        print(f"Error: {e}. Please make sure the transcripts are not empty.")
        return
    finally:
        print(f"Transcript file processed to output directory: {transcript_file}")
#============================================================================================================




# ============================================= EVALUATIONS FUNCTIONS =============================================
#--------------------------------------------------
# EVALUATE WHISPER
#--------------------------------------------------
def get_transcripts(
        dataset_dir="../datasets/switchboard/whisper/",
        model_name="openai/whisper-large-v2", #TODO: choosing between `whisper-large-v2` and `wav2vec2-large-960h-lv60`
        model_type="whisper",
        pretrained_model_dir="../ref_models/pre_trained",
        output_dir="../alignment_transcripts/whisper" #TODO: Change to `whisper` or `wav2vec2` based on the model used
        ):
    """
    Get all the transcripts for the Switchboard test set, for specific model, including:
    - Original Reference Transcripts (REF)
    - Original Hypothesis Transcripts (HYP)
    - Normalised Reference Transcripts (NORMED_REF)
    - Normalised Hypothesis Transcripts (NORMED_HYP)

    Each transcript store separately in different text file.
    """
    print(f"Evaluate Model - {model_name} \n")
    print(f"Dataset Directory: {dataset_dir} \n")

    # check GPU availability    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #---------------------------------------
    # LOAD PRETRAINED MODEL + PROCESSOR
    #---------------------------------------
    processor = None
    model = None

    if model_name.startswith("openai/whisper"):
        processor = WhisperProcessor.from_pretrained(model_name, cache_dir=pretrained_model_dir, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(model_name, cache_dir=pretrained_model_dir, local_files_only=True)
    
    
    elif model_name.startswith("facebook/wav2vec2"):

        #FIXME: If using Wav2Vec2, using pipeline instead
        #FIXME: However, pipeline might not work fully-functioned. CONSIDER TRY WITH: `Wav2Vev2ProcessorWithLM` 

        processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=pretrained_model_dir, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=pretrained_model_dir, local_files_only=True)

        # FIXME: Using pipeline doesnt work
        # pipe = pipeline(
        #     "automatic-speech-recognition", 
        #     model=model,
        #     tokenizer=tokenizer,
        #     feature_extractor=feature_extractor,
        #     device=device
        # )
    
    if model is not None:
        print(f"Model {model_name} loaded successfully!")
        model.to(device)
    else:
        raise ValueError(f"Model not found: {model_name}. Please choose the model types of 'openai/whisper-*' or 'facebook/wav2vec2-*'.")


    #================================================================================================
    #                                           LOAD DATASET
    #================================================================================================
    try:
        #../datasets/switchboard/whisper/swb_test  
        swb_test = load_from_disk(dataset_dir)

        print("Loaded Test Dataset:", swb_test)
    
    except FileNotFoundError:
        raise ValueError(f"Dataset not found: {dataset_dir}. Please choose the correct path for the test dataset.")

    #==============================================================================================
    #                                           MAIN PROCESS
    #==============================================================================================

    ref, hyp = [], []
    normalised_ref, normalised_hyp = [], []
    i = 0
    
    for recording in swb_test:
        i += 1
        print(f"Recording: {i} / {len(swb_test)}")

        try:
            # print(recording)

            #-------------------------------------------------------------------------------------------------
            #                                   ORIGINAL REF TRANSCRIPTS
            #-------------------------------------------------------------------------------------------------
            original_ref = recording['transcript']
            print(f"REF: {original_ref}")
            ref.append(original_ref)


            #-------------------------------------------------------------------------------------------------
            #                                   ORIGINAL HYP TRANSCRIPTS
            #-------------------------------------------------------------------------------------------------
            # Load the audio
            audio = recording['audio']['array']
            sr = recording['audio']['sampling_rate']

            # if sr != 16000:
            #     audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000) # Resample the audio to 16kHz
            
    
            # Extracting the audio to input_features and predicted_ids
            input_features = None
            predicted_ids = None
            hyp_transcript = None

            if model_name.startswith("openai/whisper"):
                # Load and preprocess the audio
                input_features = processor.feature_extractor(
                    audio, 
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features

                input_features = input_features.to(device) # Move input feature to GPUs

                # Generate the predicted transcript
                predicted_ids = model.generate(input_features)

                hyp_transcript = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            elif model_name.startswith("facebook/wav2vec2"):
                # FOR WA2VEC2, using `processor` instead of `processor.feature_extractor`
                # as it need to map both audio to specified tokens in the vocabulary when producing input_values
                input_features = processor(
                        audio,
                        sampling_rate=16000,
                        return_tensors="pt"
                    ).input_values
                    
                input_features = input_features.to(device)

                with torch.no_grad():
                    logits = model(input_features).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                
                hyp_transcript = processor.batch_decode(predicted_ids)[0]
                
                # IF WE USED `pipeline`, use this instead
                # hyp_transcript = pipe(audio, batch_size=1)["text"]

            else:
                raise ValueError(f"Model not found: {model_name}. Please choose the model types of 'openai/whisper-large-v2' or 'facebook/wav2vec2-large-lv60'.")
            
            if  hyp_transcript is None:
                raise ValueError("Unable to generate the transcript. This could be due to `pipeline` not used properly.")
            
            print(f"HYP: {hyp_transcript}")
            hyp.append(hyp_transcript)


            #-------------------------------------------------------------------------------------------------
            #                                   NORMALISED REF TRANSCRIPTS
            #-------------------------------------------------------------------------------------------------
            normalised_ref_transcript = transform_alignment_sentence(original_ref)
            print(f"NORMED REF: {normalised_ref_transcript}")

            normalised_ref.append(normalised_ref_transcript)


            #-------------------------------------------------------------------------------------------------
            #                                   NORMALISED HYP TRANSCRIPTS
            #-------------------------------------------------------------------------------------------------
            hyp_transcript = transform_number_words(hyp_transcript, reverse=True) 
            normalised_hyp_transcript = transform_alignment_sentence(hyp_transcript)
            print(f"NORMED HYP: {normalised_hyp_transcript}")

            normalised_hyp.append(normalised_hyp_transcript)

        except Exception as e:
            print(f"Error: {e}. Unable to generate the transcript for the recording {i}. Please check the audio file.")
            continue
    
    print("Finished processing all the recordings. Outputing to different transcript lists.")
    return ref, hyp, normalised_ref, normalised_hyp #TODO: return 4 types of transcripts
#============================================================================================================


#--------------------------------------------------
# EVALUATE WAVE2VEC2 
# with `Wave2Vec2ProcessorWithLM`
#--------------------------------------------------
#TODO


#-----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # csv_file = "train_switchboard.csv"  # Replace with your actual CSV file path
    parser = argparse.ArgumentParser(description="Evaluate Model on Switchboard test dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True, default="./datasets/switchboard", help="Path to the dataset directory.")
    parser.add_argument("--model_name", type=str, required=True, default="openai/whisper-large-v2", help="Name of the Whisper model to use.")
    parser.add_argument("--model_type", type=str, required=True, default="whisper", help="Type of model to use: 'whisper' or 'wav2vec2'.")
    parser.add_argument("--pretrained_model_dir", type=str, required=True, default="../ref_models/pre_trained", help="Path to the pretrained model directory.")
    parser.add_argument("--output_dir", type=str, default="./alignment_transcripts", help="Directory to write the alignment transcripts.")

    args = parser.parse_args()

    start_time = time.time()

    # ============================== GET TRANSCRIPTS =======================================
    
    ref, hyp, normed_ref, normed_hyp = get_transcripts(
        dataset_dir=args.dataset_dir,
        model_name=args.model_name,
        model_type=args.model_type,
        pretrained_model_dir=args.pretrained_model_dir,
        output_dir=args.output_dir
    )
    # save the transcripts to json file
    with open(os.path.join(args.output_dir, f"all_transcripts_{args.model_type}.json"), "w") as f:
        json.dump({
            "references": ref,
            "hypotheses": hyp,
            "normalised_references": normed_ref,
            "normalised_hypotheses": normed_hyp
        }, f, indent=4)

    # ============================== WRITE TRANSCRIPTS TO TEXT FILES  =======================================

    # Original Reference Transcripts
    ref_file = os.path.join(args.output_dir, f"original_ref_{args.model_type}.txt") #e.g. original_ref_whisper.txt
    write_transcript(
        ref_file, 
        ref, 
        transcript_type="ref")
    #---------------------------------------------------------------------------------------------------------

    # Original Hypothesis Transcripts
    hyp_file = os.path.join(args.output_dir, f"original_hyp_{args.model_type}.txt") #e.g. original_hyp_whisper
    write_transcript(
        hyp_file, 
        hyp, 
        transcript_type="hyp")
    #---------------------------------------------------------------------------------------------------------
    
    # Normalised Reference Transcripts
    normed_ref_file = os.path.join(args.output_dir, f"normalised_ref_{args.model_type}.txt") #e.g. normalised_ref_whisper.txt
    write_transcript(
        normed_ref_file, 
        normed_ref, 
        transcript_type="normalised ref")
    #---------------------------------------------------------------------------------------------------------
    
    # Normalised Hypothesis Transcripts
    normed_hyp_file = os.path.join(args.output_dir, f"normalised_hyp_{args.model_type}.txt") #e.g. normalised_hyp_whisper.txt
    write_transcript(
        normed_hyp_file, 
        normed_hyp, 
        transcript_type="normalised hyp")
    #---------------------------------------------------------------------------------------------------------
    
    # Alignment Transcripts
    alignment_file = os.path.join(args.output_dir, f"alignment_{args.model_type}.txt") #e.g. alignment_whisper.txt
    write_alignment_transcript(
        alignment_file,
        model_type=args.model_type,
        alignment_ref=normed_ref,
        alignment_hyp=normed_hyp
    )
    
    end_time = time.time()
    print(f"Finished! Total runtime: {end_time - start_time} seconds")