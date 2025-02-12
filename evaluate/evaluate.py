import numpy as np
from transformers import (
        WhisperProcessor, 
        WhisperTokenizer,
        WhisperFeatureExtractor,
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
def seperate_file_transcripts(
        all_transcripts_file,
        # all_transcripts_json_file,
        ref_file,
        hyp_file,
        normed_ref_file,
        normed_ref_upper_file,
        normed_hyp_file,
        alignment_file,
    ):
    """
    Write the transcripts to the corresponding text files, namely:

    - original_ref_[model_type].txt
    - original_hyp_[model_type].txt
    - normalised_ref_[model_type].txt
    - normalised_hyp_[model_type].txt

    THIS FUNCTION ONLY USED IN CASE ALL TRANSCRIPTS ARE WRITTEN IN THE SAME FILE.

    """
    print("Writing the transcript files to separate file ...")
    ref, hyp, normed_ref, normed_ref_upper, normed_hyp = [], [], [], [], []
    try:
        if all_transcripts_file is None:
            raise ValueError("Transcript file is None. Unable to write the transcript files.")

        with open(all_transcripts_file, "r") as f:
            all_transcripts = f.readlines()

            if all_transcripts is None:
                raise ValueError("All transcripts is None. Unable to write the transcript files.")
            
            for sentence in tqdm(all_transcripts, desc="Seperating transcripts to files"):
                ref.append(sentence.removeprefix("REF: ")) if sentence.startswith("REF:") else None
                hyp.append(sentence.removeprefix("HYP: "))  if sentence.startswith("HYP:") else None
                normed_ref.append(sentence.removeprefix("NORMED REF: ")) if sentence.startswith("NORMED REF:") else None
                normed_ref_upper.append(sentence.removeprefix("NORMED REF UPPER: ")) if sentence.startswith("NORMED REF UPPER:") else None
                normed_hyp.append(sentence.removeprefix("NORMED HYP: ")) if sentence.startswith("NORMED HYP:") else None
            
            # Write the transcripts to the corresponding text files
            write_transcript(ref_file, ref, transcript_type="ref")
            write_transcript(hyp_file, hyp, transcript_type="hyp")
            write_transcript(normed_ref_upper_file, normed_ref_upper, transcript_type="normalised ref upper")
            write_transcript(normed_ref_file, normed_ref, transcript_type="normalised ref")
            write_transcript(normed_hyp_file, normed_hyp, transcript_type="normalised hyp")

            write_alignment_transcript(
                alignment_file=alignment_file,
                model_type="finetuned-whisper-nolaugh",
                alignment_ref=normed_ref,
                alignment_hyp=normed_hyp
            )
        f.close()

    except Exception as e:
        print(f"Error: {e}. Please make sure the transcripts are not empty.")
        return
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
    
    # SPECFIC GLOBAL PATH FOR BUCKEYE AUDIO
    audio_global_path = "/deepstore/datasets/hmi/speechlaugh-corpus/buckeye_data/buckeye_refs_wavs2/audio_wav/"

    #---------------------------------------
    # LOAD PRETRAINED MODEL + PROCESSOR
    #---------------------------------------
    processor = None
    model = None
    # transcriber = None

    if model_name.startswith("openai/whisper"):
        model = WhisperForConditionalGeneration.from_pretrained(model_name, cache_dir=pretrained_model_dir, local_files_only=True)
        processor = WhisperProcessor.from_pretrained(model_name, cache_dir=pretrained_model_dir, local_files_only=True)

        # using pipeline for ASR with long-form audio
        # transcriber = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, device=device)
    
    
    elif model_name.startswith("facebook/wav2vec2"):

        #FIXME: If using Wav2Vec2, using pipeline instead
        #FIXME: However, pipeline might not work fully-functioned. CONSIDER TRY WITH: `Wav2Vev2ProcessorWithLM` 

        processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=pretrained_model_dir, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=pretrained_model_dir, local_files_only=True)


    elif model_name.startswith("finetuned-wav2vec2"):
        print("Loading the finetuned Wav2Vec2 model ...")
        model_path = os.path.join(pretrained_model_dir, model_name)
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
    
    elif model_name.startswith("finetuned-whisper"):
        print("Loading the finetuned Whisper model ...")
        model_path = os.path.join(pretrained_model_dir, model_name)
        print(f"Model Path: {model_path}")
        
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        
        #-------------------------------------------------------------------
        # TODO: REMEMBER TO ADD TO `preprocessor_config.json` in `processor` folder to the checkpoint folder
        # Otherwise it will not work
        #-------------------------------------------------------------------
        processor = WhisperProcessor.from_pretrained(model_path)

        # ----------------------------- USING HUGGINGFACE PIPELINE ------------------------------
        # tokenizer = WhisperTokenizer.from_pretrained(model_path)
        # feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
        # # using pipeline for ASR with long-form audio
        # transcriber = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor, chunk_length_s=30, device=device)

        # ----------------------------- TOKEN DECODER + CHUNK -----------------------------------

    if model is not None:
        print(f"Model `{model_name}` loaded successfully!")
        model.to(device) # use `.half()` to use mixed precision for faster inference (fp16)
    else:
        raise ValueError(f"Model not found: {model_name}. Please choose the model types of 'openai/whisper-*' or 'facebook/wav2vec2-*'.")


    #================================================================================================
    #                                           LOAD DATASET
    #================================================================================================
    try:
        #../datasets/switchboard/whisper/swb_test  
        test_dataset = load_from_disk(dataset_dir)

        print("Loaded Test Dataset:", test_dataset)
    
    except FileNotFoundError:
        raise ValueError(f"Dataset not found: {dataset_dir}. Please choose the correct path for the test dataset.")

    #==============================================================================================
    #                                           MAIN PROCESS
    #==============================================================================================

    ref, hyp = [], []
    normalised_ref_upper = []
    normalised_ref, normalised_hyp = [], []
    i = 0
    
    for recording in test_dataset:
        #-------------------------------------------------------------------------------------------------
        #                                   ORIGINAL REF TRANSCRIPTS
        #-------------------------------------------------------------------------------------------------
        original_ref = recording['transcript']

        # Load the audio
        audio_name = recording['audio']['path']
        audio_path = os.path.join(audio_global_path, audio_name)
        audio = recording['audio']['array']
        sr = recording['audio']['sampling_rate']
        
        if sr != 16000:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
        
        # skip the recording if the transcript is empty
        if not original_ref.strip():
            print(f"Recording {audio_path} empty transcript. Skipped.")
            continue
        elif audio is None:
            print(f"Recording {audio_path} has empty audio. Skipped.")
            continue
        else:
            i += 1
            print(f"Recording: {i} / {len(test_dataset)}")
            print(f"Audio Path: {audio_path}")

            print(f"REF: {original_ref}")
            ref.append(original_ref)

            try:
                #-------------------------------------------------------------------------------------------------
                #                                   ORIGINAL HYP TRANSCRIPTS
                #-------------------------------------------------------------------------------------------------
        
                # Extracting the audio to input_features and predicted_ids
                input_features = None
                predicted_ids = None
                hyp_transcript = None

                if model_name.startswith("openai/whisper") or model_name.startswith("finetuned-whisper"):
                    
                    #------------------------ PROCESSING USING TOKENIZER DECODER --------------------------
                    # # Load and preprocess the audio
                    # input_features = processor.feature_extractor(
                    #     audio, 
                    #     sampling_rate=16000,
                    #     return_tensors="pt"
                    # ).input_features

                    # input_features = input_features.to(device) # Move input feature to GPUs

                    # with torch.no_grad(): #FIXME: added `with torch.no_grad()` to avoid gradient computation
                    #     # Generate the predicted transcript
                    #     predicted_ids = model.generate(input_features)
                    # hyp_transcript = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    #--------------------------------------------------------------------------------------


                    #------------------------ PROCESSING USING PIPELINE -----------------------------------
                    # using pipeline for ASR with long-form audio
                    # hyp_transcript = transcriber(recording["audio"], chunk_length_s=10)['text']
                    #--------------------------------------------------------------------------------------


                    #------------------------ PROCESSING USING TOKENIZER DECODE + CHUNK -------------------#
                    chunk_length_s = 10.0 #chunk length in seconds (10s - 15s for accurate transcribing)
                    chunk_length_samples = int(chunk_length_s * sr) #chunk length in samples (rate)

                    hyp_transcript_chunks = []

                    # Split the audio into chunks
                    for start_idx in range(0, len(audio), chunk_length_samples):
                        end_idx = min(start_idx + chunk_length_samples, len(audio))
                        chunk = audio[start_idx:end_idx]

                        # Process audio chunk
                        input_features = processor(
                            chunk, 
                            sampling_rate=16000, 
                            return_tensors="pt"
                        ).input_features

                        input_features = input_features.to(device)

                        with torch.no_grad():
                            predicted_ids = model.generate(input_features) #TODO:use num_beams=5 for more audio content?
                        
                        # hyp_transcript = processor.batch_decode(predicted_ids)[0]
                        transcript_chunk = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                        hyp_transcript_chunks.append(transcript_chunk)
                    
                    hyp_transcript = " ".join(hyp_transcript_chunks)
                    #--------------------------------------------------------------------------------------#
                

                elif model_name.startswith("facebook/wav2vec2") or model_name.startswith("finetuned-wav2vec2"):
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

                else:
                    raise ValueError(f"Model not found: {model_name}. Please choose the model relatives to 'openai/whisper-*', 'facebook/wav2vec2-*', or a fine-tuned version of these model such as `finetuned-whisper-*`, or `finetuned-wav2vec2-*`.")
                
                if hyp_transcript is None:
                    raise ValueError("Unable to generate the transcript. This could be due to `pipeline` not used properly.")
                
                print(f"HYP: {hyp_transcript}")
                hyp.append(hyp_transcript)


                #-------------------------------------------------------------------------------------------------
                #                                   NORMALISED REF UPPER TRANSCRIPTS
                # Normalise the reference transcript without lowercasing
                #-------------------------------------------------------------------------------------------------
                normalised_ref_upper_transcript = transform_alignment_sentence(original_ref)

                print(f"NORMED REF UPPER: {normalised_ref_upper_transcript}")
                normalised_ref_upper.append(normalised_ref_upper_transcript)


                #-------------------------------------------------------------------------------------------------
                #                                   NORMALISED REF TRANSCRIPTS
                #-------------------------------------------------------------------------------------------------
                normalised_ref_transcript = jiwer.ToLowerCase()(normalised_ref_upper_transcript)

                print(f"NORMED REF: {normalised_ref_transcript}")
                normalised_ref.append(normalised_ref_transcript)

                #-------------------------------------------------------------------------------------------------
                #                                   NORMALISED HYP TRANSCRIPTS
                #-------------------------------------------------------------------------------------------------
                # hyp_transcript = transform_number_words(hyp_transcript, reverse=True)
                
                if model_type == "wav2vec2":
                    # replace the token "<" to "<laugh>" to match the reference transcript in CTC model
                    hyp_transcript = hyp_transcript.replace("<", " <laugh> ")
                
                normalised_hyp_transcript = transform_alignment_sentence(hyp_transcript)
                normalised_hyp_transcript = jiwer.ToLowerCase()(normalised_hyp_transcript)

                print(f"NORMED HYP: {normalised_hyp_transcript}")
                normalised_hyp.append(normalised_hyp_transcript)

            except Exception as e:
                print(f"Error: {e}. Unable to generate the transcript for the recording {i}. Please check the audio file.")
                print(f"Recording {audio_path} skipped.")
                continue
    
    print("Finished processing all the recordings. Outputing to different transcript lists.")
    return ref, hyp, normalised_ref, normalised_ref_upper, normalised_hyp #TODO: return 5 types of transcripts
#============================================================================================================


if __name__ == "__main__":

    # ============================================================  REMOVE BELOW IF UNUSED =======================================

    # start_time = time.time()
    # transcripts_dir = os.path.join("../alignment_transcripts", "buckeye2", "finetuned_whisper_nolaugh")
    # seperate_file_transcripts(
    #     all_transcripts_file=os.path.join(transcripts_dir, "all_transcripts.txt"),
    #     # all_transcripts_json_file=os.path.join(transcripts_dir, f"all_transcripts_wav2vec2.json"),
    #     ref_file=os.path.join(transcripts_dir, "original_ref_whisper.txt"),
    #     hyp_file=os.path.join(transcripts_dir, "original_hyp_whisper.txt"),
    #     normed_ref_file=os.path.join(transcripts_dir, "normalised_ref_whisper.txt"),
    #     normed_ref_upper_file=os.path.join(transcripts_dir, "normalised_ref_upper_whisper.txt"),
    #     normed_hyp_file=os.path.join(transcripts_dir, "normalised_hyp_whisper.txt"),
    #     alignment_file=os.path.join(transcripts_dir, "alignment_whisper.txt")
    # )
    # end_time = time.time()
    # print(f"Finished! Total runtime: {end_time - start_time} seconds")
    
    

    #----------------------------------------------------------------------------------------------------------------------#
    # THE CODE ABOVE USED TO SEPARATE THE TRANSCRIPTS AND WRITE TO THE CORRESPONDING TEXT FILES SEPARATELY AND LOCALLY
    #           BUT IT CAN ONLY BE USED IN THE CASE ALL TRANSCRIPTS HAS BEEN PROCESSED AND WRITTEN TO THE SAME FILE.
    #----------------------------------------------------------------------------------------------------------------------#


    # # ====================================================== REMOVE ABOVE IF UNUSED ==================================================
    

    # csv_file = "train_switchboard.csv"  # Replace with your actual CSV file path
    parser = argparse.ArgumentParser(description="Evaluate Model on Switchboard test dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True, default="./datasets/switchboard", help="Path to the dataset directory.")
    parser.add_argument("--model_name", type=str, required=True, default="openai/whisper-large-v2", help="Name of the Whisper model to use.")
    parser.add_argument("--model_type", type=str, required=True, default="whisper", help="Type of model to use: 'whisper' or 'wav2vec2'.")
    parser.add_argument("--pretrained_model_dir", type=str, required=True, default="../ref_models/pre_trained", help="Path to the pretrained model directory.")
    parser.add_argument("--output_dir", type=str, default="./alignment_transcripts", help="Directory to write the alignment transcripts.")

    args = parser.parse_args()

    start_time = time.time()


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # ============================== GET TRANSCRIPTS ==================================================
    
    ref, hyp, normed_ref, normed_ref_upper, normed_hyp = get_transcripts(
        dataset_dir=args.dataset_dir,
        model_name=args.model_name,
        model_type=args.model_type,
        pretrained_model_dir=args.pretrained_model_dir,
        output_dir=args.output_dir
    )

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

    normed_ref_upper_file = os.path.join(args.output_dir, f"normalised_ref_upper_{args.model_type}.txt") #e.g. normalised_ref_whisper.txt
    write_transcript(
        normed_ref_upper_file, 
        normed_ref_upper, 
        transcript_type="normalised ref upper")
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


    # ==========================================================================================