import numpy as np
from transformers import (
        WhisperProcessor, 
        WhisperForConditionalGeneration, 
        Wav2Vec2Processor, 
        Wav2Vec2ForCTC,
        Wav2Vec2ProcessorWithLM,
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
def get_transcripts_whisper(
        dataset_dir="../datasets/switchboard/whisper/",
        model_name="openai/whisper-large-v2", #TODO: choosing between `whisper-large-v2` and `wav2vec2-large-lv60`
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
        processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=pretrained_model_dir, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=pretrained_model_dir, local_files_only=True)
    
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
            if model_name.startswith("openai/whisper"):
                # Load and preprocess the audio
                input_features = processor.feature_extractor(audio, sampling_rate=16000,return_tensors="pt").input_features

                input_features = input_features.to(device) # Move input feature to GPUs

                # Generate the predicted transcript
                predicted_ids = model.generate(input_features)
            
            elif model_name.startswith("facebook/wav2vec2"):
                # with torch.no_grad():

                #     input_features = processor(
                #         audio,
                #         sampling_rate=16000,
                #         return_tensors="pt"
                #     ).input_values
                    
                #     input_features = input_features.to(device)

                #     # Generate the predicted transcript
                #     logits = model(input_features).logits

                # predicted_ids = torch.argmax(logits, dim=-1)
                
                # FOR WA2VEC2, using `processor` instead of `processor.feature_extractor`
                # as it need to map both audio to specified tokens in the vocabulary when producing input_values
                input_features = processor(
                    audio, sampling_rate=16000, return_tensors="pt", 
                    # padding=True
                ).input_values

                input_features = input_features.to(device)

                # Get the logits
                with torch.no_grad():
                    logits = model(input_features).logits

                # Decode the predicted ids
                predicted_ids = torch.argmax(logits, dim=-1)
            else:
                raise ValueError(f"Model not found: {model_name}. Please choose the model types of 'openai/whisper-large-v2' or 'facebook/wav2vec2-large-lv60'.")
            
            # hyp_transcript = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            hyp_transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

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
#--------------------------------------------------
def get_transcripts_wav2vec2(
    dataset_dir="../datasets/switchboard/whisper/",
    model_name="facebook/wav2vec2-large-lv60",
    model_type="wav2vec2",
    pretrained_model_dir="../ref_models/pre_trained",
    output_dir="../alignment_transcripts/wav2vec2",
):
    """
    # ... (rest of your docstring) ...
    """

    logging.set_verbosity_error()  # Suppress warnings

    print(f"Evaluate Model - {model_name} \n")
    print(f"Dataset Directory: {dataset_dir} \n")

    # check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------
    # LOAD PRETRAINED MODEL + PROCESSOR
    # ---------------------------------------
    processor = None
    model = None

    if model_name.startswith("facebook/wav2vec2"):
        processor = Wav2Vec2Processor.from_pretrained(
            model_name, cache_dir=pretrained_model_dir, local_files_only=True
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            model_name, cache_dir=pretrained_model_dir, local_files_only=True
        )

    if model is not None:
        print(f"Model {model_name} loaded successfully!")
        model.to(device)
    else:
        raise ValueError(
            f"Model not found: {model_name}. Please choose the model types of 'facebook/wav2vec2-*'."
        )

    # ================================================================================================
    #                                         LOAD DATASET
    # ================================================================================================
    try:
        # ../datasets/switchboard/whisper/swb_test
        swb_test = load_from_disk(dataset_dir)

    except FileNotFoundError:
        raise ValueError(
            f"Dataset not found: {dataset_dir}. Please choose the correct path for the test dataset."
        )

    # ==============================================================================================
    #                             CTC-BASED EVALUATION
    # ==============================================================================================
    
    # Create a CTC decoder
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
    decoder = ctcdecode.CTCBeamDecoder(
        list(sorted_vocab_dict.keys()),
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=processor.tokenizer.pad_token_id,
        log_probs_input=True
    )

    ctc_losses = []
    ref, hyp = [], []
    normalised_ref, normalised_hyp = [], []
    i = 0

    for recording in swb_test:
        i += 1
        print(f"Recording: {i} / {len(swb_test)}")

        try:
            # -------------------------------------------------------------------------------------------------
            #                                         ORIGINAL REF TRANSCRIPTS
            # -------------------------------------------------------------------------------------------------
            original_ref = recording["transcript"]
            print(f"REF: {original_ref}")
            ref.append(original_ref)

            # -------------------------------------------------------------------------------------------------
            #                                         NORMALISED REF TRANSCRIPTS
            # -------------------------------------------------------------------------------------------------
            normalised_ref_transcript = transform_alignment_sentence(original_ref)
            print(f"NORMED REF: {normalised_ref_transcript}")

            normalised_ref.append(normalised_ref_transcript)

            # -------------------------------------------------------------------------------------------------
            #                                         AUDIO PREPROCESSING
            # -------------------------------------------------------------------------------------------------
            audio = recording["audio"]["array"]
            sr = recording["audio"]["sampling_rate"]

            if sr != 16000:
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)

            # -------------------------------------------------------------------------------------------------
            #                                         MODEL PREDICTION AND CTC LOSS
            # -------------------------------------------------------------------------------------------------
            inputs = processor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(device)

            # Get the logits
            with torch.no_grad():
                logits = model(input_values).logits

            # Convert ground truth transcript to token IDs
            target_ids = processor.tokenizer(original_ref, add_special_tokens=False, return_tensors="pt").input_ids
            target_ids = target_ids.to(device)

            # CTC loss calculation
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            input_lengths = torch.full(
                size=(log_probs.shape[0],),
                fill_value=log_probs.shape[1],
                dtype=torch.long
            )
            target_lengths = torch.full(
                size=(target_ids.shape[0],),
                fill_value=target_ids.shape[1],
                dtype=torch.long
            )

            ctc_loss = torch.nn.functional.ctc_loss(
                log_probs.transpose(0, 1),
                target_ids,
                input_lengths,
                target_lengths,
                reduction="mean"
            )

            ctc_losses.append(ctc_loss.item())
            print(f"CTC Loss: {ctc_loss.item()}")

            # -------------------------------------------------------------------------------------------------
            #                                         BEAM SEARCH DECODING
            # -------------------------------------------------------------------------------------------------

            beam_results, beam_scores, timesteps, out_lens = decoder.decode(log_probs)
            # beam_results shape: (batch_size, num_beams, num_tokens)
            # beam_scores shape: (batch_size, num_beams)
            # timesteps shape: (batch_size, num_beams)
            # out_lens shape: (batch_size, num_beams)

            # Get the top beam for each item in the batch
            top_beam_results = beam_results[:, 0, :]
            top_beam_lens = out_lens[:, 0]

            # Convert the top beam results to text
            batch_texts = []
            for result, length in zip(top_beam_results, top_beam_lens):
                text = processor.decode(result[:length])
                batch_texts.append(text)

            # In this example, batch_texts will contain a single transcript since the batch size is 1
            hyp_transcript = batch_texts[0]

            print(f"HYP: {hyp_transcript}")
            hyp.append(hyp_transcript)

            # -------------------------------------------------------------------------------------------------
            #                                         NORMALISED HYP TRANSCRIPTS
            # -------------------------------------------------------------------------------------------------
            hyp_transcript = transform_number_words(hyp_transcript, reverse=True)
            normalised_hyp_transcript = transform_alignment_sentence(hyp_transcript)
            print(f"NORMED HYP: {normalised_hyp_transcript}")

            normalised_hyp.append(normalised_hyp_transcript)

        except Exception as e:
            print(
                f"Error: {e}. Unable to process recording {i}. Please check the audio file and transcript."
            )
            continue

    avg_ctc_loss = sum(ctc_losses) / len(ctc_losses)
    print(f"Average CTC Loss: {avg_ctc_loss}")

    print(
        "Finished processing all the recordings. Outputing to different transcript lists."
    )
    return (
        ref,
        hyp,
        normalised_ref,
        normalised_hyp,
        avg_ctc_loss
    )




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
    ref_file = os.path.join(args.output_dir, f"original_ref_{args.model_type}.txt") #original_ref_whisper.txt
    write_transcript(
        ref_file, 
        ref, 
        transcript_type="ref")
    #---------------------------------------------------------------------------------------------------------

    # Original Hypothesis Transcripts
    hyp_file = os.path.join(args.output_dir, f"original_hyp_{args.model_type}.txt") #original_hyp_whisper
    write_transcript(
        hyp_file, 
        hyp, 
        transcript_type="hyp")
    #---------------------------------------------------------------------------------------------------------
    
    # Normalised Reference Transcripts
    normed_ref_file = os.path.join(args.output_dir, f"normalised_ref_{args.model_type}.txt") #normalised_ref_whisper.txt
    write_transcript(
        normed_ref_file, 
        normed_ref, 
        transcript_type="normalised ref")
    #---------------------------------------------------------------------------------------------------------
    
    # Normalised Hypothesis Transcripts
    normed_hyp_file = os.path.join(args.output_dir, f"normalised_hyp_{args.model_type}.txt") #normalised_hyp_whisper.txt
    write_transcript(
        normed_hyp_file, 
        normed_hyp, 
        transcript_type="normalised hyp")
    #---------------------------------------------------------------------------------------------------------
    
    # Alignment Transcripts
    alignment_file = os.path.join(args.output_dir, f"alignment_{args.model_type}.txt") #alignment_whisper.txt
    write_alignment_transcript(
        alignment_file,
        model_type=args.model_type,
        alignment_ref=normed_ref,
        alignment_hyp=normed_hyp
    )
    
    end_time = time.time()
    print(f"Finished! Total runtime: {end_time - start_time} seconds")