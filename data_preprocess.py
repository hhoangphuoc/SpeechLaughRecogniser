import librosa
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch

from datasets import load_dataset, Dataset, DatasetDict, Audio
from preprocess import (
    process_switchboard_transcript, 
    cut_audio_based_on_transcript_segments,
    filter_laughter_dataset,
    filter_intext_laughter_dataset,
    filter_speech_laugh_dataset,
    filter_speech_dataset
)
import utils.params as prs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#===================================================================
#           PROCESS A CSV FILE TO A HUGGINGFACE DATASET
#===================================================================
def csv_to_dataset(csv_input_path):
    """
    Load the dataset from the csv file and convert to HuggingFace Dataset object
    Args:
    - csv_input_path: path to the csv file (train.csv, eval.csv)
    Return:
    - dataset: HuggingFace Dataset object
    """

    df = pd.read_csv(csv_input_path)

    df["sampling_rate"] = df["sampling_rate"].apply(lambda x: int(x))
    #shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    
    #Resample the audio_array column if it not 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    return dataset

#===================================================================
#           PROCESSING A CORPUS TO A DATASET / CSV FILE
#===================================================================
def switchboard_to_ds(
    data_name="switchboard", #also implement for AMI, VocalSound, LibriSpeech, ...
    audio_dir='/switchboard_data/switchboard/audio_wav', #FIXME - This ./switchboard_data is GLOBALLY: ~deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data/...
    transcript_dir='/switchboard_data/switchboard/audio_wav',
    audio_segment_dir='/switchboard_data/audio_segments',
    batch_audio=[],
    batch_sr = [],
    batch_transcript=[],
    dataset_dir = "../datasets/switchboard/",
    retokenize_type = "speechlaugh",
    to_csv = False,
    to_dataset = False,
):
    """
    Combines audio files and their corresponding transcripts into
    - a dataframe and save to csv if the to_csv flag is set
    - a HuggingFace Dataset object if the to_dataset flag is set

    Args:
        data_name (str): Name of the dataset
        audio_dir (str): Path to the directory containing audio files.
        transcript_dir (str): Path to the root directory containing transcript subfolders.
        batch_audio (list): List of path to audio file segments
        batch_transcript (list): List of transcript segments

    Returns:
        - switchboard_dataset (HuggingFace Dataset): Dataset object containing the audio and transcript data
        - OR df (pd.DataFrame): Dataframe containing the audio and transcript data
    """

    print(f"Flags: \n--to_csv: {to_csv}; \n--to_dataset: {to_dataset}; \n--retokenize_type: {retokenize_type}; \n--audio_segment_dir: {audio_segment_dir}; \n--dataset_dir: {dataset_dir}")

    for audio_file in tqdm(os.listdir(audio_dir), desc="Processing Switchboard dataset..."):
        if audio_file.endswith(".wav"):
            audio_path = os.path.join(audio_dir, audio_file) #audio_wav/sw02001A.wav
            transcript_lines = process_switchboard_transcript(
                audio_file,
                transcript_dir=transcript_dir,
                retokenize_type=retokenize_type
            )
            # ==================================== THE TRANSCRIPT LINES CAN HAVE 3 TYPE ==========================================
            #                           1. just speech - normal transcript that has no special token  
            #                           2. speechlaugh - transcript that has special token: WORD
            #                           3. laugh - transcript that has special token: [LAUGH]
            # ====================================================================================================================
            if transcript_lines is not None:
                audio_file_segments, audio_segments, transcripts_segments = cut_audio_based_on_transcript_segments(
                audio_path, 
                transcript_lines,
                padding_time=0.01, #seconds~ 10ms padded both sides #default: 0.2s
                data_name=data_name, #switchboard
                audio_segments_directory=audio_segment_dir, #../datasets/switchboard_data/short_padded_segments
                )
            else:
                print(f"Skipping audio file due to missing transcript: {audio_file}")
                continue
            
            # Append to the batch for each audio file
            batch_audio.extend(audio_file_segments)
            batch_sr.extend([16000]*len(audio_file_segments))
            batch_transcript.extend(transcripts_segments)

    print(f"Successfully combined audio and transcript segments for [{data_name}] data")
    print(f"Start creating dataset...")
    df = pd.DataFrame({
        "audio": batch_audio, #batch["audio"],
        "sampling_rate": batch_sr, #batch["sampling_rate"],
        "transcript": batch_transcript, #batch["transcript"]
    })


    if to_dataset:
        print(f"Saving {dataset_dir}/{data_name}_dataset to HuggingFace Dataset on disk...")
        switchboard_dataset = Dataset.from_pandas(df)
        switchboard_dataset = switchboard_dataset.cast_column("audio", Audio(sampling_rate=16000))

        #======================= FILTER OUT THE DATASET THAT CORRESPOND TO THE SPECIFIC RETOKENIZATION TYPE =======================
        if retokenize_type == "speechlaugh":
            switchboard_dataset = filter_speech_laugh_dataset(switchboard_dataset)
        elif retokenize_type == "speech":
            switchboard_dataset = filter_speech_dataset(switchboard_dataset)    
        elif retokenize_type == "laugh":
            switchboard_dataset = filter_laughter_dataset(
                dataset=switchboard_dataset,
                intext=True # filter out the sentences that only contain [LAUGH]    
            )

        #=======================================================================================================================================

        # Save the dataset to disk
        switchboard_dataset.save_to_disk(
            dataset_path=f"{dataset_dir}/{data_name}_dataset", #swb_speechlaugh_dataset
            num_proc=8 # working on CPU so try num_proc=8 for 8 cores
        )
    if to_csv:
        # csv_path = os.path.join(dataset_dir, data_name)
        os.makedirs(dataset_dir, exist_ok=True)
        output_file = os.path.join(dataset_dir, f"{data_name}.csv") #../datasets/switchboard.csv
        df.to_csv(output_file, index=False)

    return switchboard_dataset if to_dataset else df
#-------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_process", type=bool, default=False, help="Determine to skip or run processing steps for each dataset separately")
    parser.add_argument("--data_names", nargs="+", default=["switchboard", "ami", "vocalsound"], required=False, help="List of the datasets to process")
    parser.add_argument("--audio_segment_name", type=str, default="swb_speechlaugh", help="Name of the audio segment directory")

    parser.add_argument("--global_data_dir", type=str, default="/deepstore/datasets/hmi/speechlaugh-corpus/", help="Path to the directory containing original data")
    parser.add_argument("--dataset_dir", type=str, default="../datasets/switchboard/", help="Path to the directory that store the Arrow, or Path to the actual directory to direct the dataset to actual storage.")
    
    parser.add_argument("--to_csv", type=bool, default=False, help="Save the processed data to csv. Better for visualisation")
    parser.add_argument("--to_dataset", type=bool, default=False, help="Decide whether to return the HuggingFace Dataset. Better for training")
    parser.add_argument("--retokenize_type", type=str, default="speechlaugh", help="Decide whether to retokenize to [LAUGH] or WORD, or normal speech")     # ARGUMENTS FOR SPECIAL PROCESSING

#-------------------------------------------------------------------------------------------------------------

    args = parser.parse_args()
    
    # combined = args.do_combine
    dataset_dir = args.dataset_dir # DATASET DIRECTORY TO ACCESS
    global_data_dir = args.global_data_dir #/deepstore/datasets/hmi/speechlaugh-corpus/
    
    if not args.skip_process:
        for data_name in args.data_names:
            if data_name == "switchboard":
                # /deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data/short_padded_segments/
                audio_segment_dir = os.path.join(
                    global_data_dir, 
                    "switchboard_data", 
                    # "short_padded_segments"
                    # "swb_laugh"
                    args.audio_segment_name
                ) #FIXME: Change back to audio_segments
  
                if args.retokenize_type == "speechlaugh":
                    print("Processing Speech Laugh Switchboard... (special token: WORD)")
                    dataset_dir = os.path.join(dataset_dir, "swb_speechlaugh")
                elif args.retokenize_type == "laugh":
                    print("Processing Laugh Switchboard... (special token: [LAUGH])")
                    dataset_dir = os.path.join(dataset_dir, "swb_laugh")
                elif args.retokenize_type == "speech":
                    print("Processing Normal Speech Switchboard... (special token: None)")
                    dataset_dir = os.path.join(dataset_dir, "swb_speech")
                print(f"Process with: \n -Audio segment directory: {audio_segment_dir}; \n -Data directory: {dataset_dir}")
                df = switchboard_to_ds(
                    data_name = data_name,
                    audio_dir=os.path.join(global_data_dir, "switchboard_data", "switchboard","audio_wav"),
                    transcript_dir=os.path.join(global_data_dir, "switchboard_data", "switchboard","transcripts"),
                    audio_segment_dir=audio_segment_dir,
                    dataset_dir = dataset_dir,
                    to_dataset=args.to_dataset,
                    to_csv = args.to_csv,
                    retokenize_type=args.retokenize_type,
                )