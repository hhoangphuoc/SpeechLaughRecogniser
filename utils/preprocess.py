import librosa
from tqdm import tqdm
import pandas as pd
import torchaudio
import librosa
import re
import os
import argparse

from datasets import load_dataset

from transcript_process import *
from audio_process import *
    
# 2. Combined audio and transcript into csv files
def switchboard_to_df(
    data_name="switchboard", #also implement for AMI, VocalSound, LibriSpeech, ...
    audio_dir='../data/switchboard/audio_wav', 
    transcript_dir='../data/switchboard/transcripts',
    batch_audio=[],
    batch_sr = [],
    batch_transcript=[],
    csv_dir = "../datasets/",
    to_csv = True,
):
    """
    Combines audio files and their corresponding transcripts into a batch

    Args:
        data_name (str): Name of the dataset
        audio_dir (str): Path to the directory containing audio files.
        transcript_dir (str): Path to the root directory containing transcript subfolders.
        batch_audio (list): List of path to audio file segments
        batch_transcript (list): List of transcript segments

    Returns:
        batch (dict): {"audio": batch_audio, "sample", "transcript": batch_transcript}
    """

    for audio_file in tqdm(os.listdir(audio_dir), desc="Combining audio and transcript..."):
        if audio_file.endswith(".wav"):
            audio_path = os.path.join(audio_dir, audio_file) #audio_wav/sw02001A.wav
            transcript_lines = process_switchboard_transcript(audio_file) #produce the transcript lines for each corresponding audio
            if transcript_lines is not None:
                audio_file_segments, transcripts_segments = cut_audio_based_on_transcript_segments(
                audio_path, 
                transcript_lines,
                padding_time=0.3,
                data_name=data_name,
                audio_segments_dir=f"../audio_segments/{data_name}/"
                )
            else:
                print(f"Skipping audio file due to missing transcript: {audio_file}")

        #TODO: Applied Audio Cutting and Segmenting based on transcripts_lines
        #EXPECTED: 
        # audio_file_segments: ["../audio_segments/sw2001A/sw2001A_0.0_0.977625.wav", "../audio_segments/sw2001A/sw2001A_0.977625_11.561375.wav", ...]
        # audio_segments: [array(..., dtype=float32), array(..., dtype=float32), ...]
        # transcripts_segments: ["[silence]", "hi um yeah i'd like to.....", "..."]
                
        # Add to batch
        batch_audio.extend(audio_file_segments)
        batch_sr.extend([16000]*len(audio_file_segments))
        # batch_audio.extend([{"array": segment, "sampling_rate": 16000} for segment in audio_segments])
        batch_transcript.extend(transcripts_segments)
    
    # batch = {
    #     "audio": batch_audio, 
    #     "sampling_rate": batch_sr,
    #     "transcript": batch_transcript
    # }
    print(f"Successfully combined audio and transcript and added to batch")
    
    df = pd.DataFrame({
        "audio": batch_audio, #batch["audio"],
        "sampling_rate": batch_sr, #batch["sampling_rate"],
        "transcript": batch_transcript, #batch["transcript"]
    })

    if to_csv:
        csv_path = os.path.join(csv_dir, data_name)
        os.makedirs(csv_path, exist_ok=True)
        output_file = os.path.join(csv_path, f"{data_name}.csv") #../datasets/switchboard.csv
        df.to_csv(output_file, index=False)

    return df


def vocalsound_to_df(
    data_name="vocalsound",
    batch_audio=[],
    batch_sr = [],
    batch_transcript=[],
    csv_dir = "../datasets/",
    to_csv = True,
    # vocalsound_dataset = None,
):
    """
    Process the vocalsound dataset
    """
    label_to_transcript = {
        "laughter": "[LAUGHTER]",
        "cough": "[COUGH]",
        "sigh": "[SIGH]",
        "sneeze": "[SNEEZE]",
        "sniff": "[SNIFF]",
        "throatclearing": "[THROAT-CLEARING]"
    }

    vocalsound_dataset = load_dataset("flozi00/VocalSound_audio_16k", split="train", cache_dir="../cache_data/")
    
    if vocalsound_dataset is None:
        print("Unable to find VocalSound dataset")
        return 

    for example in tqdm(vocalsound_dataset, desc="Processing VocalSound dataset..."):
        # audio_segment = example["audio"]["array"]
        audio_path = example['audio']['path']
        sampling_rate = example["audio"]["sampling_rate"] 
        label = example["label"]

        # batch_audio.append(audio_segment)
        batch_audio.append(audio_path) #adding audio path instead of audio array
        batch_sr.append(sampling_rate)
        batch_transcript.append(label_to_transcript[label])
        
    # batch = {
    #     "audio": batch_audio,
    #     "sampling_rate": batch_sr,
    #     "transcript": batch_transcript
    #     }
    df = pd.DataFrame({
        "audio": batch_audio, #batch["audio"],
        "sampling_rate": batch_sr, #batch["sampling_rate"],
        "transcript": batch_transcript, #batch["transcript"]
    })
    
    # df = pd.DataFrame({
    #     "audio": batch["audio"],
    #     "sampling_rate": batch["sampling_rate"],
    #     "transcript": batch["transcript"]
    # })

    if to_csv:
        csv_path = os.path.join(csv_dir, data_name)
        os.makedirs(csv_path, exist_ok=True)
        output_file = os.path.join(csv_path, f"{data_name}.csv") #../datasets/switchboard.csv
        df.to_csv(output_file, index=False)

    print(f"Successfully processed VocalSound dataset!")
    return df

def ami_to_df(
    data_name="ami",
    batch_audio=[],
    batch_sr = [],
    batch_transcript=[],
    csv_dir = "../datasets/",
    to_csv = True,
):
    # load the data in here
    ami_dataset = load_dataset("edinburghcstr/ami", "ihm", cache_dir="../cache_data/")
    if ami_dataset is None:
        print("Unable to load ami_dataset")
        
    print("Load ami dataset sucessfully, start processing...")
    print(ami_dataset)
    # only use the trained dataset
    for example in tqdm(ami_dataset['train'], desc="Processing AMI dataset..."):
        # AMI Dataset have the format:
        """
        ami_dataset["train"][0] = 
        {'meeting_id': 'EN2001a',
        'audio_id': 'AMI_EN2001a_H00_MEE068_0000557_0000594',
        'text': 'OKAY',
        'audio': {'path': '/cache/dir/path/downloads/extracted/2d75d5b3e8a91f44692e2973f08b4cac53698f92c2567bd43b41d19c313a5280/EN2001a/train_ami_en2001a_h00_mee068_0000557_0000594.wav',
        'array': array([0.        , 0.        , 0.        , ..., 0.00033569, 0.00030518,
                0.00030518], dtype=float32),
        'sampling_rate': 16000},
        'begin_time': 5.570000171661377,
        'end_time': 5.940000057220459,
        'microphone_id': 'H00',
        'speaker_id': 'MEE068'
        }
        """
        # Process audio:
        try: 
            audio_path = example["audio"]["path"]
            # audio_array = example["audio"]["array"]
            sampling_rate = example["audio"]["sampling_rate"]
            transcript_line = example["text"]
            # Process transcript (extract timestamps, etc.):
            ami_text = process_ami_transcript(transcript_line)

            if ami_text:  # Check if processing was successful
                # batch_audio.append(audio_array)
                batch_audio.append(audio_path)
                batch_sr.append(sampling_rate)  
                batch_transcript.append(ami_text)
        except (FileNotFoundError, KeyError, TypeError) as e:
            print(f"Warning: Example file not found: {example}")
            continue
    # batch = {
    #     "audio": batch_audio,
    #     "sampling_rate": batch_sr,
    #     "transcript": batch_transcript
    #     }
    
    df = pd.DataFrame({
        "audio": batch_audio, #batch["audio"],
        "sampling_rate": batch_sr, #batch["sampling_rate"],
        "transcript": batch_transcript, #batch["transcript"]
    })

    if to_csv:
        csv_path = os.path.join(csv_dir, data_name)
        os.makedirs(csv_path, exist_ok=True)
        output_file = os.path.join(csv_path, f"{data_name}.csv") #../datasets/switchboard.csv
        df.to_csv(output_file, index=False)
    
    print(f"Successfully processed AMI dataset!")
    return df

def combine_data(
    csv_dir="../datasets",
    dataframes=[],
    train_val_split=True,
    to_csv=True,
    shuffle_ratio=0.8
    ):
    """
    Load all the csv files and combine them into one dataframe
    Return the combined dataframe and output in csv files, splitted into train and validation set
    """
    print("Start combining the dataframes...")
    
    for folder in os.listdir(args.csv_dir):
        folder = os.path.join(args.csv_dir,folder)
        if os.path.isdir(folder) and not folder.endswith(".ipynb_checkpoints"):
            print(folder)
            for file in os.listdir(folder):
                if file.endswith(".csv"):
                    print(f"Loading file {file}")
                    df = pd.read_csv(os.path.join(folder,file))
                    dataframes.append(df)
    print("Get total of {} dataframes".format(len(dataframes)))
    
    try: 
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        #remove missing value rows
        combined_df.dropna(inplace=True)
        
        #FIXME: Temporary solution to avoid mismatching, should be done in transcript_process instead
        combined_df["transcript"] = combined_df["transcript"].apply(lambda x: x.upper() if x == "[laughter]" else x)
        
        #FIXME: In combined_df, drop row that have empty array
        # combined_df = combined_df[combined_df["audio"].apply(lambda x: len(x) > 0)]

        # shuffle
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)

        if not train_val_split:
            print("Not splitting the dataset into train and val sets, only returning the combined dataframe")
            if to_csv:
                combined_df.to_csv(f"{csv_dir}/combined.csv", index=False)
            else:
                # return combined_df
                return combined_df
        else:
            # split the dataset into train and validation set
            train_df = combined_df[:int(len(combined_df)*shuffle_ratio)]
            val_df = combined_df[int(len(combined_df)*shuffle_ratio):]

            if to_csv:
                os.makedirs(csv_dir, exist_ok=True)
                # save to csv
                train_df.to_csv(f"{csv_dir}/train.csv", index=False)
                val_df.to_csv(f"{csv_dir}/val.csv", index=False)
            else:
                print("Not saving to csv, but returning the train, val dataframes")
                # return train_df, val_df
            return train_df, val_df
        print("Successfully generate combined datasets from different data!!")
    except ValueError as e:
        print("Unable to combine the datasets: {}".format(e))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_process", type=bool, default=False, help="Determine to skip or run processing steps for each dataset separately")
    parser.add_argument("--data_names", nargs="+", default=["switchboard", "ami", "vocalsound"], required=False, help="List of the datasets to process")
    parser.add_argument("--csv_dir", type=str, default="../datasets/", help="Path to the directory containing audio files")
    parser.add_argument("--to_csv", type=bool, default=True, help="Whether to save the processed data to csv or not")
    parser.add_argument("--do_combine", type=bool, default=False, help="Determined if you want to combined different datasets into the same file")
    parser.add_argument("--train_val_split", type=bool, default=False, help="Decide whether not want to split the data")
    
#-------------------------------------------------------------------------------------------------------------

    args = parser.parse_args()
    
    
    dataframes = []
    combined = args.do_combine
    
    if not args.skip_process:
        for data_name in args.data_names:
            if data_name == "switchboard":
                df = switchboard_to_df(
                    data_name = data_name,
                    csv_dir = args.csv_dir,
                    to_csv = args.to_csv
                )

            elif data_name == "vocalsound":
                df = vocalsound_to_df(
                    data_name = data_name,
                    batch_audio=[], 
                    batch_transcript=[],
                    csv_dir = args.csv_dir,
                    to_csv = args.to_csv,
                    # vocalsound_dataset = vocalsound_dataset
                )

            elif data_name == "ami":
                # ami_dataset = load_dataset("edinburghcstr/ami", "ihm", split="train")
                # this return a DatasetDict,that contains train, test, val sets
                df = ami_to_df(
                    data_name = data_name,
                    csv_dir = args.csv_dir,
                    to_csv = args.to_csv,
                    # ami_dataset = ami_dataset
                )
    if args.do_combine:
        if args.train_val_split:
            train_df, val_df = combine_data(
                csv_dir=args.csv_dir,
                train_val_split=args.train_val_split,
                to_csv=args.to_csv
            )
        else:
            combined_df = combine_data(
                csv_dir=args.csv_dir,
                train_val_split=args.train_val_split,
                to_csv=args.to_csv
            )
    else:
        print("Not combining any datasets, only separate csv file provides")