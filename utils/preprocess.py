import librosa
from tqdm import tqdm
import pandas as pd
import torchaudio
import librosa
import re
import os

from datasets import load_dataset

from transcript_process import *
from audio_process import *
    
# 2. Combined audio and transcript into csv files
def switchboard_to_batch(
    data_name="switchboard", #also implement for AMI, VocalSound, LibriSpeech, ...
    audio_dir=f'../data/{data_name}/audio_wav', 
    transcript_dir=f'../data/{data_name}/transcripts',
    batch_audio=[],
    batch_transcript=[]):
    """
    Combines audio files and their corresponding transcripts into a batch

    Args:
        data_name (str): Name of the dataset
        audio_dir (str): Path to the directory containing audio files.
        transcript_dir (str): Path to the root directory containing transcript subfolders.
        batch_audio (list): List of audio segments
        batch_transcript (list): List of transcript segments

    Returns:
        batch (dict): {"audio": batch_audio, "transcript": batch_transcript}
    """

    for audio_file in tqdm(os.listdir(audio_dir), desc="Combining audio and transcript..."):
        if audio_file.endswith(".wav"):
            audio_path = os.path.join(audio_dir, audio_file) #audio_wav/sw02001A.wav
            transcript_lines = process_switchboard_transcript(audio_file) #produce the transcript lines for each corresponding audio

        #TODO: Applied Audio Cutting and Segmenting based on transcripts_lines
        #EXPECTED: 
        # audio_file_segments: ["../audio_segments/sw2001A/sw2001A_0.0_0.977625.wav", "../audio_segments/sw2001A/sw2001A_0.977625_11.561375.wav", ...]
        # audio_segments: [array(..., dtype=float32), array(..., dtype=float32), ...]
        # transcripts_segments: ["[silence]", "hi um yeah i'd like to.....", "..."]
        audio_file_segments, audio_segments, transcripts_segments = cut_audio_based_on_transcript_segments(
                audio_path, 
                transcript_lines,
                padding_time=0.3,
                data_name=data_name,
                audio_segments_dir=f"../audio_segments/{data_name}/"
        )
        # Add to batch
        batch_audio.extend([{"array": segment, "sampling_rate": 16000} for segment in audio_segments])
        batch_transcript.extend(transcripts_segments)
    
    batch = {"audio": batch_audio, "transcript": batch_transcript}
    print(f"Successfully combined audio and transcript and added to batch")
    return batch

def vocalsound_to_batch(
    batch_audio=[], 
    batch_transcript=[]):
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

    # for audio, label in tqdm(zip(vocalsound_dataset["audio"], vocalsound_dataset["label"]), desc="Processing vocalsound dataset..."):
    #     batch_audio.append({"array": audio, "sampling_rate": 16000})
    #     batch_transcript.append(label_to_transcript[label])

    # batch = {"audio": batch_audio, "transcript": batch_transcript}
    # print(f"Successfully processed vocalsound dataset!")
    # return batch
    for example in tqdm(vocalsound_dataset, desc="Processing VocalSound dataset..."):
        audio_array = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"] 
        label = example["label"] 
        batch_audio.append({"array": audio_array, "sampling_rate": sampling_rate})
        batch_transcript.append(label_to_transcript[label]) 
    batch = {"audio": batch_audio, "transcript": batch_transcript}
    print(f"Successfully processed VocalSound dataset!")
    return batch

# def ami_to_batch(
#     batch_audio=[], 
#     batch_transcript=[]):
#     """
#     Process the ami dataset
#     """
#     for audio, transcript in tqdm(zip(ami_dataset["audio"], ami_dataset["text"]), desc="Processing ami dataset..."):
#         batch_audio.append({"array": audio, "sampling_rate": 16000})

#         #TODO: process the transcript of AMI dataset
#         transcript = process_ami_transcript(transcript) #Expected: I'VE GOTTEN MM HARDLY ANY -> I'v gotten [MM] hardly any
#         batch_transcript.append(transcript)
    
#     batch = {"audio": batch_audio, "transcript": batch_transcript}
#     print(f"Successfully processed ami dataset!")
#     return batch

def ami_to_batch(batch_audio=[], batch_transcript=[]):
    for example in tqdm(ami_dataset, desc="Processing AMI..."):
        audio_array = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        transcript_line = example["text"]  
        # Process transcript (extract timestamps, etc.):
        transcript_data = process_ami_transcript(transcript_line) 
        if transcript_data:  # Check if processing was successful
            start_time, end_time, transcript_text = transcript_data
            # Segment audio (use start_time, end_time from transcript_data):
            audio_segment = audio_array[int(start_time * sampling_rate): int(end_time * sampling_rate)]
            batch_audio.append({"array": audio_segment, "sampling_rate": sampling_rate})
            batch_transcript.append(transcript_text)
        
    batch = {"audio": batch_audio, "transcript": batch_transcript}
    print(f"Successfully processed AMI dataset!")

    return batch

def prepare_csv_data(
    data_name, 
    to_csv=False):
    """
    Prepare the specific dataset to be match the same input
    for the batch
    Output the csv file and return the dataframe
    """

    if data_name == "switchboard":
        batch = switchboard_audio_transcript_to_batch(data_name=data_name)

    if data_name == "vocalsound":
        vocalsound_dataset = load_dataset("flozi00/VocalSound_audio_16k", split="train")
        batch = vocalsound_to_batch(batch_audio=[], batch_transcript=[])

    elif data_name == "ami":
        ami_dataset = load_dataset("edinburghcstr/ami", "ihm", split="train")
        batch = ami_to_batch(batch_audio=[], batch_transcript=[])
    else:
        raise ValueError(f"Dataset {data_name} is not supported")
    
    df = pd.DataFrame({
        "audio": batch["audio"],
        "transcript": batch["transcript"]
    })

    if to_csv:
        csv_dir=f'../datasets/{data_name}'
        os.makedirs(csv_dir, exist_ok=True)
        output_file = os.path.join(csv_dir, f"{data_name}.csv") #../datasets/switchboard.csv
        df.to_csv(output_file, index=False)

    return df

if __name__ == "__main__":
    #PROCESS SWITCHBOARD DATASET
    switchboard_df = prepare_csv_data(data_name="switchboard", to_csv=True) #switchboard.csv
    vocalsound_df = prepare_csv_data(data_name="vocalsound", to_csv=True) #vocalsound.csv
    ami_df = prepare_csv_data(data_name="ami", to_csv=True) #ami.csv


    # combine all datasets
    combined_df = pd.concat([switchboard_df, vocalsound_df, ami_df], ignore_index=True)
    print(combined_df.head())

    # shuffle the dataset
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    print(combined_df.head())

    # save to csv
    combined_df.to_csv("../datasets/train.csv", index=False)

