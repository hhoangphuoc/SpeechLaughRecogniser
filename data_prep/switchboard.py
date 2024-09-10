# Processing the Switchboard audio data to be used for model
import argparse
import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

def process_switchboard_audio(
    data_dir="../data/switchboard/audio_wav", 
    output_dir="../switchboard/audio"):
    """
    Process the Switchboard data to be used for model
    Input: original audio data source
    Output: folder for processed data
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # TODO: Process audio data by 
    # rename the file to specify format
    # resampling to 16kHz
    for file in tqdm(os.listdir(data_dir), desc="Resampling switchboard audio..."):
        # Load the audio file
        audio, sr = librosa.load(os.path.join(data_dir, file), sr=16000)
        # Save the audio file
        sf.write(os.path.join(output_dir, file), audio, sr)

    print("Switchboard audio processed successfully")


def process_switchboard_transcript(
    data_dir="../data/switchboard/transcripts", 
    output_dir="../switchboard/transcripts"):
    """
    Process the Switchboard data to be used for model
    Input: original transcripts data source
    Output: folder for processed data
    """
    pass
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process Switchboard data")
    parser.add_argument("--data_dir", type=str, default="../data/switchboard", help="Directory of the data source (can be audio or transcripts)")
    parser.add_argument("--output_dir", type=str, default="../switchboard", help="Directory of the processed data")
    parser.add_argument("--process_audio", type=bool, default=False, help="Determine to process audio data or not")
    parser.add_argument("--process_transcript", type=bool, default=False, help="Determine to process transcripts data or not")
    
    args = parser.parse_args()

    if args.process_audio:
        data_dir = os.path.join(args.data_dir, "audio")
        output_dir = os.path.join(args.output_dir, "audio")
        process_switchboard_audio(data_dir, output_dir)

    if args.process_transcript:
        data_dir = os.path.join(args.data_dir, "transcripts")
        output_dir = os.path.join(args.output_dir, "transcripts")
        process_switchboard_transcript(data_dir, output_dir)






