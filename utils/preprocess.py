import librosa
from tqdm import tqdm
import pandas as pd
import torchaudio
import librosa
import re
import os

# Clean and format transcripts
def clean_transcript(transcript):
    # TODO: Clean and format transcripts
    # Could apply GPT-4o or other LLMs to clean and format transcripts
    return transcript

# 1. Combined audio and transcript into csv files
def combine_audio_and_transcript(
    audio_dir='../data/switchboard/audio_wav', 
    transcript_dir='../data/switchboard/transcripts', 
    csv_dir='../datasets/switchboard/'
):
    """
    Combines audio files and their corresponding transcripts into separate CSV files 
    named after the audio files.

    Args:
        audio_dir (str): Path to the directory containing audio files.
        transcript_dir (str): Path to the root directory containing transcript subfolders.
        output_dir (str): Path to the directory where CSV files will be saved.

    Returns:
        None: The function saves the combined data to CSV files in the output directory.
    """

    for audio_file in tqdm(os.listdir(audio_dir), desc="Combining audio and transcript..."):
        if audio_file.endswith(".wav"):
            audio_path = os.path.join(audio_dir, audio_file) #audio_wav/sw02001A.wav
            filename = audio_file.split('.')[0] #sw02001A
            speaker = filename[-1] #A or B
            file_prefix = filename[3:-1] #2001
            subfolder1 = file_prefix[:2] #20
        #   subfolder2 = file_prefix
            transcript_file = f"sw{file_prefix}{speaker}-ms98-a-trans.text"
            transcript_path = os.path.join(transcript_dir, subfolder1, file_prefix, transcript_file)

        with open(transcript_path, 'r') as f:
            transcript_lines = f.readlines() #["sw2001A-ms98-a-0001 0.000000 0.977625 [silence]", "sw2001A-ms98-a-0002 0.977625 11.561375 hi um yeah i'd like to.....","..."]
            
            #TODO: Applied Transcript Processing
            # transcript_lines = clean_transcript(transcript_lines)

            #TODO: Applied Audio Cutting and Segmenting based on transcripts_lines
            audio_segments, transcripts_segments = cut_audio_based_on_transcript_segments(
                audio_path, 
                transcript_lines,
                audio_segments_dir=f"../audio_segments/{filename}/"
            ) #audio_segments: ["../audio_segments/sw2001A/sw2001A_0.0_0.977625.wav", "../audio_segments/sw2001A/sw2001A_0.977625_11.561375.wav", ...]
            #transcripts_segments: ["[silence]", "hi um yeah i'd like to.....", "..."]

            #construct dataframe for audio segments and transcripts_segments
            segments_df = pd.DataFrame({
                "audio": audio_segments,
                "transcript": transcripts_segments
            })
            # data = [{"audio": audio_path, "transcript": transcript_lines}]
            # df = pd.DataFrame(data)

            # Create output file name based on audio file name
            output_file = os.path.join(csv_dir, f"{filename}.csv")
            df.to_csv(output_file, index=False)

    print(f"Successfully combined audio and transcript for switchboard dataset")

# 2. Cut audio based on transcript segments, and extend the csv file
def cut_audio_based_on_transcript_segments(
    audio_path,
    transcript_lines,
    padding_time=0.1, #seconds
    audio_segments_dir = "../audio_segments/"):  
    # load audio
    audio, sr = librosa.load(audio_path)

    #resample audio to 16kHz
    if sr != 16000:
        audio = librosa.resample(audio, sr, 16000)

    filename = os.path.basename(audio_path).split(".")[0]
    #Extract timestamps from transcript_lines
    audio_segments = []
    transcripts_segments = []


    for line in transcript_lines:
        #line = "sw2001A-ms98-a-0001 0.000000 0.977625 [silence]"
        match = re.match(r"sw\S+ (\d+\.\d+) (\d+\.\d+) (.*)", line) #sw.. <start_time> <end_time> <text>
        start_time, end_time, text = float(match.group(1)), float(match.group(2)), match.group(3)

        #remove line [silence]
        if text == "[silence]":
            #skip the line
            continue

        #get audio segment with adding padding_time seconds to start and end
        audio_segment = audio[int((start_time-padding_time)*sr):int((end_time+padding_time)*sr)] #the audio segment for specific text

        #TODO: Could apply GPT-4o to refine the text
        # text = clean_transcript(text)

        #create audio_segments_dir if not exists
        os.makedirs(audio_segments_dir, exist_ok=True)

        #save audio segment
        output_file = f"{audio_segments_dir}/{filename}_{start_time}_{end_time}.wav"
        torchaudio.save(output_file, audio_segment, sr)

        #save audio and transcript segment
        audio_segments.append(output_file)
        transcripts_segments.append(text)

    return audio_segments, transcripts_segments


#-----------------------------------








