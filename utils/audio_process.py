import librosa
from tqdm import tqdm
import torchaudio
import librosa
import re
import os

# 2. Cut audio based on transcript segments, and extend the csv file
def cut_audio_based_on_transcript_segments(
    audio_path, #path to original audio to be segmented
    transcript_lines, #list of tuples: (start_time, end_time, text)
    padding_time=0.3, #seconds
    data_name="switchboard",
    audio_segments_dir = f"../audio_segments/{data_name}"):
    """
    Use to cut audio based on transcript segments,
    and only apply for the dataset which have the transcripts: switchboard, ami
    after segmenting the audio, store the new audio segments in seperate folders
    and align it with the transcripts, in the csv file
    """ 
    # load audio
    audio, sr = librosa.load(audio_path)

    #resample audio to 16kHz
    if sr != 16000:
        audio = librosa.resample(audio, sr, 16000)

    filename = os.path.basename(audio_path).split(".")[0] #sw02001A
    #Extract timestamps from transcript_lines
    audio_file_segments = []
    audio_segments = []
    transcripts_segments = []

    os.makedirs(audio_segments_dir, exist_ok=True)
    #transcript_line format: (start_time, end_time, text)
    for start_time, end_time, text in transcript_lines:
        
        audio_segment = audio[int((start_time-padding_time)*sr):int((end_time+padding_time)*sr)] #the audio segment for specific text
        
        #save the audio segment to corresponding folder
        output_file = f"{audio_segments_dir}/{filename}_{start_time}_{end_time}.wav"
        torchaudio.save(output_file, audio_segment, sr)

        #append to list
        audio_file_segments.append(output_file)
        audio_segments.append(audio_segment)
        transcripts_segments.append(text)

        
    #list of audio segments (list of array) and transcripts (list of text)
    return audio_file_segments, audio_segments, transcripts_segments 
