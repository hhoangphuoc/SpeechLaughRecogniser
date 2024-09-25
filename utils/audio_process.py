import librosa
from tqdm import tqdm
# import torchaudio
import soundfile as sf
import librosa
import re
import os
import torch

# 2. Cut audio based on transcript segments, and extend the csv file
def cut_audio_based_on_transcript_segments(
    audio_path, #path to original audio to be segmented
    transcript_lines, #list of tuples: (start_time, end_time, text)
    padding_time=0.2, #seconds
    data_name="switchboard",
    #audio_segments_dir = f"../audio_segments/{data_name}"):
    audio_segments_dir=""):
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
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)

    filename = os.path.basename(audio_path).split(".")[0] #sw02001A
    #Extract timestamps from transcript_lines
    audio_file_segments = []
    audio_segments = []
    transcripts_segments = []
    if audio_segments_dir is None:
        audio_segments_dir = f"../audio_segments/{data_name}"
    
    os.makedirs(audio_segments_dir, exist_ok=True)

    os.makedirs(audio_segments_dir, exist_ok=True)
    #transcript_line format: (start_time, end_time, text)
    for start_time, end_time, text in transcript_lines:

        # segmenting audio sample
        start_sample = librosa.time_to_samples(start_time, sr=sr)
        end_sample = librosa.time_to_samples(end_time, sr=sr)
        # Calculate padding samples
        padding_samples = int(padding_time * sr)

        # Add padding to start_sample and end_sample
        start_sample_padded = max(0, start_sample - padding_samples)
        end_sample_padded = min(len(audio), end_sample + padding_samples)

        # Cut audio segment
        audio_segment = audio[start_sample_padded:end_sample_padded]
        
        # audio_segment = audio[int((start_time-padding_time)*sr):int((end_time+padding_time)*sr)] #the audio segment for specific text
        
        # Convert audio_segment to a 2D tensor:
        # audio_segment_tensor = torch.tensor(audio_segment).unsqueeze(0)  # Add channel dimension

        #save the audio segment to corresponding folder
        start_time_str=str(start_time).replace(".","")
        end_time_str=str(end_time).replace(".","")
        output_file = f"{audio_segments_dir}/{filename}_{start_time_str}_{end_time_str}.wav"

        # torchaudio.save(output_file, audio_segment_tensor, sr)  # FIXME: NOT SAVE AS the tensor
        sf.write(output_file, audio_segment, sr)

        #append to list
        audio_file_segments.append(output_file)
        # audio_segments.append(audio_segment)
        transcripts_segments.append(text)

        
    #list of audio segments (list of array) and transcripts (list of text)
    return audio_file_segments, transcripts_segments #audio_segments
