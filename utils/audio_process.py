import librosa
from tqdm import tqdm
import numpy as np
import soundfile as sf
import os

# 2. Cut audio based on transcript segments, and extend the csv file
def cut_audio_based_on_transcript_segments(
    audio_path, #path to original audio to be segmented
    transcript_lines, #list of tuples: (start_time, end_time, text)
    padding_time=0.2, #seconds
    data_name="switchboard",
    audio_segments_directory=""):
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
    
    os.makedirs(audio_segments_directory, exist_ok=True)

    #transcript_line format: (start_time, end_time, text)
    for start_time, end_time, text in transcript_lines:
        # if the text is empty string or None, skip
        if text is None or not text.strip():
            continue
        # if not text.strip():
        #     continue

        # segmenting audio sample
        start_sample = librosa.time_to_samples(start_time, sr=sr)
        end_sample = librosa.time_to_samples(end_time, sr=sr)
        
        # Calculate padding samples and add these to start and end
        padding_samples = int(padding_time * sr)
        start_sample_padded = max(0, start_sample - padding_samples)
        end_sample_padded = min(len(audio), end_sample + padding_samples)

        # Cut audio segment
        audio_segment = audio[start_sample_padded:end_sample_padded]

        #check if the audio segment is empty, skip
        if len(audio_segment) == 0:
            continue

        #convert audio segment to numpy array
        if type(audio_segment) is not np.ndarray:
            audio_segment = np.array(audio_segment)

        #save the audio segment to corresponding folder
        start_time_str=str(start_time).replace(".","")
        end_time_str=str(end_time).replace(".","")
        output_file = f"{audio_segments_directory}/{filename}_{start_time_str}_{end_time_str}.wav"

        sf.write(output_file, audio_segment, sr)

        #append to list
        audio_file_segments.append(output_file)
        audio_segments.append(audio_segment)
        transcripts_segments.append(text)

        
    #list of audio segments path and transcripts (list of text)
    # return audio_file_segments, transcripts_segments
    return audio_file_segments, audio_segments, transcripts_segments

def preprocess_noise(noise_audio, noise_sr, target_sr=16000):
    """
    Preprocesses the noise audio:
        - Resamples to the target sample rate (if necessary).
        - Normalizes the audio to the range [-1, 1].
        - Optionally applies other preprocessing (e.g., filtering).

    Args:
        noise_audio (np.ndarray): The raw noise audio waveform.
        target_sr (int, optional): The target sample rate. Defaults to 16000.

    Returns:
        np.ndarray: The preprocessed noise audio.
    """

    # 1. Resample to target sample rate (if needed)
    if noise_sr != target_sr:
        noise_audio = librosa.resample(y=noise_audio, orig_sr=noise_sr, target_sr=target_sr)
        noise_sr = target_sr
    if len(noise_audio.shape) > 1:
        noise_audio = noise_audio[:,0] # use the first channel if the input is multi-channel
    if not isinstance(noise_audio, np.ndarray):
        noise_audio = np.array(noise_audio)

    if noise_audio.dtype == 'int16':
        noise_audio = noise_audio.astype(np.float32) / 32767  # Normalize int16 to [-1, 1]
    else:
        noise_audio = librosa.util.normalize(noise_audio)  # Normalize to [-1, 1]

    return noise_audio