import librosa
from tqdm import tqdm
import numpy as np
import soundfile as sf
import os
import torch

# 2. Cut audio based on transcript segments, and extend the csv file
def cut_audio_based_on_transcript_segments(
    audio_path, #path to original audio to be segmented
    transcript_lines, #list of tuples: (start_time, end_time, text)
    padding_time=0.01, #seconds~ 10ms # BE CAREFUL WITH THIS PARAMETER (Before 0.2s is too large)
    sample_rate=16000,
    data_name="switchboard",
    audio_segments_directory=""
    ):
    
    """
    Use to cut audio based on transcript segments,
    and only apply for the dataset which have the transcripts: switchboard, ami
    after segmenting the audio, store the new audio segments in seperate folders
    and align it with the transcripts, in the csv file

    NOTES: Sample rate need to make sure to be the same =16kHz during segmentations
    """ 
    # load audio
    audio, orig_sr = librosa.load(audio_path)

    #resample audio to 16kHz
    if orig_sr != sample_rate:
        audio = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=sample_rate)
    else:
        sample_rate = orig_sr

    filename = os.path.basename(audio_path).split(".")[0] #sw02001A
    #Extract timestamps from transcript_lines
    audio_file_segments = []
    audio_segments = []
    transcripts_segments = []
    
    os.makedirs(audio_segments_directory, exist_ok=True)

    #transcript_line format: (start_time, end_time, text)
    for start_time, end_time, text in transcript_lines:
        # if the text is empty string or None, skip
        print(f"Cut audio {filename} segment:({start_time} - {end_time}) with text: {text}")
        if text is None or not text.strip():
            continue

        #if the text is [silence] or [noise], or silence, or noise, we skip
        if text.strip() == "[silence]" or text.strip() == "[noise]" or text.strip() == "[vocalized-noise]":
            continue

        # KEEP IN MIND THE SAMPLE RATE
        # segmenting audio sample
        start_sample = librosa.time_to_samples(start_time, sr=sample_rate)
        end_sample = librosa.time_to_samples(end_time, sr=sample_rate)
        
        # Calculate padding samples and add these to start and end
        padding_samples = int(padding_time * sample_rate)
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

        sf.write(output_file, audio_segment, sample_rate)

        #append to list
        audio_file_segments.append(output_file)
        audio_segments.append(audio_segment)
        transcripts_segments.append(text)

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

def add_noise(audio_batch, noise_batch):
    """
    Add noise to the input audio batch at a controlled SNR.
    The noise is generated from a random noise dataset.
    """

    # Mix noise with original audio at a controlled SNR (vectorized)
    snr_db = 10  # Adjust as needed
    snr = 10 ** (snr_db / 10)
    signal_power = torch.mean(audio_batch**2)
    noise_power = torch.mean(noise_batch**2)
    scale_factor = torch.sqrt(signal_power / (snr * noise_power))

    # Add noise to the entire batch using tensor operations
    audio_batch = audio_batch + scale_factor * noise_batch

    return audio_batch