
import torchaudio
import os
import librosa

def cut_audio_based_on_transcript_segments(
    csv_path, #path to csv file with transcript and timestamps
    audio_path, #original audio
    transcript_path, #corresponding transcript with timestamps
    output_dir #directory to save the audio segments
):
    # Load audio file
    audio, sr = torchaudio.load(audio_path)

    # Load transcript
    with open(transcript_path, 'r') as file:
        transcript = file.read()

    # Process transcript to get segments
    audio_segments = []
    current_start = 0
    for segment in transcript.split():
        if segment.startswith('['):
            # Skip silence segments
            continue
        end_time = segment.split(' ')[1]
        # Load audio file
