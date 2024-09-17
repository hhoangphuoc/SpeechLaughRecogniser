# This file using to process the transcript to get the word-level timestamps for Whisper fine-tuning.
# The file includes:
# 1. Merging multiple transcripts into a single transcript.
# 2. Retokenization of a given transcript
# 3. Extract the word-level timestamps
# 4. matching timestamps with the audio segments

import re
import os
import openai #for clean and format transcripts


# SETS OF SPECIAL PATTERNS FOR TRANSCRIPT TOKENS:
word_pattern = r"\b\w+\b"
pause_pattern = r"\[silence\]"
noise_pattern = r"\[noise\]"
filler_pattern = r"\b(uh|um|mm|uh[ -]huh|ah|hmm+|yeah|well)\b"
speech_laugh_pattern = r"\[laughter-(\w+)\]"
vocalsound_pattern = r"\b([laughter]|[cough]|[sigh]|[sniff]|[throatclearing]|[sneeze])\b"

#-----------------------------------------------------#

def retokenize_and_extract_timestamps_from_transcript(transcript):
    """
    Retokenizes a list of transcripts and extracts word-level timestamps 
    also remove the transcript lines that have "[silence]"

    Args:
        transcript(list): A list of transcripts, each containing multiple sentences.
        eg:
        ["sw2001A-ms98-a-0001 0.000000 0.977625 [silence]", "sw2001A-ms98-a-0002 0.977625 11.561375 hi um yeah i'd like to.....","..."]

    Returns:
        list: A list of lists, each containing retokenized sentences with timestamps.
        list: A list of tuples, each containing a word and its corresponding audio segment.
    """

    all_tokens_with_timestamps = []
    all_audio_segments = []

    # for line in transcript_lines:
    #     tokens_with_timestamps = []
    #     audio_segments = []

    for line in transcript.splitlines():
        if not line.strip():  # Skip empty lines
            continue

        _, start_time, end_time, token = line.split(" ") #split by space -> #["sw2001A-ms98-a-0001", "0.000000", "0.977625", "[silence]"]
        start_time, end_time = float(start_time), float(end_time)

        if re.match(word_pattern, token):
            token_type = "word"
            audio_segments.append((token, token_type, (start_time, end_time)))
        elif re.match(pause_pattern, token):
            token_type = "pause"
            audio_segments.append((token, token_type, (start_time, end_time)))
        elif re.match(noise_pattern, token):
            token_type = "noise"
            audio_segments.append((token, token_type, (start_time, end_time)))
        elif re.match(filler_pattern, token):
            token_type = "filler"
            audio_segments.append((token, token_type, (start_time, end_time)))
        elif match := re.match(speech_laugh_pattern, token):
            token_type = "speech_laugh"
            token = match.group(1) + "_speech_laugh"
            audio_segments.append((token, token_type, (start_time, end_time)))

        tokens_with_timestamps.append((token, token_type, (start_time, end_time)))

    all_tokens_with_timestamps.append(tokens_with_timestamps)
    all_audio_segments.extend(audio_segments)

    return all_tokens_with_timestamps, all_audio_segments

    # Read the file
    with open(words_transcript, 'r') as file:
        lines = file.readlines()
    audio_segments = []
    # Process each line
    for line in lines:
        _, start_time, end_time, token = line.split()
        start_time, end_time = float(start_time), float(end_time)
        audio_segments.append((token, (start_time, end_time)))
    return audio_segments

def retokenize_transcript_pattern(transcript_line):
    """
    Retokenize the transcript line based on the given pattern.
    The rules are:
    - Remove the line that has only "[silence]" or "[noise]"
    - Uppercase the line that only contains vocalsound
    - For the word in the line that matches filler, speech_laugh, noise:
        - Ignore the noise
        - Uppercase the filler
        - Replace the speech_laugh with the corresponding token
    """
    if transcript_line.strip() == "[silence]" or transcript_line.strip() == "[noise]":
        continue #ignore the line
    elif re.match(vocalsound_pattern, transcript_line):
        #if the line only contains vocalsound
        transcript_line = transcript_line.upper()
            # new_transcript_lines.append(line)
    else:
        # transform the word in the line when it matches the pattern
        #filler, speech_laugh, noise inside line
        new_line = ""
        for word in transcript_line.split():
            if re.match(noise_pattern, word):
                continue #ignore the noise
            elif re.match(filler_pattern, word): #uh, um,...
                word = "[" + word.upper() + "]" # [UH], [UM], [MM], [YEAH],...
            elif re.match(speech_laugh_pattern, word):
                #TODO: Find a way to annotate the speech_laugh as a token
                token = match.group(1) + "_speech_laugh"
                word = "[" + token + "]" # [LAUGHTER_speech_laugh]
            else:
                #normal word
                word = word
            new_line += word + " "
        transcript_line = new_line.strip()
    return transcript_line

#------------------------
# Functions created for separate datasets
def process_switchboard_transcript(audio_file):
    filename = audio_file.split('.')[0] #sw02001A
    speaker = filename[-1] #A or B
    file_prefix = filename[3:-1] #2001
    subfolder1 = file_prefix[:2] #20
    subfolder2 = file_prefix
    transcript_file = f"sw{file_prefix}{speaker}-ms98-a-trans.text"
    transcript_path = os.path.join(transcript_dir, subfolder1, file_prefix, transcript_file)

    with open(transcript_path, 'r') as f:
        transcript_lines = f.readlines()
    
    switchboard_pattern = r"sw\S+ (\d+\.\d+) (\d+\.\d+) (.*)"
    new_transcript_lines = []
    for line in transcript_lines:
        
        match = re.match(switchboard_pattern, line) #sw.. <start_time> <end_time> <text>
        start_time, end_time, text = float(match.group(1)), float(match.group(2)), match.group(3)
        text = retokenize_transcript_pattern(text)
        #TODO: apply cleaning with GPT-4o
        # text = clean_transcript(text)

        # TODO: get audio segment with adding padding_time seconds to start and end
        # audio_segment = audio[int((start_time-padding_time)*sr):int((end_time+padding_time)*sr)] #the audio segment for specific text

        new_transcript_lines.append((start_time, end_time, text))

    return new_transcript_lines

def process_ami_transcript(transcript_line):
    """
    Process the transcript of AMI dataset
    """
    transcript = transcript_line.lower()
    transcript = retokenize_transcript_pattern(transcript) #expected to change the filler token in the transcript

    #TODO: apply cleaning with GPT-4o
    # transcript = clean_transcript(transcript)
    return transcript


# Clean and format transcripts
def clean_transcript(transcript):
    # Could apply GPT-4o or other LLMs to clean and format transcripts
    return transcript
