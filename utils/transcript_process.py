# This file using to process the transcript to get the word-level timestamps for Whisper fine-tuning.
# The file includes:
# 1. Merging multiple transcripts into a single transcript.
# 2. Retokenization of a given transcript
# 3. Extract the word-level timestamps
# 4. matching timestamps with the audio segments

import re
import os
# import openai #for clean and format transcripts


# SETS OF SPECIAL PATTERNS FOR TRANSCRIPT TOKENS:
word_pattern = r"\b\w+\b"
pause_pattern = r"\[silence\]"
noise_pattern = r"\[noise\]"
filler_pattern = r"\b(uh|um|mm|uh[ -]huh|ah|hmm+|yeah|well)\b"
speech_laugh_pattern = r"\[laughter-(\w+)\]"
vocalsound_pattern = r"\b([laughter]|[cough]|[sigh]|[sniff]|[throatclearing]|[sneeze])\b"

#-----------------------------------------------------#
# Clean and format transcripts
def clean_transcript(transcript):
    # TODO: Could apply GPT-4o or other LLMs to clean and format transcripts
    return transcript


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
    #initially, lowercase the entire line
    transcript_line = transcript_line.lower()
    #if the line only contains "[silence]" or "[noise]", ignore the line and not call this function

    #TODO: Apply clean text function to change the transcript into bare text
    # transcript_line = clean_text(transcript_line)

    if transcript_line.strip() == "[silence]" or transcript_line.strip() == "[noise]":
        return #ignore the line and not call this function
    # elif match:= re.match(vocalsound_pattern, transcript_line):
    #     #if the line only contains vocalsound
    #     transcript_line = transcript_line.upper()
            # new_transcript_lines.append(line)
    else:
        # transform the word in the line when it matches the pattern
        new_line = ""
        for word in transcript_line.split():
            if re.match(noise_pattern, word):
                continue  # ignore the noise
            elif re.match(filler_pattern, word):  # uh, um,...
                word = "[" + word.upper() + "]"  # [UH], [UM], [MM], [YEAH],...
            elif match := re.match(speech_laugh_pattern, word):
                # Now 'match' is guaranteed to be defined here
                speech_laugh_token = match.group(1) + "_speech_laugh"
                word = "[" + speech_laugh_token.upper() + "]"  # [LAUGHTER_speech_laugh]
            elif match := re.match(vocalsound_pattern, word):
                vocalsound_token = match.group(1)
                word = "[" + vocalsound_token.upper() + "]"  # expected [LAUGHTER]|[COUGH],....
            else:
                # normal word
                word = word
            new_line += word + " "
        transcript_line = new_line.strip()
    return transcript_line

#------------------------
# Transcript processing functions for each dataset
#------------------------
def process_switchboard_transcript(audio_file, transcript_dir='../data/switchboard/transcripts'):
    """
    Processes a Switchboard transcript file.

    Args:
        audio_file (str): Name of the audio file (e.g., 'sw02001A.wav').
        transcript_dir (str, optional): Path to the root directory containing transcript subfolders. 
                                         Defaults to '../data/switchboard/transcripts'. 

    Returns:
        list: A list of tuples (start_time, end_time, text), or None if the file is not found.
    """
    filename = audio_file.split('.')[0]  # sw02001A
    speaker = filename[-1]  # A or B
    file_prefix = filename[3:-1]  # 2001
    subfolder1 = file_prefix[:2]  # 20
    subfolder2 = file_prefix  # 2001

    transcript_file = f"sw{file_prefix}{speaker}-ms98-a-trans.text"
    transcript_path = os.path.join(transcript_dir, subfolder1, subfolder2, transcript_file)

    try:
        with open(transcript_path, 'r') as f:
            transcript_lines = f.readlines()

        switchboard_pattern = r"sw\S+ (\d+\.\d+) (\d+\.\d+) (.*)"
        new_transcript_lines = []
        for line in transcript_lines:
            if not line.strip():
                continue #skip the line
            
            match = re.match(switchboard_pattern, line)  # sw.. <start_time> <end_time> <text>
            if match:
                start_time, end_time, text = float(match.group(1)), float(match.group(2)), match.group(3)
                text = retokenize_transcript_pattern(text)
                
                ##TODO: apply clean_transcript if needed
                #text = clean_transcript(text)
                new_transcript_lines.append((start_time, end_time, text))
            else:
                continue

        return new_transcript_lines

    except FileNotFoundError:
        print(f"Warning: Transcript file not found: {transcript_path}")
        return None

def process_ami_transcript(transcript_line):
    """
    Process the transcript of AMI dataset
    and return it as a tuple of (start_time, end_time, text)
    """
    # lowercase the transcript
    transcript_line = transcript_line.lower()
    
    #TODO: Clean the transcript with LLM
    # transcript_line = clean_text(transcript_line)
    # ami_pattern = r"\S+ (\d+\.\d+) (\d+\.\d+) (.*)" 
    # match = re.match(ami_pattern, transcript_line)
    # if match:
        # start_time, end_time, text = float(match.group(1)), float(match.group(2)), match.group(3)
    ami_text = retokenize_transcript_pattern(transcript_line) 
    # return start_time, end_time, text
    return ami_text
        # return transcript


