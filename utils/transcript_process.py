# This file using to process the transcript to get the word-level timestamps for Whisper fine-tuning.
# The file includes:
# 1. Merging multiple transcripts into a single transcript.
# 2. Retokenization of a given transcript
# 3. Extract the word-level timestamps
# 4. matching timestamps with the audio segments

import re
import os
# import openai #for clean and format transcripts

#------- SWITCHBOARD TRANSCRIPT PATTERN -----------------------------------------------------------------------------#
switchboard_pattern = r"sw\S+ (\d+\.\d+) (\d+\.\d+) (.*)"  # sw.. <start_time> <end_time> <text>
#------------------------------------------------



# SETS OF SPECIAL PATTERNS FROM ORIGINAL SWITCHBOARD TRANSCRIPT THAT NEED TO BE HANDLED ---------------------------------#
word_pattern = r"\b\w+\b"
pause_pattern = r"\b\[silence\]\b" #pattern: [silence]
noise_pattern = r"\[noise\]|\[vocalized-noise\]"
laughter_pattern = r"\[laughter\]" #pattern: [laughter]
pronunciation_variant_pattern = r"(\w+)_1"
asides_pattern = r"<b_aside|e_aside>" # pattern: <b_aside> or <e_aside>
coinages_pattern = r"{(\w+)}" # pattern: {word}
# speech_laugh_pattern = r"\[laughter-(\w+)\]"
speech_laugh_pattern = r"\[laughter-([\w'\[\]-]+)\]"

# partialword_pattern = r"-\b\w+\]|\b\w+-|\[\b\w+\]|\[\b\w+-|\b\w+\[\w+\]-"  # partial word: [word]- or w[ord]- or -[wor]d 
# partialword_pattern = r"-\[\w+\]\w+|\w+\[\w+\]-"  # partial word: w[ord]- or -[wor]d
partialword_pattern = r"-\[\w+['\w+]\]\w+|\w+\[\w+['\w+]\]-"  # partial word: w[ord]- or -[wor]d, or sh[ouldn't]- or -[shouldn't]d

# TOBE CONSIDERED FOR RETOKENIZATION ------------------------
#filler_pattern = r"\b(uh|um|mm|uh[ -]huh|ah|oh|hmm+)\b"
# vocalsound_pattern = r"\b([laughter]|[cough]|[sigh]|[sniff]|[throatclearing]|[sneeze])\b"

#-----------------------------------------------------------------------------------------------#



#-----------------------------------------------------------------------------------------------#
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

    if transcript_line.strip() == "[silence]" or transcript_line.strip() == "[noise]" or transcript_line.strip() == "[vocalize-noise]":
        return #ignore the line and not call this function
    else:
        # transform the word in the line when it matches the pattern
        new_line = ""
        for word in transcript_line.split():
            if re.match(speech_laugh_pattern, word):
                # print("Matched speech_laugh pattern:", word)
                # if the word is [laughter-...], change it to the token [SPEECH_LAUGH]
                word = "[SPEECH_LAUGH]"
            elif re.match(laughter_pattern, word):
                # word = match.group(0)
                # print("Matched laughter pattern:", word)
                word = word.upper() #[LAUGHTER]
            if re.match(partialword_pattern, word):
                # if the word is a partial word, remove the partial word
                continue
            elif re.match(pronunciation_variant_pattern, word):
                word = re.sub(r"_1", "", word)
            elif re.match(coinages_pattern, word):
                word = re.sub(r"{|}", "", word)    
            elif re.match(noise_pattern, word) or re.match(pause_pattern, word) or re.match(asides_pattern, word):
                continue  
            else:
                # normal word
                word = word
            
            new_line += word + " "
        transcript_line = new_line.strip()
        
        # print("Processed Transcript line:", transcript_line)
    return transcript_line

#------------------------
# Transcript processing functions for each dataset
#------------------------
def process_switchboard_transcript(
        audio_file, 
        transcript_dir='/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data/transcripts'):
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
    ami_text = retokenize_transcript_pattern(transcript_line) 
    return ami_text



