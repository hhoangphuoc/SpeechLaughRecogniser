# This file using to process the transcript to get the word-level timestamps for Whisper fine-tuning.
# The file includes:
# 1. Merging multiple transcripts into a single transcript.
# 2. Retokenization of a given transcript
# 3. Extract the word-level timestamps
# 4. matching timestamps with the audio segments

import re
import os
import jiwer
# import openai #for clean and format transcripts

#------- SWITCHBOARD TRANSCRIPT PATTERN -----------------------------------------------------------------------------#
switchboard_pattern = r"sw\S+ (\d+\.\d+) (\d+\.\d+) (.*)"  # sw.. <start_time> <end_time> <text>
#-------------------------------------------------------------------------------------------------------------------#

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
transcript_processing_composer = jiwer.Compose([
    jiwer.RemoveSpecificWords(['[silence]', '[noise]', '[vocalized-noise]', '<b_aside>', '<e_aside>']),
    jiwer.RemovePunctuation(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemoveEmptyStrings(),
    jiwer.ToLowerCase(),
    # after to lower case, uppercase the word [laughter]
    jiwer.SubstituteWords({
        '[laughter]': '[LAUGHTER]',
    })
])

#--------------------------------
# Clean and format the transcript sentence with Jiwer Compose
#--------------------------------
def clean_transcript_sentence(sentence):
    """
    Clean and format the transcript text using Jiwer Componse
    """
    return transcript_processing_composer(sentence)


#--------------------------------
# Retokenize the transcript line based on the given pattern
#--------------------------------

def retokenize_transcript_pattern(transcript_line):
    """
    Retokenize the transcript line based on the given pattern.
    The rules are:
    - Remove the line that has only "[silence]" or "[noise]"
    - Uppercase the line that only contains vocalsound
    - For the word in the line that matches filler, speech_laugh, noise:

        - Remove the partial word
        - Replace the speech_laugh with the corresponding token
        - Remove the pronunciation variant
        - Remove the coinages
        - Keep the normal word

    Args:
        transcript_line (str): A original line of transcript text.

    Returns:
        text (str): A retokenized line of transcript text, processed based on the given rules
    """
    #initially, lowercase the entire line
    transcript_line = transcript_line.lower()

    new_line = ""
    for word in transcript_line.split():
        if re.match(partialword_pattern, word):
            # if the word is a partial word, remove the partial word
            continue
        elif re.match(speech_laugh_pattern, word):
            # if the word is [laughter-...], change it to the token [SPEECH_LAUGH]
            word = "[SPEECH_LAUGH]"
        elif re.match(pronunciation_variant_pattern, word):
            word = re.sub(r"_1", "", word)
        elif re.match(coinages_pattern, word):
            word = re.sub(r"{|}", "", word)    
        else:
            # normal word
            word = word
        
        new_line += word + " "
        transcript_line = new_line.strip()
    return transcript_line

#------------------------
# Transcript processing functions for each dataset
#------------------------
def process_switchboard_transcript(
        audio_file, 
        transcript_dir='/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data/transcripts'
        ):
    """
    Processes a Switchboard transcript file.
    The transcript file is expected to be in the format:
    sw02001A 0.000 0.500 sentence
    The transcript was processed in following steps:
    1. Read the transcript file. (specified by sw{file_prefix}{speaker}-ms98-a-trans.text)
    2. Find the sentence pattern in the transcript file: extract the start_time, end_time, and text.
    3. Preprocess the transcript lines with Jiwer Compose, including: 
        - remove [silence], [noise], [vocalized-noise], <b_aside>, <e_aside>
        - remove puctuation (, . ! ?)
        - expand common English contractions (e.g., "I'm" -> "I am")
        - remove multiple spaces
        - strip the line
        - remove empty strings
        - lowercase the line
        - substitute [laughter] with [LAUGHTER]
    4. Process each line of the transcript sentence and process more specific patterns, including:
        - Remove the partial word
        - Replace the speech_laugh with [SPEECH_LAUGH]
        - Remove the pronunciation variant
        - Remove the coinages
        - Keep the normal word
        This function calling the retokenize_transcript_pattern().

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

    print(f"Processing transcript: {transcript_path}")
    #FIXME: REMOVE ONCE THE TRANSCRIPTS ARE MATCH PERFECTLY ----------
    transcript_lines_debug = os.path.abspath("../debug/switchboard_transcripts_debug.txt")
    #-----------------------------------------------------------------
    try:
        with open(transcript_path, 'r') as f:
# 1. Read the transcript file
            transcript_lines = f.readlines()
            new_transcript_lines = []
            # with open(transcript_lines_debug, 'a') as f2:
            #     f2.write(f"Transcript file: {transcript_path}\n")
            #     f2.write(f"Transcript lines:\n")
            #     count_line = 0
            for line in transcript_lines:
                if not line.strip():
                    continue #skip the empty line

# 2. Match the sentence pattern in the transcript file and extract the start_time, end_time, and text

                match = re.match(switchboard_pattern, line)  # sw.. <start_time> <end_time> <text>
                if match:
                    start_time, end_time, text = float(match.group(1)), float(match.group(2)), match.group(3)

# 3. processing the sentence by Jiwer Compose 
                    # text = text.strip()
                    # if text == "[silence]" or text == "[noise]" or text == "[vocalized-noise]":
                    #     continue #ignore the line and not call this function
                    # else: 

                    text = clean_transcript_sentence(text)

# 4. Retokenize the sentence based on the specific patterns

                    retokenize_text = retokenize_transcript_pattern(text) # return the retokenized text (either removed or replaced)
                    # f2.write(f"Retokenized Text: {retokenize_text}\n")
                    # f2.write(f"({start_time},{end_time},{retokenize_text})\n")
                    new_transcript_lines.append((start_time, end_time, retokenize_text))
            
            # f2.write("\n\n")
            return new_transcript_lines
    except FileNotFoundError:
        print(f"Warning: Transcript file not found: {transcript_path}")
        return None

#--------------------------------
# Process the AMI dataset transcript
#--------------------------------
def process_ami_transcript(transcript_line):
    """
    Process the transcript of AMI dataset
    and return it as a tuple of (start_time, end_time, text)
    """
    # lowercase the transcript
    transcript_line = transcript_line.lower()
    ami_text = retokenize_transcript_pattern(transcript_line) 
    return ami_text



