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
silence_pattern = r"\b\[silence\]\b" #pattern: [silence]
noise_pattern = r"\[noise\]|\[vocalized-noise\]"
asides_pattern = r"<\w+_aside>" # pattern: <b_aside> or <e_aside>

pronunciation_variant_pattern = r"(\w+)_1" # pattern: word_1
# coinages_pattern = r"{(\w+)}" # pattern: {word}
coinages_pattern = r'\{(\w+(?:-\w+)*)\}' # pattern: {brother-in-laws}

# partialword_pattern = r"-w+\[\w+['\w+\-]\]\w+|\w+\[\w+['\w+\-]\]w+-"  # partial word: w[ord]- or -[wor]d, or -sh[ouldn't], or [shouldn't]d-
# partialword_pattern = r"-w+\[\w+['\w-]+\]w+|\w+\[\w+['\w-]+\]w+-"
# partialword_pattern = r"-\w+\[\w+['\w-]+\]\w+|\b\w+\[\w+['\w-]+\]-"
partialword_pattern = r"\b\w*(?:\[\w+'?\w*\])?-|-\[\w+'?\w*\]\w*\b"

laughter_pattern = r"\[laughter\]" #pattern: [laughter]
speech_laugh_pattern = r"\[laughter-([\w'\[\]-]+)\]"



# TOBE CONSIDERED FOR RETOKENIZATION ------------------------
#filler_pattern = r"\b(uh|um|mm|uh[ -]huh|ah|oh|hmm+)\b"
# vocalsound_pattern = r"\b([laughter]|[cough]|[sigh]|[sniff]|[throatclearing]|[sneeze])\b"

#-----------------------------------------------------------------------------------------------#
# SUBTITUTE THE PARTIAL WORDS WITH REGEX PATTERNS
transcript_processing_composer = jiwer.Compose([
    # jiwer.RemovePunctuation(), # FIXME: NOT USING THIS TO REMOVE PUNCTUATION BECAUSE IT ALSO REMOVE - AND ' WHICH ARE IMPORTANT FOR RETOKENIZATION
    jiwer.ExpandCommonEnglishContractions(),
    # jiwer.SubstituteRegexes(substitution_patterns),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemoveEmptyStrings(),
    jiwer.ToLowerCase(),
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

def retokenize_transcript_pattern(
        transcript_line,
        tokenize_speechlaugh = False,
        ):
    """
    Retokenize the transcript line based on the given pattern.
    The rules are:
    - Remove the line that has only "[silence]" or "[noise]"
    - Uppercase the line that only contains vocalsound
    - For the word in the line that matches filler, speech_laugh, noise:

        - Remove the partial word
        - Replace the speech_laugh with the corresponding token:
            - [SPEECH_LAUGH] if tokenize_speechlaugh is True, otherwise the laughing word
        - Remove the pronunciation variant
        - Remove the coinages
        - Keep the normal word

    Args:
        transcript_line (str): A original line of transcript text.

    Returns:
        text (str): A retokenized line of transcript text, processed based on the given rules
    """

    new_line = ""
    for word in transcript_line.split():
        if re.match(noise_pattern, word):
            # remove if [noise], [vocalized-noise]
            continue
        elif match := re.match(speech_laugh_pattern, word):
            # if the word is [laughter-...], change it to the token [SPEECH_LAUGH]
            word = "[SPEECH_LAUGH]" if tokenize_speechlaugh else match.group(1).upper()
        elif re.match(laughter_pattern, word):
            # if the word is [laughter], change it to the token [LAUGHTER]
            word = "[LAUGHTER]"
        elif match := re.match(coinages_pattern, word):
            replace_pattern = match.group(1).replace("-", " ")
            word = re.sub(coinages_pattern,replace_pattern, word) # {brother-in-laws} -> brother in laws
        elif re.match(pronunciation_variant_pattern, word):
            word = re.sub(r"_1", "", word)
        elif re.match(asides_pattern, word):
            word = re.sub(asides_pattern, "", word) # remove the <b_aside>, <e_aside>
        else:
            word = word # normal word
        
        # in the end, remove if it is a partial word
        word = re.sub(partialword_pattern, "", word)
        
        new_line += word + " "
        # transcript_line = new_line.strip()
    # finally extent the english, remove multiple spaces and strip
    transcript_line = jiwer.Compose([
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])(new_line)

    return transcript_line

#------------------------
# Transcript processing functions for each dataset
#------------------------
def process_switchboard_transcript(
        audio_file, 
        transcript_dir='/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data/transcripts',
        tokenize_speechlaugh = False
    ):
    """
    Processes a Switchboard transcript file.
    The transcript file is expected to be in the format:
    sw02001A 0.000 0.500 sentence
    The transcript was processed in following steps:
    1. Read the transcript file. (specified by sw{file_prefix}{speaker}-ms98-a-trans.text)
    2. Find the sentence pattern in the transcript file: 
        - extract the start_time, end_time, and text.
        - if the sentence only contains [silence] or [noise], skip
    3. Preprocess the transcript lines with Jiwer Compose, including: 
        - remove puctuation (, . ! ?)
        - expand common English contractions (e.g., "I'm" -> "I am")
        - remove multiple spaces
        - strip the line
        - remove empty strings
        - lowercase the line
    4. Process each line of the transcript sentence and process more specific patterns, particularly,
    each word in the sentence:
        - Remove the partial word
        - Remove the word that only contains [noise] or [vocalized-noise]
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
    try:
        with open(transcript_path, 'r') as f:

# 1. Read the transcript file
            transcript_lines = f.readlines()
            new_transcript_lines = []
            for line in transcript_lines:
                if not line.strip():
                    continue #skip the empty line

# 2. Match the sentence pattern in the transcript file and extract the start_time, end_time, and text

                match = re.match(switchboard_pattern, line)  # sw.. <start_time> <end_time> <text>
                if match:
                    start_time, end_time, text = float(match.group(1)), float(match.group(2)), match.group(3)

# 3. Processing the sentence by Jiwer Compose 
                    text = text.strip()
                    if text == "[silence]" or text == "[noise]" or text=="[vocalized-noise]": 
                        # if the sentence only contains [silence] or [noise], skip
                        continue 

                    text = clean_transcript_sentence(text)

# 4. Retokenize the sentence based on the specific patterns

                    retokenize_text = retokenize_transcript_pattern(
                        text,
                        tokenize_speechlaugh=tokenize_speechlaugh
                    ) # return the retokenized text (either removed or replaced)
                    new_transcript_lines.append((start_time, end_time, retokenize_text))
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



