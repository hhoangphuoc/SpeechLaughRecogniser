# This file using to process the transcript to get the word-level timestamps for Whisper fine-tuning.
# The file includes:
# 1. Merging multiple transcripts into a single transcript.
# 2. Retokenization of a given transcript
# 3. Extract the word-level timestamps
# 4. matching timestamps with the audio segments

import re
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jiwer
from preprocess import load_word_sets
# import openai #for clean and format transcripts

#=========================================================================================================================
#                   SWITCHBOARD TRANSCRIPT PATTERN
#=========================================================================================================================  
switchboard_pattern = r"sw\S+ (\d+\.\d+) (\d+\.\d+) (.*)"  # sw.. <start_time> <end_time> <text>


#=========================================================================================================================
#               SETS OF SPECIAL PATTERNS FROM ORIGINAL SWITCHBOARD TRANSCRIPT THAT NEED TO BE HANDLED
#=========================================================================================================================
word_pattern = r"\b\w+\b"
silence_pattern = r"\b\[silence\]\b" #pattern: [silence]
noise_pattern = r"\[noise\]|\[vocalized-noise\]"
asides_pattern = r"<\w+_aside>" # pattern: <b_aside> or <e_aside>

pronunciation_variant_pattern = r"(\w+)_1" # pattern: word_1
coinages_pattern = r'\{([^}]+)\}'

partialword_pattern = r'\b\w+\[[^\]]+\]-'

laughter_pattern = r"\[laughter\]" #pattern: [laughter]
speech_laugh_pattern = r"\[laughter-([\w'\[\]-]+)\]"

# ===========================TOBE CONSIDERED FOR RETOKENIZATION =============================================== 
#filler_pattern = r"\b(uh|um|mm|uh[ -]huh|ah|oh|hmm+)\b"
# vocalsound_pattern = r"\b([laughter]|[cough]|[sigh]|[sniff]|[throatclearing]|[sneeze])\b"
#=========================================================================================================================  

#=========================================================================================================================

alignment_transformation = jiwer.Compose([
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ToLowerCase(), #FIXME- NOT LOWERCASE IN ALIGNMENT
    jiwer.RemoveEmptyStrings(),
    jiwer.SubstituteWords({
        "uhhuh": "uh-huh",
        "mmhmm": "um-hum",
        "mmhum": "um-hum",
        "umhum": "um-hum",
        "umhmm": "um-hum",
    })
])

transcript_transformation = jiwer.Compose([
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ToLowerCase(), #FIXME - LOWERCASE IN TRANSCRIPT PROCESSING (BEFORE RETOKENIZATION)
    jiwer.RemoveEmptyStrings(),
])

word_sets = load_word_sets(
    "../datasets/word_sets", 
    set_names=['partial_words', 'coinages', 'anomalous_words']
    )
partialword_set = word_sets['partial_words'] # not contains every partial word :(
coinages_set = word_sets['coinages'] # not contains every {coinages} :(
anomalous_words_set = word_sets['anomalous_words'] #format [word1/word2]

def transform_number_words(text, reverse=False):
    """
    Transform number words between spelled out form and digit pairs.
    If reverse=False (default): Convert number words to digit pairs
    If reverse=True: Convert digit pairs back to number words
    
    Examples with reverse=False:
    - nineteen -> one nine
    - twenty -> two zero
    
    Examples with reverse=True:  
    - one nine -> nineteen
    - two zero -> twenty
    """
    
    # Dictionary for number word mappings
    number_mappings = {
        'eleven': 'one one',
        'twelve': 'one two', 
        'thirteen': 'one three',
        'fourteen': 'one four',
        'fifteen': 'one five',
        'sixteen': 'one six',
        'seventeen': 'one seven',
        'twenty': 'two zero',
        'thirty': 'three zero',
        'forty': 'four zero',
        'fifty': 'five zero',
        'sixty': 'six zero',
        'seventy': 'seven zero',
        'eighty': 'eight zero',
        'ninety': 'nine zero'
    }

    # Create reverse mapping for converting back
    reverse_mappings = {v: k for k, v in number_mappings.items()}
    
    words = text.split()
    transformed_words = []
    
    if not reverse:
        # Forward transformation (number words to digit pairs)
        for word in words:
            if word in number_mappings:
                transformed_words.append(number_mappings[word])
            else:
                transformed_words.append(word)
    else:
        # Reverse transformation (digit pairs to number words)
        i = 0
        while i < len(words):
            if i < len(words) - 1:
                word_pair = words[i] + ' ' + words[i+1]
                if word_pair in reverse_mappings:
                    transformed_words.append(reverse_mappings[word_pair])
                    i += 2
                    continue
            transformed_words.append(words[i])
            i += 1
            
    return ' '.join(transformed_words)
#--------------------------------------------------------

#=========================================================================================================================
# Clean and format the transcript sentence with Jiwer Compose
#=========================================================================================================================
def clean_transcript_sentence(sentence):
    """
    Clean and format the transcript text using Jiwer Compose
    This function mainly used for the transcript processing before retokenization.
    Including:
    - Expand common English contractions (e.g., "I'm" -> "I am")
    - Remove punctuation (, . ! ?)
    - Remove multiple spaces in the sentence
    - Strip the line
    - Remove empty strings
    - Lowercase the line
    - Substitute hesitation words to their canonical forms (e.g., "uhhuh" -> "uh-huh")
    """
    return transcript_transformation(sentence)

def transform_alignment_sentence(sentence):
    """
    Transform the alignment sentence using Jiwer Compose
    This function is used for alignment processing, i.e. the REF and HYP in the evaluation.
    Including:
    - Expand common English contractions (e.g., "I'm" -> "I am")
    - Remove punctuation (, . ! ?)
    - Remove multiple spaces
    - Strip the line
    - Remove empty strings
    """
    return alignment_transformation(sentence)

#=========================================================================================================================
# Retokenize the transcript line based on the given pattern
#=========================================================================================================================
def retokenize_transcript_pattern(
        transcript_line,
        ):
    """
    Retokenize the transcript line based on the given pattern.
    The rules are:
    - Remove the line that has only "[silence]" or "[noise]"
    - Uppercase the line that only contains vocalsound
    - For the word in the line that matches filler, speech_laugh, noise:

        - Remove the partial word
        - Remove the pronunciation variant
        - Remove the coinages
        - Keep the normal word
        - Replace the speech_laugh with the corresponding token:
            - WORD if retokenize_type="speechlaugh", otherwise skipping it
        - Replace the laughter with the corresponding token:
            - [LAUGH] if retokenize_type="laugh", otherwise skipping it
    Args:
        transcript_line (str): A original line of transcript text.

    Returns:
        text (str): A retokenized line of transcript text, processed based on the given rules
    """

    new_line = ""
    for word in transcript_line.split():
        #============ PATTERN MATCHING AND ADJUST TRANSCRIPT============
        if re.match(noise_pattern, word):
            # remove if [noise], [vocalized-noise]
            continue
        elif match := re.match(partialword_pattern, word) or word in partialword_set:
            word = "" #removing the partial word
        elif match := re.match(coinages_pattern, word):
            # remove surrounding curly braces
            word = re.sub(coinages_pattern, r"\1", word)
        elif word in anomalous_words_set:
            # anomalous words are [word1/word2]
            word = re.sub(r"\[|\]", "", word)
            word = word.split('/')[1]
        elif re.match(pronunciation_variant_pattern, word):
            word = re.sub(r"_1", "", word)
        elif re.match(asides_pattern, word):
            word = re.sub(asides_pattern, "", word) # remove the <b_aside>, <e_aside>
        else:
            word = word # normal word
        #=============================== RETOKENIZE SPEECH_LAUGH AND LAUGHTER=================================
        if match := re.match(speech_laugh_pattern, word):
            # if retokenize_type == "speechlaugh":
                # check if the speech-laugh is a form of partial word
            laughed_word = match.group(1)
            if re.match(partialword_pattern, laughed_word): 
                word = "" #removing the partial word
            elif re.match(coinages_pattern, laughed_word):
                laughed_word = re.sub(coinages_pattern, r"\1", laughed_word)
            
            word = laughed_word.upper() #Uppercase the laughing word if retokenize_type=speechlaugh
            # else:
            #     word = "" #otherwise removing it
        elif re.match(laughter_pattern, word):
            word = "[LAUGH]"
            # if retokenize_type == "laugh":
            #     word = "[LAUGH]" #change it to the token [LAUGH] if retokenize_type=laugh
            # else:
            #     word = "" #otherwise removing it
        #==================================================================   
        # Finally, remove trailing hyphens (if any)
        if word.endswith('-'):
            word = word[:-1]

        new_line += word + " "
        # transcript_line = new_line.strip()
    # FINALLY extent the english, remove multiple spaces and strip
    transcript_line = jiwer.Compose([
        jiwer.ExpandCommonEnglishContractions(), #'ll -> will, 're -> are, etc.
        jiwer.RemoveMultipleSpaces(), #remove multiple spaces
        jiwer.Strip(), #strip the line
    ])(new_line)

    return transcript_line

#=========================================================================================================================
#                           Transcript processing functions for each dataset
#=========================================================================================================================

#--------------------------------------
# Process the Switchboard transcript
#-------------------------------------- 
def process_switchboard_transcript(
        audio_file, 
        transcript_dir='/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data/transcripts',
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
        - Replace the speech_laugh with WORD or laughter with [LAUGH]
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
                    text = jiwer.Compose([
                        jiwer.RemoveMultipleSpaces(), #remove multiple spaces
                        jiwer.ToLowerCase(), #lowercase every words
                    ])(text)

# 4. Retokenize the sentence based on the specific patterns=========================
# while retokenizing, we uppercase special tokens like [LAUGH] and WORD
# we do this for every transcript sentence, regardless of any sub-datasets
                    retokenize_text = retokenize_transcript_pattern(
                        text,
                        # retokenize_type=retokenize_type
                    ) # return the retokenized text (either removed or replaced)
                    new_transcript_lines.append((start_time, end_time, retokenize_text))
            return new_transcript_lines
    except FileNotFoundError:
        print(f"Warning: Transcript file not found: {transcript_path}")
        return None