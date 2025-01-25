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
import json
from preprocess import load_word_sets
# import openai #for clean and format transcripts

#=========================================================================================================================
alignment_transformation = jiwer.Compose([
    jiwer.RemoveEmptyStrings(),
    jiwer.ToLowerCase(), #FIXME- LOWERCASE FIRST
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.SubstituteWords({
        "uhhuh": "uh-huh",
        "uh huh": "uh-huh", #for buckeye
        "mmhmm": "um-hum",
        "mmhum": "um-hum",
        "umhum": "um-hum",
        "umhmm": "um-hum",
        "yknow": "you know", #for buckeye
    }),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])

transcript_transformation = jiwer.Compose([
    jiwer.RemoveEmptyStrings(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ToLowerCase(), #FIXME - LOWERCASE IN TRANSCRIPT PROCESSING (BEFORE RETOKENIZATION)
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
])


#=========================================================================================================================
#                               SWITCHBOARD TRANSCRIPT PATTERN
#=========================================================================================================================  
switchboard_pattern = r"sw\S+ (\d+\.\d+) (\d+\.\d+) (.*)"  # sw.. <start_time> <end_time> <text>
word_sets = load_word_sets(
    "../datasets/word_sets", 
    set_names=['partial_words', 'coinages', 'anomalous_words']
    )
partialword_set = word_sets['partial_words'] # not contains every partial word :(
coinages_set = word_sets['coinages'] # not contains every {coinages} :(
anomalous_words_set = word_sets['anomalous_words'] #format [word1/word2]
#=========================================================================================================================

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
    """
    return transcript_transformation(sentence)

def transform_alignment_sentence(sentence):
    """
    Transform the alignment sentence using Jiwer Compose
    This function is used for alignment processing, i.e. the REF and HYP in the evaluation.
    Including:
    - Lowercase the line
    - Expand common English contractions (e.g., "I'm" -> "I am")
    - Remove punctuation (, . ! ?)
    - Remove multiple spaces
    - Strip the line
    - Remove empty strings
    - Substitute hesitation words to their canonical forms (e.g., "uhhuh" -> "uh-huh")
    """
    return alignment_transformation(sentence)

#=========================================================================================================================
# Retokenize the transcript line based on the given pattern
#=========================================================================================================================
def retokenize_transcript_pattern(
        transcript_line,
        transcript_type="switchboard",
        ):
    """
    Retokenize the transcript line based on the given pattern.
    General rules are:
    - Remove the line that has only "[silence]" or "[noise]"
    - Uppercase the line that only contains speechlaugh (annotated by [laughter-word] or <LAUGH-word>)
    - R
    - For the word in the line that matches filler, vocalised-noise, noise:
        - Remove the vocalized-noise (annotated by [vocalized-noise] or <vocnoise>)
        - Remove the partial word
        - Remove the pronunciation variant
        - Remove the coinages
        - Keep the normal word
        - Replace the speech_laugh with the corresponding token:
            - WORD if retokenize_type="speechlaugh", otherwise skipping it
        - Replace the laughter with the corresponding token:
            - [LAUGH] if retokenize_type="laugh", otherwise skipping it
        - Remove trailing hyphens (if any)
    Args:
        transcript_line (str): A original line of transcript text.
        transcript_type (str): The type of the transcript, either "switchboard" or "buckeye".

        #Expected inputs:
        - switchboard: 

    Returns:
        text (str): A retokenized line of transcript text, processed based on the given rules
    """

    new_line = ""
    if transcript_type == "switchboard":
        #====================================================================
        #  SPECIAL ANNOTATED WORD PATTERNS (IN SWITCHBOARD)
        #====================================================================
        noise_pattern = r"\[noise\]|\[vocalized-noise\]" # pattern: [noise], [vocalized-noise] (switchboard)
        asides_pattern = r"<\w+_aside>" # pattern: <b_aside> or <e_aside> (switchboard)

        pronunciation_variant_pattern = r"(\w+)_1" # pattern: word_1 (switchboard)
        coinages_pattern = r'\{([^}]+)\}'
        partialword_pattern = r'\b\w+\[[^\]]+\]-'

        laughter_pattern = r"\[laughter\]" #pattern: [laughter] (switchboard)
        speech_laugh_pattern = r"\[laughter-([\w'\[\]-]+)\]"
        #====================================================================
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
            #==================================================================   
            # Finally, remove trailing hyphens (if any)
            if word.endswith('-'):
                word = word[:-1]
            new_line += word + " "
            # transcript_line = new_line.strip()
    elif transcript_type == "buckeye":
        """
        Retokenize the Buckeye transcript line based on the given pattern.
        """
        #====================================================================
        #  SPECIAL ANNOTATED WORD PATTERNS (IN BUCKEYE)
        #====================================================================
        # Noise pattern: include silences, noises, interviewer background noise, cutoffs, and errors in produced words
        # Since the transcript is lowercased, the pattern is in lowercase format
        noise_pattern = r"\<sil\>|\<noise\>|\<vocnoise\>|\<iver\>|\<cutoff\>|\<error\>|\<unknown\>" #TOBE remove

        # Partial words which is cut off or error in lexical or phonological error
        # These words should be REMOVED
        #---------------------------------------------------------------------------------------------
        # cutoff_pattern = r"\<cutoff-(\w+)\>|\<error-(\w+)\>"
        # #hesitation words: including words being extending, repeating or hesitating during speech.
        # hesitation_pattern = r"\<ext-(\w+)\>|\<hes-(\w+)\>" #KEEP the word
        #---------------------------------------------------------------------------------------------

        #partial and vocalised word in Buckeye (except speechlaugh) are words speaking along with the noises above,
        # however still contain speech content -> KEEP
        # annotated as: <NOISE-word>, <IVER-word>, <VOCNOISE-word>, <EXTEND-word>,...
        vocalise_word_pattern = r"\<noise-(\w+)\>|\<vocnoise-(\w+)\>|\<iver-(\w+)\>" #get the word from it


        laugh_pattern = r"\<laugh\>"
        speech_laugh_pattern = r"\<laugh-(\w+)\>"
        #====================================================================
        for word in transcript_line.split(" "):
            print("original word: ", word)
            #============ PATTERN MATCHING AND ADJUST TRANSCRIPT============
            if re.match(noise_pattern, word):
                # remove if <SIL>, <NOISE>, <VOCNOISE>, <IVER>, <CUTOFF>, <ERROR>
                word = " " #removing the noise word
            elif word.startswith("<cutoff-") or word.startswith("<error-") or word.startswith("<unknown-") or word.startswith("<exclude-"):
                word = " " #removing the cutoff word - <cutoff-clipping=word>, <error-error=word>
            elif word.startswith("<noise-") or word.startswith("<vocnoise-") or word.startswith("<iver-"):
                word = word[:-1].split('-')[1] # KEEP the word
            elif word.startswith("<ext-") or word.startswith("<hes-"):
                word = word[:-1].split('-')[1] # KEEP the word
            else:
                word = word
            
            #===RETOKENIZE SPEECH_LAUGH AND LAUGHTER============================
            if word.startswith("<laugh-"):
                laughed_word = word[:-1].split('-')[1]
                if len(laughed_word.split("_")) > 1:
                    # contruct the word as a sentence of all words
                    laughed_word = " ".join(laughed_word.split("_"))
                word = laughed_word.upper()

            elif re.match(laugh_pattern, word):
                word = "<LAUGH>"
            print("processed word: ", word)
            #==================================================================
            new_line += word + " "

    # FINALLY extent the english, remove multiple spaces and strip
    transcript_line = jiwer.Compose([
        jiwer.RemoveEmptyStrings(), #remove empty strings
        jiwer.ExpandCommonEnglishContractions(), #'ll -> will, 're -> are, etc.
        jiwer.RemovePunctuation(), #remove punctuation
        jiwer.RemoveWhiteSpace(replace_by_space=True), #remove multiple spaces
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
    This function is designed to specifically process the Switchboard clipped dataset, with the specific annotations rules for 
    speech events in the Switchboard corpus.
    This function parsing the original transcript files from the `transcript_dir` and shortly processed it with Jiwer Compose, before
    retokenizing the transcript line based on the specific patterns, using `retokenize_transcript_pattern()` with the `transcript_type="switchboard"`.

    The transcript file is expected to be in the format:
    `<file_id> <start_time> <end_time> text`
    The transcript was processed in following steps:
    1. Read the transcript file. (specified by `sw{file_prefix}{speaker}-ms98-a-trans.text`)
    2. Find the sentence pattern in the transcript file: 
        - extract the `<start_time>`, `<end_time>`, and text.
        - if the sentence only contains `[silence], [noise] or [vocalised-noise]`, skip
    3. Preprocess the transcript lines with Jiwer Compose, including: 
        - remove multiple spaces
        - lowercase the line
        - remove empty strings
        - strip the line
    
    For the retokenization process using `retokenize_transcript_pattern()`, 
    the transcript line was processed based on the specific patterns, particularly:
        - Remove the partial word
        - Remove the word that only contains [noise] or [vocalized-noise]
        - Replace the speech_laugh with WORD or laughter with [LAUGH]
        - Remove the pronunciation variant
        - Remove the coinages
        - Keep the normal word

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
                        jiwer.ToLowerCase(), #lowercase every words to normalise before retokenization
                        jiwer.Strip(), #strip the line
                    ])(text)

# 4. Retokenize the sentence based on the specific patterns=========================
# while retokenizing, we uppercase special tokens like [LAUGH] and WORD
# we do this for every transcript sentence, regardless of any sub-datasets
                    retokenize_text = retokenize_transcript_pattern(
                        text,
                        transcript_type="switchboard"
                    ) # return the retokenized text (either removed or replaced)
                    new_transcript_lines.append((start_time, end_time, retokenize_text))
            return new_transcript_lines
    except FileNotFoundError:
        print(f"Warning: Transcript file not found: {transcript_path}")
        return None
    
#=========================================================================================================================

def process_buckeye_transcript(
        audio_file, # THIS DATA IS ALREADY CLIPPED AND ONLY CONTAINS 1 TRANSCRIPT LINE FOR EACH AUDIO
        transcript_dir='/deepstore/datasets/hmi/speechlaugh-corpus/buckeye_data/buckeye_refs_wavs/transcripts',
    ):
    """
    Processes a Buckeye transcript file.
    This function is designed to specifically process the Buckeye clipped dataset, with specific annotations pattern
    of Buckeye corpus. 
    This was done in retokenization process, using: `retokenize_transcript_pattern()`
    with the `transcript_type="buckeye"`.
    
    Args:
        audio_file (str): Name of the audio file in Buckeye data path (e.g., 's0201a_1.wav').
        transcript_dir (str, optional): Path to the root directory containing transcript subfolders of Buckeye Corpus. 
                                         Defaults to '../data/buckeye/transcripts'.
    
    Returns:
        list: `new_transcript_lines` A list of retokenized transcript lines, or None if the file is not found.

    """

    filename = audio_file.split('.')[0]  # s0201a_1

    transcript_file = f"{filename}.txt" # s0201a_1.txt
    transcript_path = os.path.join(transcript_dir, transcript_file)

    print(f"Processing transcript: {transcript_path}")
    try:
        with open(transcript_path, 'r') as f:
            # transcript_lines = f.readlines()
            text = f.readline()
            # new_transcript_lines = []
            # for text in transcript_lines:
            if not text.strip():
                return #skip the empty line
            # line is each transcript sentence to the corresponding audio segment

            # Remove only silence, only noise, and vocalized-noise, or interviewer background noise
            if re.match(r"\<SIL\>|\<NOISE\>|\<VOCNOISE\>|\<IVER\>", text):
                return #skip the line

            text = jiwer.Compose([
                    jiwer.RemoveMultipleSpaces(), #remove multiple spaces in the sentence
                    jiwer.Strip(), #strip the line to remove trailing
                    jiwer.ToLowerCase(), #lowercase the line
            ])(text)
            # Retokenize the sentence based on the specific patterns 
            retokenize_text = retokenize_transcript_pattern(
                text,
                transcript_type="buckeye"
            )
            return retokenize_text
        
    except FileNotFoundError:
        print(f"Warning: Transcript file not found: {transcript_path}")
        return None
