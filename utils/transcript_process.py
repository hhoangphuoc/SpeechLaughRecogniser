# This file using to process the transcript to get the word-level timestamps for Whisper fine-tuning.
# The file includes:
# 1. Merging multiple transcripts into a single transcript.
# 2. Retokenization of a given transcript
# 3. Extract the word-level timestamps
# 4. matching timestamps with the audio segments

import re
import os
import openai
#---------------THESE FUNCTIONS ARE USED FOR MERGING TRANSCRIPTS---------------#
def merge_transcripts(transcript_a, transcript_b):
    """
    Merges two transcripts from the same conversation, ensuring proper 
    timeline ordering and handling silence annotations.

    Args:
        transcript_a (str): Transcript of speaker A.
        transcript_b (str): Transcript of speaker B.

    Returns:
        str: The merged transcript with each line in the format:
             "Turn number Speaker (start_timestamp, stop_timestamp) annotated_token\n"
    """

    lines_a = transcript_a.splitlines()
    print("Lines of transcript A: ", lines_a)
    lines_b = transcript_b.splitlines()
    print("Lines of transcript B: ", lines_b)

    merged_transcript = []
    i, j = 0, 0  # Indices to iterate through lines_a and lines_b
    turn_number = 0

    while i < len(lines_a) or j < len(lines_b):
        if i < len(lines_a) and j < len(lines_b):
            _, start_a, end_a, token_a = lines_a[i].split()
            _, start_b, end_b, token_b = lines_b[j].split()
            start_a, end_a, start_b, end_b = map(float, [start_a, end_a, start_b, end_b])

            # Check if both speakers are talking in the same turn
            if abs(start_a - start_b) < 0.01: 
                merged_transcript.append(f"{turn_number} A ({start_a:.2f}, {end_a:.2f}) {token_a}\n")
                merged_transcript.append(f"{turn_number} B ({start_b:.2f}, {end_b:.2f}) {token_b}\n")
                i += 1
                j += 1
                turn_number += 1
            elif start_a < start_b:
                # A starts talking before B
                merged_transcript.append(f"{turn_number} A ({start_a:.2f}, {end_a:.2f}) {token_a}\n")
                if token_b == "[silence]":
                    # Adjust B's silence annotation if needed
                    lines_b[j] = f"{lines_b[j].split()[0]} {end_a:.2f} {end_b} {token_b}" 
                i += 1
            else:
                # B starts talking before A
                merged_transcript.append(f"{turn_number} B ({start_b:.2f}, {end_b:.2f}) {token_b}\n")
                if token_a == "[silence]":
                    # Adjust A's silence annotation if needed
                    lines_a[i] = f"{lines_a[i].split()[0]} {end_b:.2f} {end_a} {token_a}"
                j += 1

        elif i < len(lines_a):
            # Only A is left
            _, start_a, end_a, token_a = lines_a[i].split()
            start_a, end_a = float(start_a), float(end_a)
            merged_transcript.append(f"{turn_number} A ({start_a:.2f}, {end_a:.2f}) {token_a}\n")
            i += 1
            turn_number += 1
        else:
            # Only B is left
            _, start_b, end_b, token_b = lines_b[j].split()
            start_b, end_b = float(start_b), float(end_b)
            merged_transcript.append(f"{turn_number} B ({start_b:.2f}, {end_b:.2f}) {token_b}\n")
            j += 1
            turn_number += 1
    
    return "".join(merged_transcript)
def merge_transcripts_from_files(
    file_name_a, 
    file_name_b,
    base_dir = "../data/switchboard/transcripts/20/2001"):
    """
    Merges two transcripts from files, ensuring proper timeline ordering and handling 
    silence annotations.

    Args:
        file_path_a (str): Path to the transcript file of speaker A.
        file_path_b (str): Path to the transcript file of speaker B.

    Returns:
        str: The merged transcript with each line in the format:
             "Turn number Speaker (start_timestamp, stop_timestamp) annotated_token\n"
    """
    
    file_path_a = os.path.join(base_dir, file_name_a)
    file_path_b = os.path.join(base_dir, file_name_b)
    merged_file_name = os.path.basename(file_name_a).split('-')[0]
    output_file_path = os.path.join(base_dir, f"{merged_file_name}-merged.txt")
    
    if os.path.exists(output_file_path):
        print(f"Merged transcript already exists at {output_file_path}")
        return output_file_path
    else:

        print(f"Merging transcripts from {file_path_a} and {file_path_b}")

        with open(file_path_a, 'r') as f_a, open(file_path_b, 'r') as f_b:
            transcript_a = f_a.read()
            transcript_b = f_b.read()

        with open(output_file_path, 'w') as f_out:
            f_out.write(merge_transcripts(transcript_a, transcript_b))

    return output_file_path
#--------------------------------------NOT USED--------------------------------------#


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

    # Define regular expressions to match different token types
    word_pattern = r"\b\w+\b"
    pause_pattern = r"\[silence\]"
    noise_pattern = r"\[noise\]"
    filler_pattern = r"\b(uh|um|uh[ -]huh|ah|hmm+)\b"
    speech_laugh_pattern = r"\[laughter-(\w+)\]"

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