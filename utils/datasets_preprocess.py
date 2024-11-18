#==========================================================================
#                           FILTER AND MATCH DATASETS
#==========================================================================
def filter_and_match_datasets(source_token_dataset, target_word_dataset):
    """
    Filter dataset for laughs words and match with another dataset based on audio filenames.
    The intention of this function is to get the subset of all the rows contains 
    [SPEECH_LAUGH] and [LAUGHTER] words in the transcript column 
    and match with another dataset which instead the [SPEECH_LAUGH] is annotated as a laughing word

    This match is ensure we can extract the sub-dataset with only the laughing words
    and using it for evaluation and alignment by WER
    
    Args:
        source_token_dataset: HuggingFace dataset containing transcript column with [SPEECH_LAUGH] token
        target_word_dataset: Dataset to filter based on matching audio paths
        
    Returns:
        tuple: (laugh_dataset, laughing_words_dataset) in which:
        - laugh_dataset: HuggingFace dataset containing transcript column with [SPEECH_LAUGH] token
        - laughing_words_dataset: HuggingFace dataset containing transcript with laughing words
    """
    # Filter rows containing laugh markers
    laugh_filter = lambda x: '[SPEECH_LAUGH]' in x['transcript'] or '[LAUGHTER]' in x['transcript']
    laugh_dataset = source_token_dataset.filter(laugh_filter)
    
    # Extract filenames from laugh dataset audio paths
    laugh_filenames = set()

    for audio_data in laugh_dataset:
        laugh_filenames.add(audio_data['audio']['path'])
    
    # Filter other dataset based on matching filenames
    filename_filter = lambda x: x['audio']['path'] in laugh_filenames
    laughing_words_dataset = target_word_dataset.filter(filename_filter)
    
    return laugh_dataset, laughing_words_dataset

def filter_laughter_words(dataset):
    """
    Filter dataset for laughing words
    """
    laughter_filter = lambda x: '[LAUGHTER]' in x['transcript']
    laughter_dataset = dataset.filter(laughter_filter)
    return laughter_dataset

def filter_speech_laugh_words(dataset):
    """
    Filter dataset for speech laugh words
    """
    speech_laugh_filter = lambda x: '[SPEECH_LAUGH]' in x['transcript']
    speech_laugh_dataset = dataset.filter(speech_laugh_filter)
    return speech_laugh_dataset