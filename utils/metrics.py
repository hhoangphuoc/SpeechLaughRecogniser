# THIS FILE CONTAINS THE LOADING AND CUSTOM METRICS
#========================================================================================================================
def calculate_iou(ref_sentence, hyp_sentence):
    """
    Calculate the Intersection Over Union (IOU) between reference and hypothesis sentence
    By this we can evaluate the similarity between the two sentences, and how accurate
    the model is in predicting the laughter words
    """
    ref_words = set(ref_sentence.split())
    hyp_words = set(hyp_sentence.split())

    intersection = ref_words.intersection(hyp_words)
    union = ref_words.union(hyp_words)

    if len(union) == 0:
        return 0
    
    similarity = len(intersection) / len(union)

    return similarity

def calculate_f1(ref_sentence, hyp_sentence):
    """
    Calculate the F1 score between reference and hypothesis segments.
    F1 score is the harmonic mean of precision and recall.
    This is used to evaluate the accuracy of the model in recognising the laughing words
    In this case, we only use it in the dataset that having [SPEECH_LAUGH] or [LAUGHTER] token
    
    Args:
        ref_segments: List of reference segments (ground truth)
        hyp_segments: List of hypothesis segments (predictions)
        
    Returns:
        float: F1 score between 0 and 1
    """
    # Convert segments to sets of words
    ref_words = set(ref_sentence.split())
    hyp_words = set(hyp_sentence.split())

    # Calculate true positives (intersection)
    true_positives = len(ref_words.intersection(hyp_words))
    
    # Handle edge case of empty sets
    if len(ref_words) == 0 and len(hyp_words) == 0:
        return 1.0
    if len(ref_words) == 0 or len(hyp_words) == 0:
        return 0.0
    
    # Calculate precision and recall
    precision = true_positives / len(hyp_words)
    recall = true_positives / len(ref_words)
    
    # Handle edge case where both precision and recall are 0
    if precision + recall == 0:
        return 0.0
        
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1

#-----------------------------------------------------------------------------------
def track_laugh_word_alignments(
        original_reference, 
        hypothesis, 
        alignment,
        dataset_type='speechlaugh' #'speechlaugh' or 'laugh'
    ):
    """
    Track the alignment status of laughing words (in uppercase) OR laughter tokens (in [LAUGH]) 
    between each pair ofreference (REF) and hypothesis (HYP).
    Uses JiWER's alignment chunks to accurately track operations.
    
    Args:
        original_reference (str): Original reference transcript with uppercase laugh words/laughter tokens
        hypothesis (str): Predicted transcript
        alignment: JiWER alignment object

    Returns:
        dict: Dictionary containing laughter statistics. This contains the following keys:
            - laugh_words: List of laughter words in the reference (speech-laugh or [LAUGH])
            - hits: List of hits (correctly predicted laughter words/tokens)
            - substitutions: List of substitutions (incorrectly predicted laughter words/tokens)
            - deletions: List of deletions (laughter words/tokens that are not predicted)
            - insertions: List of insertions (laughter words/tokens that are not in the reference)
            - hr: Hit Rate (hits / laugh_words)
            - sr: Substitution Rate (substitutions / laugh_words)
            - dr: Deletion Rate (deletions / laugh_words)
            - ir: Insertion Rate (insertions / laugh_words)

    """
    # Split reference into words while preserving case
    ref_words = original_reference.split() # THIS IS THE ORIGINAL TRANSCRIPT with UPPERCASE WORDS
    hyp_words = hypothesis.split()
    
    # Find indices of all positions of:
    # speechlaugh words in uppercase in case of `data_type = 'speechlaugh'`
    # laugh tokens in [LAUGH] in case of `datatype = 'laugh'`
    laugh_indices = {
        i: {
            'word': word,
            # 'type': 'laugh' if word == '[LAUGH]' else 'speechlaugh',
            'type': dataset_type, #'laugh' or 'speechlaugh' or 'laugh_intext'
            'lower': word.lower()
        }
        for i, word in enumerate(ref_words)
        if word.isupper() or word == '[LAUGH]'  #either speech-laugh (word.upper) or laugh (word = [LAUGH])
    }

    # LIST OF speechlaugh words and laugh tokens in REF
    # The list can be empty if there are no laughter words or tokens in the reference
    # speechlaugh_words = [info['word'] for info in laugh_indices.values() if info['type'] == 'speechlaugh'] 
    # laugh_tokens = [info['word'] for info in laugh_indices.values() if info['type'] == 'laugh']

    # `laugh_words` can be either speechlaugh words (UPPERCASE) or laugh tokens ([LAUGH])   
    laugh_words = [info['word'] for info in laugh_indices.values() if info['type'] == dataset_type]

    # total_laugh_words = len(laugh_words)
    # total_laughter_tokens = len(laughter_tokens)

    #============================================================================================
    #                   TRACKING HITS, SUBSTITUTIONS, DELETIONS, INSERTIONS
    #============================================================================================
    laugh_stats = {
        # 'speechlaugh_words': speechlaugh_words,
        # 'laugh_tokens': laugh_tokens,
        # 'total_laugh_words': total_laugh_words,
        # 'total_laughter_tokens': total_laughter_tokens,
        'laugh_words': laugh_words, #NOTE: All the speech-laugh OR [LAUGH] in REF
        'hits': [],
        'substitutions': [],
        'deletions': [],
        'insertions': [],
    }
    #----------------------------------------------------------------------------------------
    # alignment.alignments(): Jiwer WordOutput alignment that contains multiple Alignment Chunks
    # each chunk contains REF and HYP word indices and type of operation
    # Process each alignment chunk
    # current_ref_pos = 0
    # current_hyp_pos = 0
    # TYPE OF OPERATIONS IN ALIGNMENT CHUNK: `equal`, `substitute`, `insert`, or `delete`
    for chunk in alignment.alignments[0]:  # First sentence's alignment

        ref_start, ref_end = chunk.ref_start_idx, chunk.ref_end_idx
        hyp_start, hyp_end = chunk.hyp_start_idx, chunk.hyp_end_idx

        # Get the actual words from reference and hypothesis
        chunk_ref_words = ref_words[ref_start:ref_end] if ref_start < len(ref_words) else []
        chunk_hyp_words = hyp_words[hyp_start:hyp_end] if hyp_start < len(hyp_words) else []

        print(f"REF WORDS: {chunk_ref_words} - HYP WORDS: {chunk_hyp_words}\n")

        #==================================================================================
        #                           ALIGNMENT CHUNK BY TYPE
        #==================================================================================
        if chunk.type == "equal":
            # If the index of the word 
            for i, (ref_idx, hyp_idx) in enumerate(zip(range(ref_start, ref_end), 
                                                      range(hyp_start, hyp_end))):
                if ref_idx in laugh_indices:
                    laugh_stats['hits'].append({
                        'word': laugh_indices[ref_idx]['word'],
                        'hyp_word': chunk_hyp_words[i],
                        'type': laugh_indices[ref_idx]['type'],
                        'ref_pos': ref_idx,
                        'hyp_pos': hyp_idx
                    })
                    
        elif chunk.type == "substitute":
            # Check for substitutions
            for i, ref_idx in enumerate(range(ref_start, ref_end)):
                if ref_idx in laugh_indices:
                    laugh_stats['substitutions'].append({
                        'ref_word': laugh_indices[ref_idx]['word'],
                        'hyp_word': chunk_hyp_words[i] if i < len(chunk_hyp_words) else None,
                        'type': laugh_indices[ref_idx]['type'],
                        'ref_pos': ref_idx,
                        'hyp_pos': hyp_start + i if i < len(chunk_hyp_words) else None
                    })
        elif chunk.type == "delete":
            # Check for deletions
            for ref_idx in range(ref_start, ref_end):
                if ref_idx in laugh_indices:
                    laugh_stats['deletions'].append({
                        'word': laugh_indices[ref_idx]['word'],
                        'type': laugh_indices[ref_idx]['type'],
                        'ref_pos': ref_idx
                    })
        elif chunk.type == "insert":
            # Check for inserted laugh items
            for i, hyp_idx in enumerate(range(hyp_start, hyp_end)):
                hyp_word = chunk_hyp_words[i]
                # Check if HYP word is a speechlaugh or a laughter to be inserted.
                if hyp_word.isupper() or hyp_word == '[LAUGH]':
                    laugh_stats['insertions'].append({
                        'word': hyp_word,
                        'type': 'laugh' if hyp_word == '[LAUGH]' else 'speechlaugh',
                        'hyp_pos': hyp_idx
                    })
    #------------------------------------------------------------------------------------------
    
    #=====================================================================================================================================
    #             HITS, SUBSTITUTIONS, DELETIONS, INSERTIONS PER (REF - HYP) ALIGNMENT CHUNK    
    #=====================================================================================================================================      
    # total_speechlaugh_words = len(speechlaugh_words)
    # total_laugh_tokens = len(laugh_tokens)
    

    # #HITS
    laugh_stats['hr'] = len(laugh_stats['hits']) / len(laugh_stats['laugh_words'])
    
    #SUBSTITUTIONS
    laugh_stats['sr'] = len(laugh_stats['substitutions']) / len(laugh_stats['laugh_words'])

    #DELETIONS
    laugh_stats['dr'] = len(laugh_stats['deletions']) / len(laugh_stats['laugh_words'])

    #INSERTIONS
    laugh_stats['ir'] = len(laugh_stats['insertions']) / len(laugh_stats['laugh_words'])

    #-------------- LAUGH_STATS FORMAT -----------------
    """
    laugh_stats = {
        'laugh_words': laugh_words,
        # 'laughter_tokens': laughter_tokens,
        'hits': [],
        'substitutions': [],
        'deletions': [],
        'insertions': [],
        'hr': 0.0,
        'sr': 0.0,
        'dr': 0.0,
        'ir': 0.0,
    }
    """
    return laugh_stats


