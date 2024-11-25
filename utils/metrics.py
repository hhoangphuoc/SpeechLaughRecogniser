# THIS FILE CONTAINS THE LOADING AND CUSTOM METRICS
#========================================================================================================================
import evaluate


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
        alignment):
    """
    Track the alignment status of laughing words (in uppercase) and laughter tokens between reference and hypothesis.
    Uses JiWER's alignment chunks to accurately track operations.
    
    Args:
        original_reference (str): Original reference transcript with uppercase laugh words/laughter tokens
        hypothesis (str): Predicted transcript
        alignment: JiWER alignment object

    Returns:
        dict: Dictionary containing laughter statistics. This contains the following keys:
            - laugh_words: List of laughter words in the reference
            - laughter_tokens: List of laughter tokens in the reference
            - total_laugh_words: Total number of laughter words in the reference
            - total_laughter_tokens: Total number of laughter tokens in the reference
            - hits: List of hits (correctly predicted laughter words/tokens)
            - substitutions: List of substitutions (incorrectly predicted laughter words/tokens)
            - deletions: List of deletions (laughter words/tokens that are not predicted)
            - insertions: List of insertions (laughter words/tokens that are not in the reference)

    """
    # Split reference into words while preserving case
    ref_words = original_reference.split() # THIS IS THE ORIGINAL TRANSCRIPT with UPPERCASE WORDS
    hyp_words = hypothesis.split()
    
    # Find indices of laugh words and laughter tokens
    laugh_indices = {
        i: {
            'word': word,
            'type': 'token' if word in ['[LAUGHTER]', '[SPEECH_LAUGH]'] else 'word',
            'lower': word.lower()
        }
        for i, word in enumerate(ref_words) # GET: word (both LAUGHTER and LAUGHING WORDS) and corresponding indices
        if word.isupper() or word in ['[LAUGHTER]', '[SPEECH_LAUGH]']
    }

    # LAUGH WORDS AND LAUGHTER TOKENS IN REF
    laugh_words = [info['word'] for info in laugh_indices.values() if info['type'] == 'word'] 
    laughter_tokens = [info['word'] for info in laugh_indices.values() if info['type'] == 'token']

    total_laugh_words = len(laugh_words)
    total_laughter_tokens = len(laughter_tokens)

    #============================================================================================
    #                   TRACKING HITS, SUBSTITUTIONS, DELETIONS, INSERTIONS
    #============================================================================================
    laugh_stats = {
        'laugh_words': laugh_words,
        'laughter_tokens': laughter_tokens,
        'total_laugh_words': total_laugh_words,
        'total_laughter_tokens': total_laughter_tokens,
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
    total_hits, total_substitutions, total_deletions, total_insertions = 0, 0, 0, 0

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
            # Check for exact matches
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
                total_hits += 1
                    
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
                    total_substitutions += 1
        elif chunk.type == "delete":
            # Check for deletions
            for ref_idx in range(ref_start, ref_end):
                if ref_idx in laugh_indices:
                    laugh_stats['deletions'].append({
                        'word': laugh_indices[ref_idx]['word'],
                        'type': laugh_indices[ref_idx]['type'],
                        'ref_pos': ref_idx
                    })
                    total_deletions += 1
        elif chunk.type == "insert":
            # Check for inserted laugh items
            for i, hyp_idx in enumerate(range(hyp_start, hyp_end)):
                hyp_word = chunk_hyp_words[i]
                if hyp_word.isupper() or hyp_word in ['[LAUGHTER]', '[SPEECH_LAUGH]']:
                    laugh_stats['insertions'].append({
                        'word': hyp_word,
                        'type': 'token' if hyp_word in ['[LAUGHTER]', '[SPEECH_LAUGH]'] else 'word',
                        'hyp_pos': hyp_idx
                    })
                    total_insertions += 1
    #------------------------------------------------------------------------------------------
    
    #=====================================================================================================================================
    #                                                       CALCULATE STATISTICS
    #=====================================================================================================================================      

    #HITS----------------------------------------------------------------------
    laugh_word_hits = sum(1 for hit in laugh_stats['hits'] if hit['type'] == 'word')
    laugh_stats['lwhr'] = laugh_word_hits / total_laugh_words if total_laugh_words > 0 else 0

    laugh_token_hits = sum(1 for hit in laugh_stats['hits'] if hit['type'] == 'token')
    laugh_stats['lthr'] = laugh_token_hits / total_laughter_tokens if total_laughter_tokens > 0 else 0

    # SUBSTITUTIONS----------------------------------------------------------------------
    laugh_word_substitutions = sum(1 for substitution in laugh_stats['substitutions'] if substitution['type'] == 'word')
    laugh_stats['lwsr'] = laugh_word_substitutions / total_laugh_words if total_laugh_words > 0 else 0

    laugh_token_substitutions = sum(1 for substitution in laugh_stats['substitutions'] if substitution['type'] == 'token')
    laugh_stats['ltsr'] = laugh_token_substitutions / total_laughter_tokens if total_laughter_tokens > 0 else 0

    # DELETIONS----------------------------------------------------------------------
    laugh_word_deletions = sum(1 for deletion in laugh_stats['deletions'] if deletion['type'] == 'word')
    laugh_stats['lwdr'] = laugh_word_deletions / total_laugh_words if total_laugh_words > 0 else 0
    
    laugh_token_deletions = sum(1 for deletion in laugh_stats['deletions'] if deletion['type'] == 'token')
    laugh_stats['ltdr'] = laugh_token_deletions / total_laughter_tokens if total_laughter_tokens > 0 else 0
    
    # INSERTIONS----------------------------------------------------------------------
    laugh_word_insertions = sum(1 for insertion in laugh_stats['insertions'] if insertion['type'] == 'word')
    laugh_stats['lwir'] = laugh_word_insertions / total_laugh_words if total_laugh_words > 0 else 0
    
    laugh_token_insertions = sum(1 for insertion in laugh_stats['insertions'] if insertion['type'] == 'token')
    laugh_stats['ltir'] = laugh_token_insertions / total_laughter_tokens if total_laughter_tokens > 0 else 0
    

    #-------------- LAUGH_STATS FORMAT -----------------
    """
    laugh_stats = {
        'laugh_words': laugh_words,
        'laughter_tokens': laughter_tokens,
        'total_laugh_words': total_laugh_words,
        'total_laughter_tokens': total_laughter_tokens,
        'hits': [],
        'substitutions': [],
        'deletions': [],
        'insertions': [],
        'lwhr': 
        'lthr': 
        'lwsr': 
        'ltsr': 
        'lwdr': 
        'ltdr': 
        'lwir': 
        'ltir': 
    }
    """
    return laugh_stats


#=================================================================================================      
def load_metrics():
    """
    Load the metrics for evaluation using HuggingFace's evaluate library including:
    - Word Error Rate (WER)
    - F1 score (F1)
    """
    wer_metric = evaluate.load("wer") #Word Error Rate between the hypothesis and the reference transcript
    f1_metric = evaluate.load("f1") #F1 score between the hypothesis and the reference transcript

    return wer_metric, f1_metric
