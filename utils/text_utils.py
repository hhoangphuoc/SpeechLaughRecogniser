
import jiwer
alignment_transformation = jiwer.Compose([
    jiwer.ExpandCommonEnglishContractions(),
    # jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    # jiwer.ToLowerCase(),
    jiwer.RemoveEmptyStrings(),
])

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
    This function mainly used for REF and Hypothesis in the evaluation.
    Including:
    - Expand common English contractions (e.g., "I'm" -> "I am")
    - Remove multiple spaces
    - Strip the line
    - Remove empty strings
    """
    return alignment_transformation(sentence)