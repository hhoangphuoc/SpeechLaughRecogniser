
#====================================================================================================
#           CLEANING FUNCTIONS FOR SPECIFIC ANNOTATION TYPES
#====================================================================================================
def clean_anomalous_word(word):
    """Remove punctuation and convert to lowercase for consistent comparison"""
    return ''.join(c for c in word.lower() if c.isalnum())

def clean_laughter_word(word):
    """
    Clean and reconstruct original word from laughter section format.
    Examples:
    - "gue[ss]-" -> "guess"
    - "ha[ve]-" -> "have"
    - "ha[s]-" -> "has"
    """
    # Remove any trailing hyphen
    word = word.rstrip('-')
    if '{' in word and '}' in word:
        return word[word.find('{')+1:word.find('}')]
    
    # If word contains brackets, reconstruct the full word
    if '[' in word and ']' in word:
        before_bracket = word[:word.find('[')]
        in_bracket = word[word.find('[')+1:word.find(']')]
        # Reconstruct the full word
        return before_bracket + in_bracket
    
    return word

def clean_alternate_word(word):
    """Remove _1, _2, etc. from alternate pronunciation words
    and store it in corresponding set.
    Examples:
    - "about_1" -> "about"
    - "becau[se_1]" -> "because"
    """
    return word.split('_')[0]

def clean_coinage_word(word):
    """
    Remove {} from coinage words. And return only the word inside.
    Examples:
    - "{recycling}" -> "recycling"
    """
    return word[1:-1] # {} from the beginning and end

#====================================================================================================
#           CHECKING WORDS FUNCTIONS
#====================================================================================================
def is_partial_word(word, word_sets):
    """Check if word is in the partial words set"""
    return word in word_sets['partial_words']
    

def is_laughter_word(word, word_sets):
    return word in word_sets['words_with_laughter']

def is_alternate_pronunciation(word, word_sets):
    return word in word_sets['alternate_pronunciations']

def is_hesitation_sound(word, word_sets):
    return word in word_sets['hesitation_sounds']

def is_proper_noun(word, word_sets):
    return word in word_sets['proper_nouns']

def is_anomalous_word(word, word_sets):
    cleaned_word = clean_anomalous_word(word)
    return cleaned_word in word_sets['anomalous_words']

def is_coinage(word, word_sets):
    cleaned_word = clean_coinage_word(word)
    return cleaned_word in word_sets['coinages']