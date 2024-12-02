import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from pathlib import Path
from tqdm import tqdm
from utils.dictionary_utils import clean_laughter_word, clean_alternate_word, clean_anomalous_word, clean_coinage_word

def parse_dictionary_file(file_path):
    # Initialize sets for different word categories
    partial_words = set()
    laughing_words = set()
    alternate_pronunciations = set()
    hesitation_sounds = set()
    proper_nouns = set()
    anomalous_words = set()
    coinages = set()
    
    current_section = None
    
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Processing dictionary to store in sets"):
            # Skip empty lines and comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Check for section markers
            if line == "partial word pronunciations":
                current_section = "partial"
                continue
            elif line == "words containing laughter":
                current_section = "laughter"
                continue
            elif line == "common alternate pronunciations of words":
                current_section = "alternate"
                continue
            elif line == "hesitation sounds":
                current_section = "hesitation"
                continue
            elif line == "proper nouns":
                current_section = "proper"
                continue
            elif line == "anomalous words":
                current_section = "anomalous"
                continue
            elif line == "coinages":
                current_section = "coinages"
                continue
            elif line.startswith('----------------'):
                continue
                
            # Extract the first word from the line (before the pronunciation)
            word = line.split()[0]
            
            # Add word to appropriate set based on current section

            #=======================================================
            #           PARTIAL WORDS
            #=======================================================    
            if current_section == "partial":
                partial_words.add(word)

            #=======================================================
            #           LAUGHTER WORDS
            #=======================================================    
            elif current_section == "laughter":
                laughing_words.add(word)
                # if line.startswith('[laughter-'):
                #     # Extract word between 'laughter-' and ']'
                #     word = line[len('[laughter-'):line.find(']')]
                #     # Clean and reconstruct the word
                #     cleaned_word = clean_laughter_word(word)
                #     if cleaned_word:  # Only add if not empty
                #         laughing_words.add(cleaned_word)
                continue

            #=======================================================
            #           ALTERNATE PRONUNCIATIONS
            #=======================================================
            elif current_section == "alternate":
                # Case 1: [laughter-word_1] format
                if word.startswith('[laughter-'):
                    base_word = word[len('[laughter-'):word.find(']')]
                    cleaned_word = clean_alternate_word(base_word)
                    if cleaned_word:
                        laughing_words.add(cleaned_word)
                
                # Case 3: regular alternate pronunciation (word_1)
                else:
                    cleaned_word = clean_alternate_word(word)
                    if cleaned_word:
                        alternate_pronunciations.add(cleaned_word)

            #=======================================================
            #           HESITATION SOUNDS
            #=======================================================    
            elif current_section == "hesitation":
                hesitation_sounds.add(word)


            #=======================================================
            #           PROPER NOUNS
            #=======================================================
            elif current_section == "proper":
                proper_nouns.add(word)


            #=======================================================
            #           ANOMALOUS WORDS
            #=======================================================    
            elif current_section == "anomalous":
                anomalous_words.add(word)
                # # Handle anomalous word format: [wrong/correct] pronunciation
                # if line.startswith('[') and '/' in line:
                #     # Extract the part between [ and ]
                #     bracket_content = line[line.find('[')+1:line.find(']')]
                #     # Split by / and get the second part (correct word)
                #     parts = bracket_content.split('/')
                #     if len(parts) >= 2:
                #         correct_word = parts[1].strip()
                #         # Clean the word before adding to set
                #         cleaned_word = clean_anomalous_word(correct_word)
                #         if cleaned_word:  # Only add if not empty
                #             anomalous_words.add(cleaned_word)
            
            #=======================================================
            #           COINAGES
            #=======================================================    
            elif current_section == "coinages":
                coinages.add(word)
    print(f"Number of partial words: {len(partial_words)}")
    print(f"Number of laughing words: {len(laughing_words)}")
    print(f"Number of alternate pronunciations: {len(alternate_pronunciations)}")
    print(f"Number of hesitation sounds: {len(hesitation_sounds)}")
    print(f"Number of proper nouns: {len(proper_nouns)}")
    print(f"Number of anomalous words: {len(anomalous_words)}")
    print(f"Number of coinages: {len(coinages)}")

    print("Finished processing dictionary!")
    return {
        'partial_words': partial_words,
        'laughing_words': laughing_words,
        'alternate_pronunciations': alternate_pronunciations,
        'hesitation_sounds': hesitation_sounds,
        'proper_nouns': proper_nouns,
        'anomalous_words': anomalous_words,
        'coinages': coinages
    }


def save_word_sets(word_sets, output_dir):
    """
    Save word sets to pickle files in the specified directory.
    
    Args:
        word_sets (dict): Dictionary containing different word sets
        output_dir (str): Directory path to save the pickle files
    """
    # Create directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save each set to a separate file
    for set_name, word_set in word_sets.items():
        file_path = os.path.join(output_dir, f"{set_name}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(word_set, f)

def load_word_sets(input_dir,
                   set_names=None
                   ):
    """
    Load word sets from pickle files.
    
    Args:
        input_dir (str): Directory containing the pickle files
        set_names (list, optional): List of set names to load. If None, all sets are loaded.
    Returns:
        dict: Dictionary containing all word sets, each set is a set()
    """
    word_sets = {}
    
    # Expected set names
    if set_names is None:
        set_names = [
            'partial_words',
            'laughing_words',
            'alternate_pronunciations',
            'hesitation_sounds',
            'proper_nouns',
            'anomalous_words',
            'coinages'
        ]
    
    # Load each set
    for set_name in set_names:
        file_path = os.path.join(input_dir, f"{set_name}.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                word_sets[set_name] = pickle.load(f)
        else:
            word_sets[set_name] = set()  # Initialize empty set if file doesn't exist
            
    return word_sets

# Usage example:
if __name__ == "__main__":
    """
    Number of partial words: 6396
    Number of laughing words: 2133
    Number of alternate pronunciations: 0
    Number of hesitation sounds: 0
    Number of proper nouns: 7427
    Number of anomalous words: 344
    Number of coinages: 22647
    """
    output_dir = "../datasets/word_sets"
    #=======================EXAMPLE FOR SAVING=============================
    
    # dictionary_path = "../datasets/sw-ms98-dict.text"
    # word_sets = parse_dictionary_file(dictionary_path)
    # save_word_sets(word_sets, output_dir)
    #============================================================================   
    
    # EXAMPLE USAGE FOR LOADING==============================================:
    # set_names = ['partial_words']
    # loaded_word_sets = load_word_sets(output_dir, set_names=set_names)
    # print(f"Partial words: {loaded_word_sets['partial_words']}")
    # print(f"Laughing words: {loaded_word_sets['laughing_words']}")
    # print(f"Coinages: total: {len(loaded_word_sets['coinages'])}")

    #============================================================================
