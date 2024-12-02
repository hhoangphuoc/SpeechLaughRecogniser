from datasets import DatasetDict, concatenate_datasets
from datasets import load_from_disk
from huggingface_hub import login, HfApi, create_repo
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
    laugh_filter = lambda x: '[SPEECH_LAUGH]' in x['transcript'] or '[LAUGH]' in x['transcript']
    laugh_dataset = source_token_dataset.filter(laugh_filter)
    
    # Extract filenames from laugh dataset audio paths
    laugh_filenames = set()

    for audio_data in laugh_dataset:
        laugh_filenames.add(audio_data['audio']['path'])
    
    # Filter other dataset based on matching filenames
    filename_filter = lambda x: x['audio']['path'] in laugh_filenames
    laughing_words_dataset = target_word_dataset.filter(filename_filter)
    
    return laugh_dataset, laughing_words_dataset

def filter_laughter_dataset(
        dataset
        # intext=False
    ):
    """
    Filter dataset that only have sentences that contain laughter special token
    which is [LAUGH]. If `intext` is True, filter out the sentences that only contain [LAUGH], the sentence would include both [LAUGH] and other words.
    """
    print("Get only laughter dataset...")
    
    # if intext:
    #     print(f"`intext = {intext}`. Filter sentences that contain both [LAUGH] and other words")
    #     laughter_filter = lambda x: '[LAUGH]' in x['transcript'] and not x['transcript'].replace('[LAUGH]', '').strip()
    #     # laughter_filter = lambda x: ('[LAUGH]' in x['transcript']) and (x['transcript'].strip() != '[LAUGH]')
    # else:
    #     print(f"`intext = {intext}`. Filter all laughter sentences")
    laughter_filter = lambda x: '[LAUGH]' in x['transcript'] and (x['transcript'].strip() != '[LAUGH]')
    swb_laugh_dataset = dataset.filter(laughter_filter)
    print(f"Laughter dataset: {swb_laugh_dataset}")
    print(swb_laugh_dataset[0])
    return swb_laugh_dataset

#===================================================================================
def filter_speech_laugh_dataset(dataset):
    """
    Filter dataset that only have sentences that contain speechlaugh special token
    which is word in uppercase. For example: YOU, WORD, KNOW, WHAT, etc.
    """
    print("Get only speechlaugh dataset...")
    speech_laugh_filter = lambda x: any(word.isupper() for word in x['transcript'].split())
    swb_speechlaugh_dataset = dataset.filter(speech_laugh_filter)
    print(f"Speech-laugh dataset: {swb_speechlaugh_dataset}")
    # print the first 10 rows
    print(swb_speechlaugh_dataset.select(range(10)))
    return swb_speechlaugh_dataset
#===================================================================================

def filter_speech_dataset(dataset):
    """
    Filter dataset that only have sentences that do not contain speechlaugh or laughter special tokens
    """
    print("Get only speech dataset...")
    speech_filter = lambda x: not any(word.isupper() or word=='[LAUGH]' for word in x['transcript'].split())
    swb_speech_dataset = dataset.filter(speech_filter)
    print(f"Speech dataset: {swb_speech_dataset}")
    # print the first 10 rows
    print(swb_speech_dataset.select(range(10)))
    return swb_speech_dataset

#==========================================================================
#           SPLIT THE HUGGINGFACE DATASET INTO TRAIN AND TEST SET
#==========================================================================
def split_dataset(
        dataset,
        subset_ratio=1.0, # only take a subset of the dataset
        split_ratio=0.9,
        split="both", # get both train and test set,
        do_val_split=True # Whether to split the train set into train and validation set
    ):
    """
    Split the dataset into train and validation set
    Args:
    - dataset: HuggingFace Dataset object
    - split_ratio: ratio of the train set
    - split: "train", "test", "val" or "both" to return both train, validation and test set
    Return:
    - train_dataset: HuggingFace Dataset object
    - test_dataset: HuggingFace Dataset object
    """
    switchboard = DatasetDict()
    
    # shuffle the dataset
    dataset = dataset.shuffle(seed=42)

    # only take a subset of the dataset
    if subset_ratio < 1.0:
        print(f"Only taking {subset_ratio*100}% of the dataset")
        dataset = dataset.select(range(int(len(dataset)*subset_ratio)))

    switchboard = dataset.train_test_split(test_size=1-split_ratio, shuffle=True)
    train_switchboard = switchboard["train"]
    test_switchboard = switchboard["test"]

    if do_val_split:
        # split the train set into train and validation set
        train_switchboard, val_switchboard = train_switchboard.train_test_split(test_size=0.1, shuffle=True)

    if split == "train":
        return train_switchboard
    elif split == "test":
        return test_switchboard
    else:
        if do_val_split:
            print(f"Split = {split}, do_val_split = {do_val_split}. Returning train, validation and test set...")
            return train_switchboard, val_switchboard, test_switchboard
        else:
            print(f"Split = {split}. Returning both train and test set...")
            return train_switchboard, test_switchboard 

#==========================================================================
#==========================================================================


def push_dataset_to_hub(dataset, repo_name, token=None, private=True):
    """
    Push the dataset to the HuggingFace Hub with proper authentication and configuration.
    
    Args:
        dataset: HuggingFace Dataset object to push
        repo_name: Name of the repository (format: 'username/dataset-name')
        token: HuggingFace API token. If None, will use the token from huggingface-cli login
        private: Whether to create a private repository (default: True)
    
    Returns:
        None
    """
    try:
        # from huggingface_hub import HfApi, create_repo
        
        # Step 1: Initialize the Hugging Face API
        api = HfApi(token=token)
        
        # Step 2: Create the repository if it doesn't exist
        repo_id = f"hhoangphuoc/switchboard_{repo_name}"
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                token=token
            )
            print(f"Created new dataset repository: {repo_id}")
        except Exception as e:
            print(f"Repository {repo_id} already exists or error occurred: {e}")
        
        # Step 3: Push the dataset to the hub with progress bar
        dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=token,
            embed_external_files=True,  # Upload audio files if present
            max_shard_size="500MB"  # Split into smaller files if dataset is large
        )
        
        print(f"Successfully pushed dataset to {repo_id}")
        print(f"View your dataset at: https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        print(f"Error pushing dataset to hub: {e}")
        raise e

#=====================================================================================================
if __name__ == "__main__":
    # login()

    processed_dataset_path = "../datasets/switchboard/swb_laugh_intext/switchboard_dataset"

    laugh_dataset = load_from_disk(processed_dataset_path)

    #push to hub
    push_dataset_to_hub(
        dataset=laugh_dataset, 
        repo_name="laugh-intext", # name will be hhoangphuoc/switchboard_laugh-intext
        private=True
    )

