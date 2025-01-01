import os
from datasets import DatasetDict, concatenate_datasets
from datasets import load_from_disk
from huggingface_hub import login, HfApi, create_repo


def combined_dataset(dataset_dir, save_dataset=False):
    """
    Combine all the datasets in the dataset_dir and return a single dataset
    """
    print("Combining all the datasets in the dataset_dir...")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Datasets in the directory: {os.listdir(dataset_dir)}")

    combined_dataset = concatenate_datasets([load_from_disk(os.path.join(dataset_dir, dataset_type, "switchboard_dataset")) for dataset_type in os.listdir(dataset_dir)])
    
    # if lowercase_speechlaugh:
    #     print("Lowercasing speechlaugh...")
    #     combined_dataset = combined_dataset.map(
    #         lambda x: {"transcript": x["transcript"].lower()},
    #         remove_columns=combined_dataset.column_names,
    #         load_from_cache_file=True,
    #         desc="Lowercasing speechlaugh",
    #     )
    print(f"Combined dataset: {combined_dataset}")
    if save_dataset:
        combined_dataset.save_to_disk(os.path.join(dataset_dir, "swb_full"))
        print(f"Combined dataset SAVED to {os.path.join(dataset_dir, 'swb_full')}")
    return combined_dataset

#==========================================================================
#                           FILTER AND MATCH DATASETS
#==========================================================================
def filter_laughter_dataset(
        dataset
        # intext=False
    ):
    """
    Filter dataset that only have sentences that contain laughter special token
    which is [LAUGH]. If `intext` is True, filter out the sentences that only contain [LAUGH], the sentence would include both [LAUGH] and other words.
    """
    print("FILTERING LAUGH-ONLY DATASET...")
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
    print("FILTERING SPEECHLAUGH DATASET...")
    speech_laugh_filter = lambda x: any(word.isupper() for word in x['transcript'].split())
    swb_speechlaugh_dataset = dataset.filter(speech_laugh_filter)
    print(f"Speech-laugh dataset: {swb_speechlaugh_dataset}")
    print(swb_speechlaugh_dataset[0])
    return swb_speechlaugh_dataset
#===================================================================================

def filter_speech_dataset(dataset):
    """
    Filter dataset that only have sentences that do not contain speechlaugh or laughter special tokens
    """
    print("FILTERING SPEECH-ONLY DATASET...")
    speech_filter = lambda x: not any(word.isupper() or word=='[LAUGH]' for word in x['transcript'].split())
    swb_speech_dataset = dataset.filter(speech_filter)
    print(f"Speech dataset: {swb_speech_dataset}")
    print(swb_speech_dataset[0])
    return swb_speech_dataset

#==========================================================================
#           SPLIT THE HUGGINGFACE DATASET INTO TRAIN AND TEST SET
#==========================================================================
def split_dataset(
        dataset,
        subset_ratio=1.0, # only take a subset of the dataset
        split_ratio=0.9,
        split="both", # get both train, validation and test set,
        train_val_split=True,
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
    val_switchboard = None
    if train_val_split:
        train_val_switchboard = train_switchboard.train_test_split(test_size=0.1, shuffle=True)
        train_switchboard = train_val_switchboard["train"]
        val_switchboard = train_val_switchboard["test"]


    if split == "train":
        return train_switchboard
    elif split == "test":
        return test_switchboard
    elif split == "val":
        return val_switchboard
    else:
        # if do_val_split:
        #     print(f"Split = {split}, do_val_split = {do_val_split}. Returning train, validation and test set...")
        #     return train_switchboard, val_switchboard, test_switchboard
        # else:
        print(f"Split = {split}, train_val_split = {train_val_split}. Returning both train, val and test set...")
        return train_switchboard, val_switchboard, test_switchboard 

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

    # processed_dataset_path = 
    print("Loaded switchboard full dataset...")
    switchboard_full = load_from_disk("../datasets/switchboard/swb_full")
    print("Switchboard dataset:", switchboard_full)
    
    # total_laughter = sum(1 for x in switchboard_full if '[LAUGH]' in x['transcript'])
    # total_speechlaugh = sum(1 for word in switchboard_full if any(word.isupper() for word in x['transcript'].split()))
    # total_speech = sum(1 for x in switchboard_full if not any(word.isupper() or word =='[LAUGH]' for word in x['transcript'].split()) )
    total_laughter= 0
    total_speechlaugh= 0
    total_speech= 0
    for example in switchboard_full:
        for word in example['transcript'].split():
            if word == '[LAUGH]':
                total_laughter += 1
            elif word.isupper() and word != '[LAUGH]':
                total_speechlaugh += 1
            else:
                total_speech += 1
    
    print(f"Total laughter-intext: {total_laughter}")
    print(f"Total speechlaugh: {total_speechlaugh}")
    print(f"Total words: {total_speech}")
    print("Complete!!---------------------------------------------------")

    #push to hub
    # push_dataset_to_hub(
    #     dataset=switchboard_full, 
    #     repo_name="switchboard", # name will be hhoangphuoc/switchboard_swb_full
    #     private=True
    # )

