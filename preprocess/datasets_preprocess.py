import os
import re
import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from huggingface_hub import login, HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

def remove_special_characters(batch):
    batch["transcript"] = re.sub(chars_to_remove_regex, '', batch["transcript"])
    return batch

def combined_dataset(
        dataset_dir, 
        combined_dataset_name="swb_all",
        save_dataset=False):
    """
    Combine all the datasets in the dataset_dir and return a single dataset. Saving to the local disk if `save_dataset` is True.
    """
    print("Combining all the datasets in the dataset_dir...")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Datasets in the directory: {os.listdir(dataset_dir)}")

    combined_dataset = concatenate_datasets([load_from_disk(os.path.join(dataset_dir, dataset_type, "switchboard_dataset")) for dataset_type in os.listdir(dataset_dir) if dataset_type != "swb"])
    print(f"Combined dataset: {combined_dataset}")
    # Shuffle the dataset
    if combined_dataset is not None:
        print("Shuffling the combined dataset...")
        combined_dataset = combined_dataset.shuffle(seed=42)

        # change [LAUGH] to <LAUGH> for every transcript
        combined_dataset = combined_dataset.map(lambda x: {'transcript': x['transcript'].replace('[LAUGH]', '<LAUGH>')}, desc="Replacing [LAUGH] with <LAUGH> in combined dataset")

        # Remove  special characters from the transcript
        combined_dataset = combined_dataset.map(remove_special_characters, desc="Removing special characters in combined dataset")
    else: 
        print("Combined dataset is None, please check the dataset directory and dataset types")
        return None

    if save_dataset:
        combined_dataset.save_to_disk(os.path.join(dataset_dir, combined_dataset_name))
        print(f"Combined dataset SAVED to {os.path.join(dataset_dir, combined_dataset_name)}")
        
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
def find_total_laughter_speechlaugh(dataset):
    """
    Find the total number of laughter, speechlaugh within the dataset
    """
    total_laugh = {
        "laughter": 0,
        "speechlaugh": 0
    }
    for example in dataset:
        for word in example['transcript'].split():
            if word == '<LAUGH>':
                total_laugh["laughter"] += 1
            elif word.isupper() and word != '<LAUGH>':
                total_laugh["speechlaugh"] += 1
            else:
                continue
    return total_laugh
#==========================================================================



#==========================================================================
#           SPLIT THE HUGGINGFACE DATASET INTO TRAIN AND TEST SET
#==========================================================================
def split_dataset(
        dataset,
        subset_ratio=1.0,
        split_ratio=0.9,
        split="both", # get both train, validation and test set,
        train_val_split=True,
        val_split_ratio=0.1, # only take a subset of the dataset
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

    # only take a subset of the dataset
    if subset_ratio < 1.0:
        print(f"Only taking {subset_ratio*100}% of the dataset")
        dataset = dataset.select(range(int(len(dataset)*subset_ratio)))

    switchboard = dataset.train_test_split(test_size=1-split_ratio)
    train_switchboard = switchboard["train"]
    test_switchboard = switchboard["test"]
    val_switchboard = None
    if train_val_split:
        train_val_switchboard = train_switchboard.train_test_split(test_size=val_split_ratio)
        train_switchboard = train_val_switchboard["train"]
        val_switchboard = train_val_switchboard["test"]

    if split == "train":
        return train_switchboard
    elif split == "test":
        return test_switchboard
    elif split == "val":
        return val_switchboard
    else:
        print(f"Split = {split}, train_val_split = {train_val_split}. Returning both train, val and test set...")
        return train_switchboard, val_switchboard, test_switchboard 

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
        repo_id = f"hhoangphuoc/{repo_name}"
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
    #============================================ COMBINE DATASETS ==========================================================
    # dataset_dir = "../datasets/switchboard"
    # print(f"Combining all the datasets in {dataset_dir}, they are:\n")
    # print(f"-{data_type}\n"for data_type in os.listdir("../datasets/switchboard") if data_type != "swb")
    # combined_dataset(dataset_dir="../datasets/switchboard", combined_dataset_name="swb_all", save_dataset=True)
    # print("Combined datasets successfully!!---------------------------------------------------")

    #============================================ LOAD DATASET AND PUSH TO HUB ============================================
    print("Loaded switchboard full dataset...")
    swb = load_from_disk("../datasets/switchboard/swb_all")
    print("Switchboard dataset:", swb) 

    # FIXME: Do we need to shuffle?

    # Split the dataset into train, validation, and test sets
    swb_train, swb_eval, swb_test = split_dataset(
        swb,
        subset_ratio=1.0,
        split_ratio=0.8,
        split="both",
        train_val_split=True,
        val_split_ratio=0.1
    )

    print("Dataset Loaded....\n")
    print(f"Train Dataset (70%): {swb_train}")
    print(f"Validation Dataset (10%): {swb_eval}")
    print(f"Test Dataset (20%): {swb_test}")
    print("------------------------------------------------------")

    # # FIND TOTAL LAUGHTER SPEECHLAUGH IN THE SPLITTED DATASET ========================================
    # total_laugh_train = find_total_laughter_speechlaugh(swb_train)
    # print("Total Laughter and Speechlaugh in Train Dataset: ", total_laugh_train)

    # total_laugh_val = find_total_laughter_speechlaugh(swb_eval)
    # print("Total Laughter and Speechlaugh in Validation Dataset: ", total_laugh_val)

    # total_laugh_test = find_total_laughter_speechlaugh(swb_test)
    # print("Total Laughter and Speechlaugh in Test Dataset: ", total_laugh_test)

    # laughter_ratio = (total_laugh_train["laughter"] + total_laugh_val["laughter"]) / total_laugh_test["laughter"]
    # speechlaugh_ratio = (total_laugh_train["speechlaugh"] + total_laugh_val["speechlaugh"]) / total_laugh_test["speechlaugh"]
    # print(f"Laughter Train/Test ratio: {laughter_ratio}")
    # print(f"Speechlaugh Train/Test ratio: {speechlaugh_ratio}")
    
    # if np.abs(laughter_ratio - speechlaugh_ratio) > 0.3:
    #     # print("The laughter and speechlaugh ratio is not balanced for Train/Test")
    #     raise ValueError("The laughter and speechlaugh ratio is not balanced for Train/Test")
    
    # print("The laughter and speechlaugh ratio is balanced for Train/Test")
    # # Save the datasets to disk
    # swb_train.save_to_disk(os.path.join("../datasets/switchboard", "swb_train"))
    # swb_eval.save_to_disk(os.path.join("../datasets/switchboard", "swb_eval"))
    # swb_test.save_to_disk(os.path.join("../datasets/switchboard", "swb_test"))
    # #================================================================================================
        
    # Push the datasets to HuggingFace Hub ========================================================
    print("Loading datasets: Train, Validation, Test...")
    swb_train = load_from_disk("../datasets/switchboard/whisper/swb_train")
    swb_eval = load_from_disk("../datasets/switchboard/whisper/swb_eval")
    swb_test = load_from_disk("../datasets/switchboard/whisper/swb_test")
    print("Pushing the datasets to Huggingface Datasets...")
    # Combine the datasets into a DatasetDict
    dataset_dict = DatasetDict({
        "train": swb_train,
        "validation": swb_eval,
        "test": swb_test
    })

    # Push the combined dataset to HuggingFace Hub
    push_dataset_to_hub(
        dataset=dataset_dict,
        repo_name="switchboard",
        private=True
    )
    print("Pushed to Huggingface Datasets successfully!!------------------------")
    
    print("----------------------------- end ------------------------------------")

