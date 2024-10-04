import pandas as pd
from datasets import Dataset, Audio, DatasetDict
from datasets import ClassLabel, Features, Sequence, Value
from huggingface_hub import login
from dotenv import load_dotenv
import os
import argparse

load_dotenv()

"""
This script ONLY contains the function to process the dataset from the csv file
AND uploaded to HuggingFace Dataset Hub
"""

def process_dataset(csv_input_path):
    """
    Load the dataset from the csv file and convert to HuggingFace Dataset object
    Args:
    - csv_input_path: path to the csv file (train.csv, eval.csv)
    Return:
    - train_dataset: HuggingFace Dataset object
    """

    train_df = pd.read_csv(csv_input_path)

    train_df["sampling_rate"] = train_df["sampling_rate"].apply(lambda x: int(x))

    # train_df = train_df[train_df["transcript"].apply(lambda x: len(x) > 0)]

    #shuffle the dataframe
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    train_dataset = Dataset.from_pandas(train_df)
    
    #Resample the audio_array column if it not 16kHz
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))

    return train_dataset

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input_path", type=str, default="./datasets/train_switchboard.csv",help="Path to the train csv file")
    parser.add_argument("--eval_input_path", type=str, default="./datasets/val_switchboard.csv",help="Path to the eval csv file")

    args = parser.parse_args()
    train_dataset = process_dataset(args.train_input_path)
    print(train_dataset)
    eval_dataset = process_dataset(args.eval_input_path)
    print(eval_dataset)

    combined_dataset = DatasetDict({
        "train": train_dataset,
        "validation": eval_dataset
    })
    

    # Upload the dataset to HuggingFace Dataset Hub
    login(
        token=os.getenv("HUGGINGFACE_TOKEN"),
        add_to_git_credential=False
        )
    combined_dataset.push_to_hub(
        "hhoangphuoc/switchboard",
    )
    print("Dataset uploaded successfully!")
