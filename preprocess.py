import librosa
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
import os
import argparse
import torch

from datasets import load_dataset, Dataset, DatasetDict, Audio

from utils.transcript_process import process_switchboard_transcript, process_ami_transcript
from utils.audio_process import cut_audio_based_on_transcript_segments

import utils.params as prs


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#--------------------------------------------------
# FILTER AND MATCH DATASETS
#--------------------------------------------------
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
    laugh_filter = lambda x: '[SPEECH_LAUGH]' in x['transcript'] #or '[LAUGHTER]' in x['transcript']
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


#--------------------------------------------------
# SPLIT THE HUGGINGFACE DATASET INTO TRAIN AND TEST SET
#--------------------------------------------------
def split_dataset(
        dataset,
        subset_ratio=1.0, # only take a subset of the dataset
        split_ratio=0.9,
        split="both" # get both train and test set
    ):
    """
    Split the dataset into train and validation set
    Args:
    - dataset: HuggingFace Dataset object
    - split_ratio: ratio of the train set
    - split: "train" or "test" or "both" to return both train and test set
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

    if split == "train":
        return train_switchboard
    elif split == "test":
        return test_switchboard
    else:
        print(f"Split = {split}. Returning both train and test set...")
        return train_switchboard, test_switchboard 

#--------------------------------------------------
# PROCESS A CSV FILE TO A HUGGINGFACE DATASET
#--------------------------------------------------
def csv_to_dataset(csv_input_path):
    """
    Load the dataset from the csv file and convert to HuggingFace Dataset object
    Args:
    - csv_input_path: path to the csv file (train.csv, eval.csv)
    Return:
    - dataset: HuggingFace Dataset object
    """

    df = pd.read_csv(csv_input_path)

    df["sampling_rate"] = df["sampling_rate"].apply(lambda x: int(x))
    #shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    
    #Resample the audio_array column if it not 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    return dataset

def combine_data_csv(
    csv_dir="../datasets/combined",
    combined_data_list=[], #list of dataset names to combined
    dataframes=[],
    noise_frac=0.01,
    train_val_split=True,
    to_csv=True,
    shuffle_ratio=0.8
    
    ):
    """
    Load all the csv files and combine them into one dataframe
    Return:
     - the combined dataframe and output in csv files, 
     - if tran_val_split is set: splitted into train and validation set
    """
    print("Start combining the dataframes...")
    
    for folder in os.listdir(args.csv_dir):
        if combined_data_list and folder not in combined_data_list:
            continue
        folder = os.path.join(args.csv_dir,folder)
        if os.path.isdir(folder) and not folder.endswith(".ipynb_checkpoints"):
            print(folder)
            for file in os.listdir(folder):
                if file.endswith(".csv"):
                    print(f"Loading file {file}")
                    df = pd.read_csv(os.path.join(folder,file))
                    dataframes.append(df)
    print("Get total of {} dataframes".format(len(dataframes)))
    
    try: 

        combined_df = pd.concat(dataframes, ignore_index=True)
        
        #remove missing value rows
        # combined_df.dropna(inplace=True)
        combined_df['transcript'] = combined_df['transcript'].astype(str)

        #for empty transcript, only keep 1% of the row with empty transcript
        non_empty_df = combined_df[combined_df["transcript"].apply(lambda x: len(x) > 0)]
        empty_transcript_df = combined_df[combined_df["transcript"].apply(lambda x: len(x) == 0)]
        if len(empty_transcript_df) > 0:
            print("Number of Empty transcript: ",len(empty_transcript_df))
            empty_transcript_df = empty_transcript_df.sample(frac=noise_frac).reset_index(drop=True)
            
            combined_df = pd.concat([non_empty_df, empty_transcript_df], ignore_index=True)


        #FIXME: In combined_df, drop row that have empty audio path
        combined_df = combined_df[combined_df["audio"].apply(lambda x: len(x) > 0)]

        # shuffle
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)

        if not train_val_split:
            print("Not splitting the dataset into train and val sets, only returning the combined dataframe")
            if to_csv:
                combined_df.to_csv(f"{csv_dir}/combined.csv", index=False)
            else:
                return combined_df
        else:
            # split the dataset into train and validation set
            train_df = combined_df[:int(len(combined_df)*shuffle_ratio)]
            val_df = combined_df[int(len(combined_df)*shuffle_ratio):]

            if to_csv:
                os.makedirs(csv_dir, exist_ok=True)
                # save to csv
                train_df.to_csv(f"{csv_dir}/train_{combined_data_list[0]}.csv", index=False)
                val_df.to_csv(f"{csv_dir}/val_{combined_data_list[0]}.csv", index=False)
            else:
                print("Not saving to csv")
            return train_df, val_df
        print("Successfully generate combined datasets from different data!!")
    except ValueError as e:
        print("Unable to combine the datasets: {}".format(e))


#--------------------------------------------------
# PROCESSING A CORPUS TO A DATASET / CSV FILE
#--------------------------------------------------
def switchboard_to_ds(
    data_name="switchboard", #also implement for AMI, VocalSound, LibriSpeech, ...
    audio_dir='/switchboard_data/switchboard/audio_wav', 
    transcript_dir='/switchboard_data/switchboard/audio_wav',
    audio_segment_dir='/switchboard_data/audio_segments',
    batch_audio=[],
    batch_sr = [],
    batch_transcript=[],
    csv_dir = "../datasets/",
    to_csv = False,
    to_dataset = False,
    tokenize_speechlaugh = False,
):
    """
    Combines audio files and their corresponding transcripts into
    - a dataframe and save to csv if the to_csv flag is set
    - a HuggingFace Dataset object if the to_dataset flag is set

    Args:
        data_name (str): Name of the dataset
        audio_dir (str): Path to the directory containing audio files.
        transcript_dir (str): Path to the root directory containing transcript subfolders.
        batch_audio (list): List of path to audio file segments
        batch_transcript (list): List of transcript segments

    Returns:
        - switchboard_dataset (HuggingFace Dataset): Dataset object containing the audio and transcript data
        - OR df (pd.DataFrame): Dataframe containing the audio and transcript data
    """

    print(f"Flags: --to_csv: {to_csv}; --to_dataset: {to_dataset}; --tokenize_speechlaugh: {tokenize_speechlaugh}")

    for audio_file in tqdm(os.listdir(audio_dir), desc="Processing Switchboard dataset..."):
        if audio_file.endswith(".wav"):
            audio_path = os.path.join(audio_dir, audio_file) #audio_wav/sw02001A.wav
            transcript_lines = process_switchboard_transcript(
                audio_file,
                transcript_dir=transcript_dir,
                tokenize_speechlaugh=tokenize_speechlaugh
                ) #produce the transcript lines for each corresponding audio
            if transcript_lines is not None:
                audio_file_segments, audio_segments, transcripts_segments = cut_audio_based_on_transcript_segments(
                audio_path, 
                transcript_lines,
                padding_time=0.2,
                data_name=data_name,
                audio_segments_directory=audio_segment_dir,
                )
            else:
                print(f"Skipping audio file due to missing transcript: {audio_file}")
                continue
            
            # Append to the batch for each audio file
            batch_audio.extend(audio_file_segments)
            batch_sr.extend([16000]*len(audio_file_segments))
            batch_transcript.extend(transcripts_segments)

    print(f"Successfully combined audio and transcript segments for [{data_name}] data")
    print(f"Start creating dataset...")
    df = pd.DataFrame({
        "audio": batch_audio, #batch["audio"],
        "sampling_rate": batch_sr, #batch["sampling_rate"],
        "transcript": batch_transcript, #batch["transcript"]
    })


    if to_dataset:
        print(f"Saving {data_name} to HuggingFace Dataset on disk...")
        switchboard_dataset = Dataset.from_pandas(df)
        switchboard_dataset = switchboard_dataset.cast_column("audio", Audio(sampling_rate=16000))

        # Save the dataset to disk
        switchboard_dataset.save_to_disk(
            dataset_path=f"{csv_dir}/{data_name}_dataset",
            num_proc=8 # working on CPU so try num_proc=8 for 8 cores
        )
    if to_csv:
        csv_path = os.path.join(csv_dir, data_name)
        os.makedirs(csv_path, exist_ok=True)
        output_file = os.path.join(csv_path, f"{data_name}.csv") #../datasets/switchboard.csv
        df.to_csv(output_file, index=False)

    return switchboard_dataset if to_dataset else df
def vocalsound_to_ds(
    data_name="vocalsound",
    audio_dir='/vocalsound_data/audio_16k',
    batch_audio=[],
    batch_sr = [],
    batch_transcript=[],
    csv_dir = "../datasets/",
    to_csv = False,
    to_dataset = False,
    # vocalsound_dataset = None,
):
    """
    Process the vocalsound dataset
    """
    label_to_transcript = {
        "laughter": "[LAUGHTER]",
        "cough": "[COUGH]",
        "sigh": "[SIGH]",
        "sneeze": "[SNEEZE]",
        "sniff": "[SNIFF]",
        "throatclearing": "[THROAT-CLEARING]"
    }

    for audio_file in tqdm(os.listdir(audio_dir), desc="Processing VocalSound dataset..."):
        if audio_file.endswith(".wav"):
            audio_path = os.path.join(audio_dir, audio_file) #audio_16k/f2446_0_sniff.wav
            label = audio_file.split("_")[-1].split(".")[0] #sniff

            if label in label_to_transcript:
                audio, sample_rate = librosa.load(audio_path, sr=16000)
                if type(audio) is not np.ndarray:
                    audio = np.array(audio)
                batch_audio.append(audio)
                batch_sr.append(sample_rate)
                batch_transcript.append(label_to_transcript[label])
            else:
                print(f"Skipping audio file due to mismatching label: {audio_file}")
                continue

    print(f"Successfully added Vocal Sound to batch for dataset processing")
    
    df = pd.DataFrame({
        "audio": batch_audio, #batch["audio"] - audio array,
        "sampling_rate": batch_sr, #batch["sampling_rate"],
        "transcript": batch_transcript, #batch["transcript"]
    })

    if to_csv:
        csv_path = os.path.join(csv_dir, data_name)
        os.makedirs(csv_path, exist_ok=True)
        output_file = os.path.join(csv_path, f"{data_name}.csv") #../datasets/switchboard.csv
        df.to_csv(output_file, index=False)
    elif to_dataset:
        # Convert the dataframe to dataset
        vocalsound_dataset = Dataset.from_pandas(df)
        vocalsound_dataset = vocalsound_dataset.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Successfully processed VocalSound dataset!")
    
    return vocalsound_dataset if to_dataset else df

def ami_to_ds(
    data_name="ami",
    batch_audio=[],
    batch_sr = [],
    batch_transcript=[],
    csv_dir = "../datasets/",
    to_csv = False,
    to_dataset = False,
):
    # load the data in here
    ami_dataset = load_dataset("edinburghcstr/ami", "ihm", split="train+validation", cache_dir=prs.HUGGINGFACE_DATA_PATH, download_mode="force_redownload")
    if ami_dataset is None:
        print("Unable to load ami_dataset")
        
    print("Load ami dataset sucessfully, start processing...")
    print(ami_dataset)
    # only use the trained dataset
    for example in tqdm(ami_dataset, desc="Processing AMI dataset..."):
        # AMI Dataset have the format:
        """
        ami_dataset["train"][0] = 
        {'meeting_id': 'EN2001a',
        'audio_id': 'AMI_EN2001a_H00_MEE068_0000557_0000594',
        'text': 'OKAY',
        'audio': {'path': '/cache/dir/path/downloads/extracted/2d75d5b3e8a91f44692e2973f08b4cac53698f92c2567bd43b41d19c313a5280/EN2001a/train_ami_en2001a_h00_mee068_0000557_0000594.wav',
        'array': array([0.        , 0.        , 0.        , ..., 0.00033569, 0.00030518,
                0.00030518], dtype=float32),
        'sampling_rate': 16000},
        'begin_time': 5.570000171661377,
        'end_time': 5.940000057220459,
        'microphone_id': 'H00',
        'speaker_id': 'MEE068'
        }
        """
        # Process audio:
        try: 
            # audio_path = example["audio"]["path"]
            audio_array = example["audio"]["array"]
            sampling_rate = example["audio"]["sampling_rate"]
            transcript_line = example["text"]
            # Process transcript (extract timestamps, etc.):
            ami_text = process_ami_transcript(transcript_line)

            if ami_text:  # Check if processing was successful
                # batch_audio.append(audio_array)
                if type(audio_array) is not np.ndarray:
                    audio_array = np.array(audio_array)
                batch_audio.append(audio_array)
                batch_sr.append(sampling_rate)  
                batch_transcript.append(ami_text)
        except (FileNotFoundError, KeyError, TypeError) as e:
            print(f"Warning: Example file not found: {example}")
            continue
    
    df = pd.DataFrame({
        "audio": batch_audio, #batch["audio"],
        "sampling_rate": batch_sr, #batch["sampling_rate"],
        "transcript": batch_transcript, #batch["transcript"]
    })

    if to_csv:
        csv_path = os.path.join(csv_dir, data_name)
        os.makedirs(csv_path, exist_ok=True)
        output_file = os.path.join(csv_path, f"{data_name}.csv") #../datasets/switchboard.csv
        df.to_csv(output_file, index=False)

    if to_dataset:
        # Convert the dataframe to dataset
        ami_dataset = Dataset.from_pandas(df)
        ami_dataset = ami_dataset.cast_column("audio", Audio(sampling_rate=16000))
        # return ami_dataset
    
    print(f"Successfully processed AMI dataset!")
    return ami_dataset if to_dataset else df

def fsdnoisy_to_ds(
    data_name="fsdnoisy",
    batch_audio=[],
    batch_sr = [],
    batch_transcript=[],
    csv_dir = "../datasets/",
    to_csv = False,
    to_dataset = False,
):
    """
    Process the fsdnoisy dataset
    """
    fsdnoisy_dataset = load_dataset("sps44/fsdnoisy18k", split='train', cache_dir=prs.HUGGINGFACE_DATA_PATH, streaming=True)
    for example in tqdm(fsdnoisy_dataset, desc="Processing FSDNoisy dataset..."):
        audio_path = example["audio"]["path"]
        # audio_array = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        transcript_line = "" #making each transcript line empty for the noise dataset

        batch_audio.append(audio_path)
        batch_sr.append(sampling_rate)
        batch_transcript.append(transcript_line)

    df = pd.DataFrame({
        "audio": batch_audio,
        "sampling_rate": batch_sr,
        "transcript": batch_transcript,
    })
    df.dropna(inplace=True)

    if to_csv:
        csv_path = os.path.join(csv_dir, data_name)
        os.makedirs(csv_path, exist_ok=True)
        output_file = os.path.join(csv_path, f"{data_name}.csv") #../datasets/fsdnoisy.csv
        df.to_csv(output_file, index=False)
    if to_dataset:
        # Convert the dataframe to dataset
        fsdnoisy_dataset = Dataset.from_pandas(df)
        fsdnoisy_dataset = fsdnoisy_dataset.cast_column("audio", Audio(sampling_rate=16000))
#-------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_process", type=bool, default=False, help="Determine to skip or run processing steps for each dataset separately")
    parser.add_argument("--data_names", nargs="+", default=["switchboard", "ami", "vocalsound"], required=False, help="List of the datasets to process")
    
    parser.add_argument("--to_dataset", type=bool, default=False, help="Decide whether to return the dataset or not")
    
    parser.add_argument("--csv_dir", type=str, default="../datasets/switchboard/", help="Path to the directory containing audio files")
    parser.add_argument("--to_csv", type=bool, default=False, help="Whether to save the processed data to csv or not")

    # ARGUMENTS FOR SPECIAL PROCESSING
    parser.add_argument("--tokenize_speechlaugh", type=bool, default=False, help="Decide whether to tokenize to [SPEECH_LAUGH] or not")
    parser.add_argument("--noise_frac", type=float, default=0.01, help="Fraction of noise data to mix with original data") #FOR TRAINING

    # ARGUMENTS FOR COMBINING DATASETS
    parser.add_argument("--do_combine", type=bool, default=False, help="Determined if you want to combined different datasets into the same file")
    
    parser.add_argument("--train_val_split", type=bool, default=False, help="Decide whether not want to split the data")
    
#-------------------------------------------------------------------------------------------------------------

    args = parser.parse_args()
    
    combined = args.do_combine
    data_dir = args.csv_dir
    tokenized = args.tokenize_speechlaugh
    
    if not args.skip_process:
        for data_name in args.data_names:

            if data_name == "switchboard":
                audio_segment_dir = os.path.join(prs.GLOBAL_DATA_PATH, "switchboard_data", "audio_segments")
  
                if tokenized:
                    print("Processing Switchboard with tokenized [SPEECH_LAUGH]...")
                    audio_segment_dir = os.path.join(audio_segment_dir,"token_speechlaugh")
                    data_dir = os.path.join(data_dir, "token_speechlaugh")
                else:
                    print("Processing Switchboard with laughing word...")
                    audio_segment_dir = os.path.join(audio_segment_dir,"word_speechlaugh")
                    data_dir = os.path.join(data_dir, "word_speechlaugh")

                print(f"Process with: \n -Audio segment directory: {audio_segment_dir} \n -Data directory: {data_dir}\n")
                df = switchboard_to_ds(
                    data_name = data_name,
                    audio_dir=os.path.join(prs.GLOBAL_DATA_PATH, "switchboard_data", "switchboard","audio_wav"),
                    transcript_dir=os.path.join(prs.GLOBAL_DATA_PATH, "switchboard_data", "switchboard","transcripts"),
                    audio_segment_dir=audio_segment_dir,
                    csv_dir = data_dir,
                    to_dataset=args.to_dataset,
                    to_csv = args.to_csv,
                    tokenize_speechlaugh=args.tokenize_speechlaugh,
                )
            #----------------------------------------------end of switchboard process-------------------------------------------------
            elif data_name == "vocalsound":
                df = vocalsound_to_ds(
                    data_name = data_name,
                    audio_dir=os.path.join(prs.GLOBAL_DATA_PATH, "vocalsound_data", "audio_16k"),
                    csv_dir = args.csv_dir,
                    to_csv = args.to_csv,
                )
            elif data_name == "ami":
                df = ami_to_ds(
                    data_name = data_name,
                    csv_dir = args.csv_dir,
                    to_csv = args.to_csv,
                )
            elif data_name == "fsdnoisy":
                df = fsdnoisy_to_ds(
                    data_name = data_name,
                    csv_dir = args.csv_dir,
                    to_csv = args.to_csv,
                )
    #----------------------------End process each dataset separately--------------------------------------------

    #--------------------------------------------
    # PROCESS COMBINED DATASETS WITH CSV FILES
    #--------------------------------------------
    if args.do_combine:
        print("Combining csv files together...")
        if args.train_val_split:
            print("Spliting into train and val csv files for these data...")
            train_df, val_df = combine_data_csv(
                csv_dir=args.csv_dir,
                combined_data_list=args.data_names,
                noise_frac=args.noise_frac,
                train_val_split=args.train_val_split,

                to_csv=args.to_csv
            )
        else:
            print("Not splitting. Combining these data into one csv file...")
            combined_df = combine_data_csv(
                csv_dir=args.csv_dir,
                noise_frac=args.noise_frac,
                train_val_split=args.train_val_split,
                to_csv=args.to_csv
            )