# SpeechLaughRecogniser
An ASR model for transcribing laughter and speech-laugh in Conversational Speech

# Dataset
The global path to the dataset storage is located in:
```bash
path=/deepstore/datasets/hmi/speechlaugh-corpus # global data path
```

### Switchboard data

- Using gdown to download the `.zip` file data and then unzip it.

```bash
gdown 1VlQlyY3v3wtT2S047lwlTirWisz5mQ18 -O /path/to/data/switchboard.zip

#in this case: path/to/data can be: [global_path]/switchboard_data 

cd path/to/data #/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data

unzip switchboard.zip
```
after unzipping, the data will contain the following folders:
```bash
switchboard_data \
        |_ audio_wav
        |_ transcripts
```

- Generate audio_segments folder, this stored in the following path
```bash
 path=[global_path]/switchboard_data/audio_segments
```

### PodcastFillers
Similarly, download the PodcastFillers dataset using `gdown` and unzip it as follow:
```bash
gdown 16qY7Y6KoDcr9jnQb4lofMCDXydjO7yo9 -O [global_path]/podcastfillers_data/PodcastFillers.zip

cd [global_path]/podcastfillers_data

unzip PodcastFillers.zip
```
### Buckeye
TODO:

---


<!-- ### VocalSound
- Download the dataset from [VocalSound](https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1) and save it to `path/to/data/vocalsound_data` folder

```bash
wget -O vocalsound_16k.zip https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1

#path=/deepstore/datasets/hmi/speechlaugh-corpus/vocalsound_data

unzip vocalsound_16k.zip
```
- The path to the data would be:
```bash
path=/deepstore/datasets/hmi/speechlaugh-corpus/vocalsound_data/audio_16k
``` -->

### For Datasets from HuggingFace Datasets (e.g. AMI)
- Download these datasets from HuggingFace datasets and saving to `data/huggingface_data` folder. However, most of these datasets are cleaned, and only contain normal speech.

- If you want to have the dataset that existing paralinguistic events (e.g. laughter, speechlaugh), it is recommended to download it directly from AMI Corpus

Download the datasets from `HuggingFace Datasets` and store it locally through the following steps:

1. First set the path to HuggingFace cache to this folder
```bash
$ export HF_DATASETS_CACHE="../data/huggingface_data"

# or change to the global datasets
$ export HF_DATASETS_CACHE="/deepstore/datasets/hmi/speechlaugh-corpus/huggingface_data"

```

2. Then download the datasets, given the dataset name in HuggingFace, for example:
- ami: "edinburghcstr/ami" "ihm" split="train"

# Preprocessing

<!-- 3 seperate datasets, corresponding to 3 types of token using for training and evaluation, they are:
- `switchboard_speech`
```bash
path=/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data/swb
```

- `switchboard_laugh`
```python
path=/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data/swb_laugh

"Laughter dataset with `intext = True`": Dataset({
    features: ['audio', 'sampling_rate', 'transcript'],
    num_rows: 6900
})
```

- `switchboard_speechlaugh`
```python
path=/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data/swb_speechlaugh

"Speech-laugh dataset": Dataset({
    features: ['audio', 'sampling_rate', 'transcript'],
    num_rows: 7672
})
``` -->

# Training

## Datasets Uses for Training

We use `Switchboard` as the dataset for training, since it contains both laughter and speechlaugh events, which is benefits the purpose of our training.

The dataset has been preprocessed, audio-matching, cleaned, retokenized and seperated in train (`swb_train`), dev (`swb_eval`) and validation set (`swb_test`). The datasets stored locally in the following path:
```bash
[path\to\directory]\datasets\switchboard\
```

The summary of these datasets is below:
```python
Train Dataset (70%): Dataset({
    features: ['audio', 'sampling_rate', 'transcript'],
    num_rows: 185402
})
Validation Dataset (10%): Dataset({
    features: ['audio', 'sampling_rate', 'transcript'],
    num_rows: 20601
})
Test Dataset (20%): Dataset({
    features: ['audio', 'sampling_rate', 'transcript'],
    num_rows: 51501
})
```

## Training Process

**NOTES:** During training process, we might encountered the caching has been locally store with enormous amount of data, especially after the `prepare_dataset`steps. This is due to the tensors are stored in the cache file in the same `dataset_directory`, named as `cache-*.arrows`

These cache files are really large and only used during training process. So after training, consider removing them to get rid of memory capacity issue.

To do this, we can:

1. Check disk usage of models directory, datasets in global storage, navigate to the storage (`dataset_dir`) and use `du` command.
    ```bash
    cd /path/to/storage
    du -sh * | sort -hr
    ```

2. To remove cache files of these datasets in the same place as the data storage, cause by the flag: `load_from_cache_file=True`, set the flag in dataset mapping process to `load_from_cache_file=False`, and use the following to remove:
    ```bash
    cd /path/to/storage
    rm -rf cache-*
    ```

## Training Configuration

---
# Testing the model
There are 2 options for evaluating the models, either for `original pre-trained model` or `finetuned model`

The `swb_test` dataset is used for evaluating the models, expected to be located in the path: `path=[path\to\directory]\datasets\switchboard\swb_test`

### Evaluate pre-trained model

### Evaluate finetuned model
