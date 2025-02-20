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
gdown 16qY7Y6KoDcr9jnQb4lofMCDXydjO7yo9 -O ../podcastfillers_data/PodcastFillers.zip

cd ../podcastfillers_data

unzip PodcastFillers.zip
```
### Buckeye

Similarly, we using the following command to download the Buckeye dataset and storing in corresponding path.

1. For original Buckeye datasets
```bash
gdown 1Vz1cTpTiMGAJoGaO57YPrY0JAGzGLPdy -O ../buckeye_data/Buckeye.zip

cd [global_path]/buckeye_data #/deepstore/datasets/hmi/speechlaugh-corpus/buckeye_data/

unzip Buckeye.zip
```

The structure of files existing in Buckeye folder is followed:
```batch
Buckeye /
    |_ s01
    |    |_ s0101a
    |    |   |_ s0101a.wav [original audio]
    |    |   |_ s0101a.txt [sentence-level transcript (no-timestamp)]
    |    |   |_ s0101a.words [word-level transcript (with timestamp)]
    |    |_ s0101b
    |    |_ ...
    |_ s02
    |_ ...
    |_ tagged_words_files
```
2. For clipped corpus: already processed by clipping the audio in seperate transcription based on different sentences
- v1
```bash
gdown 17mRLTnWhtrrUud25_Ab4lBN1voCqJd7N -O ../buckeye_data/buckeye_refs_wavs.zip

cd [global_path]/buckeye_data #/deepstore/datasets/hmi/speechlaugh-corpus/buckeye_data/

unzip buckeye_refs_wavs.zip
```

-v2
```bash
gdown 1TqKGZ3W0LB9JzMk0AS65wtJBFF10EIGr -O ../buckeye_data/buckeye_refs_wavs2.zip

unzip buckeye_refs_wavs2.zip
```

-v2 with short-form audio (< 30s)
```bash
gdown 1ELichR8Hx3Lq_jwvn2O8Sdwcb3Xb3X_A -O ../buckeye_data/buckeye_refs_wavs2_30.zip
unzip buckeye_refs_wavs2_30.zip
```

The structure of files existing in these clipped buckeye folder is followed:
```batch
buckeye_refs_wavs_* /
    |_ audio_wav
    |    |_ s0101a_1.wav

    |    |_ s0101b_2.wav
    |    |_ ...
    |_ transcripts
    |    |_ s0101a_1.txt

    |    |_ s0101b_2.txt
    |_   |_ ...
```

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
