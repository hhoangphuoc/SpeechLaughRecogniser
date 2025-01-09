# SpeechLaughRecogniser
An ASR model for transcribing laughter in Speech Laugh audio

## Dataset
### Switchboard data
```bash
path=/deepstore/datasets/hmi/speechlaugh-corpus # global data path
```
- Using gdown to download the `.zip` file data and unzip it.

```bash
gdown 1VlQlyY3v3wtT2S047lwlTirWisz5mQ18 -O /path/to/data/switchboard.zip

#path=/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data # global datasets path

cd path/to/data #/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data

unzip switchboard.zip

# after unzip, the data will contain the following folders:
# - audio_wav
# - transcripts

```
- Generate audio_segments folder, this could be stored in the following path
```bash
path=/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data/audio_segments
```

### VocalSound
- Download the dataset from [VocalSound](https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1) and save it to `path/to/data/vocalsound_data` folder

```bash
wget -O vocalsound_16k.zip https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1

#path=/deepstore/datasets/hmi/speechlaugh-corpus/vocalsound_data

unzip vocalsound_16k.zip
```
- The path to the data would be:
```bash
path=/deepstore/datasets/hmi/speechlaugh-corpus/vocalsound_data/audio_16k
```

### Other datasets (Ami, VocalSound, FSD50K-noisy, etc.)
- Download these datasets from HuggingFace datasets and saving to `data/huggingface_data` folder

1. First set the path to HuggingFace cache to this folder
```bash
$ export HF_DATASETS_CACHE="../data/huggingface_data"

# or change to the global datasets
$ export HF_DATASETS_CACHE="/deepstore/datasets/hmi/speechlaugh-corpus/huggingface_data"

```

2. Then download the datasets, given the dataset name in HuggingFace as follow:
- ami: "edinburghcstr/ami" "ihm" split="train"
<!-- - vocal_sound: "flozi00/VocalSound_audio_16k" split="train" -->
- fsd50k_noisy: "sps44/fsdnoisy18k"
- audioset: "benjamin-paine/audio-set-16khz"

## Preprocessing

## Datasets Uses for Training
After preprocessing, we have seperated, cleaned, retokenized and stored the datasets in 3 seperate datasets, corresponding to 3 types of token using for training and evaluation, they are:
- `switchboard_speech`
```bash
path=/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data/swb_speech
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
```

## Training
> To check disusage of models directory, datasets in global storage, navigate to the storage and use `du` command.
```bash
cd /path/to/storage
du -sh * | sort -hr
```
> To remove datasets store in cache files in the same place as the data storage, cause by the flag: `load_from_cache_file=True` use the following to remove:
```bash
cd /path/to/storage
rm -rf cache-*
```
