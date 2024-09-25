# SpeechLaughRecogniser
An ASR model for transcribing laughter in Speech Laugh audio

## Dataset
### Switchboard data
- Using gdown to download the `.zip` file data and unzip it.

```bash
gdown 1VlQlyY3v3wtT2S047lwlTirWisz5mQ18 -O /path/to/data/switchboard.zip

#path = ../data/switchboard # local datasets path
#path=/deepstore/datasets/hmi/switchboard # global datasets path

cd path/to/data #../data/switchboard

unzip switchboard.zip
```

### Other datasets (Ami, VocalSound, FSD50K-noisy, etc.)
- Download these datasets from HuggingFace datasets and saving to `data/huggingface_data` folder

1. First set the path to HuggingFace cache to this folder
```bash
$ export HF_DATASETS_CACHE="../data/huggingface_data"
```

2. Then download the datasets, given the dataset name in HuggingFace as follow:
- ami: "edinburghcstr/ami" "ihm" split="train"
- vocal_sound: "flozi00/VocalSound_audio_16k" split="train"
- fsd50k_noisy: "sps44/fsdnoisy18k"
- audioset: "benjamin-paine/audio-set-16khz"

3.
