import numpy as np
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from datasets import load_dataset
import torch

import utils.params as prs

fsdnoisy_dataset = load_dataset("sps44/fsdnoisy18k", split='train', cache_dir=prs.HUGGINGFACE_DATA_PATH, streaming=True)
audioset_dataset = load_dataset("benjamin-paine/audio-set-16khz", split='train', cache_dir=prs.HUGGINGFACE_DATA_PATH, streaming=True) 
# #TODO: Store Noise Data in a csv path to load on demand
# print(fsdnoisy_dataset)
# print(audioset_dataset)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int
    padding: Union[bool, str, PaddingStrategy] = True
    
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{"input_features": feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        #Padding the input features and create attention mask
        batch = self.processor.feature_extractor.pad(input_features, padding=self.padding, return_tensors="pt")
        

        #Padding batch[labels]
        with self.processor.as_target_processor():
            labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)


        # ADD NOISE TO INPUT FEATURES---------------------
        # # from FSDNoisy18k and AudioSet
        audio = batch["input_features"]
        noise = self.get_random_noise(audio.shape) 
        noise_audio = audio + noise  # Simple mixing, you might want to adjust levels
        batch["input_features"] = noise_audio
        #-----------------------------------------------

        # ADD NOISE TO LABELS---------------------
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        #-----------------------------------------------

        return batch
    def get_random_noise(target_shape):
    # Choose a random noise dataset
        noise_dataset = random.choice([fsdnoisy_dataset, audioset_dataset])

        # Sample a random audio from the chosen dataset
        random_index = random.randint(0, len(noise_dataset) - 1)
        noise_audio = noise_dataset[random_index]["audio"]["array"]

        # Preprocess noise (e.g., resampling, normalization to match speech audio)

        #match the length of the noise audio to the target shape by tiling the noise
        if len(noise_audio) < target_shape[0]:
            noise_audio = np.tile(noise_audio, int(np.ceil(target_shape[0] / len(noise_audio))))

        start_idx = random.randint(0, len(noise_audio) - target_shape[0])
        noise_segment = noise_audio[start_idx:start_idx + target_shape[0]]

        # Reshape the noise segment
        noise_segment = noise_segment.reshape(target_shape)

        return noise_segment

# Create data collator
# data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, padding=True)


