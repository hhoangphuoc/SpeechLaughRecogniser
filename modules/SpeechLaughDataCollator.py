import numpy as np
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from datasets import load_dataset
import torch
from tqdm import tqdm

import params as prs
from utils.audio_process import preprocess_noise

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
    # max_length: int = 512
    
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Features format:
        [
            {
                "input_features": List[List[int]],
                "labels": List[int],
            },
            ...
        ]
        Output format:
        {
            "input_features": tensor(batch_size, max_length),
            "labels": tensor(batch_size, max_length),
        }
        """
        # split inputs and labels since they have to be of different lengths and need different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [
            {
                "input_features": feature[model_input_name]
            } 
            for feature in features
            ]
        #padding batch[input_features] 
        batch = self.processor.feature_extractor.pad(
            input_features, 
            padding=self.padding, 
            return_tensors="pt",
            # max_length=self.max_length
        )
        label_features = [
            {
                "input_ids": feature["labels"][0]
            } 
            for feature in features
        ]
        #Padding batch[labels]
        labels_batch = self.processor.tokenizer.pad(
            label_features, 
            padding=self.padding, 
            return_tensors="pt",
            # max_length=self.max_length,
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)


        # ADD NOISE TO INPUT FEATURES---------------------
        # # from FSDNoisy18k and AudioSet0
        audio = batch["input_features"]

        # instead of adding noise to each tensor in the batch, basically add noise to the whole batch together
        noise_batch = self.get_random_noise(audio.shape) 
        noise_audio = audio + noise_batch
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
    def get_random_noise(self, target_shape):
        """
        Get a random noise segment from a random noise dataset and preprocess it to match the target shape.
        
        Args:
            target_shape (Tuple): Shape of the target noise segment (batch_size, time_steps, feature_dim).
        """
        # num_samples = len(target_shape)
    #     noise_segments = []
    #     # Sample a random audio from the chosen dataset
    #     for _ in range(num_samples):
    #         # Sample a random audio from the chosen dataset
    #         noise_audio = next(iter(noise_dataset))["audio"]["array"]

    #         noise_sr = next(iter(noise_dataset))["audio"]["sampling_rate"]
    #         # Preprocess noise (e.g., resampling, normalization to match speech audio)
    #         noise_audio = preprocess_noise(noise_audio, noise_sr)
    #         print("Noise audio shape:", noise_audio.shape)

    #         target_shape = random.choice(target_shapes) # pick a random target shape from the batch

    #         #match the length of the noise audio to the target shape by tiling the noise
    #         if len(noise_audio) < target_shape[1]: # assuming target_shape is (batch_size, time_steps, feature_dim)
    #             # make sure to tile along the correct dimension (time_steps)
    #             repeats = [1] * len(target_shape)
    #             repeats[1] = int(np.ceil(target_shape[1] / len(noise_audio)))
    #             noise_audio = np.tile(noise_audio, repeats)

    #         start_idx = random.randint(0, len(noise_audio) - target_shape[1])
    #         noise_segment = noise_audio[start_idx:start_idx + target_shape[1]]
            
    #         noise_segment = noise_segment.reshape(target_shape[1:]) # reshape to (time_steps, feature_dim)
    #     # Stack the noise segments into a batch
    #     noise_segments = torch.tensor(np.stack(noise_segments, axis=0))
    #     return noise_segments

        # Choose a random noise dataset
        noise_dataset = random.choice([fsdnoisy_dataset, audioset_dataset])

        batch_size, time_steps, feature_dim = target_shape
        
        noise_batch = torch.zeros(target_shape)

        for i in tqdm(range(batch_size), desc="Generating noise for audio batch..."):
            noise_audio = next(iter(noise_dataset))["audio"]["array"]
            noise_sr = next(iter(noise_dataset))["audio"]["sampling_rate"]

            noise_audio = preprocess_noise(noise_audio, noise_sr)

            # add padding to noise audio if it's shorter than time_steps
            padding_needed = feature_dim - (len(noise_audio) % feature_dim)
            noise_audio = np.pad(noise_audio, (0, padding_needed), 'constant')

            #then reshape it
            noise_audio = noise_audio.reshape(-1, feature_dim)

            if noise_audio.shape[0] < time_steps:
                repeats = int(np.ceil(time_steps / noise_audio.shape[0])) #repeat noise to match time_steps
                noise_audio = np.tile(noise_audio, (repeats,1))
        # random_index = random.randint(0, len(noise_dataset) - 1)
        # noise_audio = noise_dataset[random_index]["audio"]["array"]

        # # Preprocess noise (e.g., resampling, normalization to match speech audio)

        # #match the length of the noise audio to the target shape by tiling the noise
        # if len(noise_audio) < target_shape[0]:
        #     noise_audio = np.tile(noise_audio, int(np.ceil(target_shape[0] / len(noise_audio))))

            start_idx = random.randint(0, noise_audio.shape[0] - time_steps)

            #segment the noise and match to the length of audio (time_steps)
            noise_segment = noise_audio[start_idx:start_idx + time_steps, :] 

            # FIXME - DO WE NEED TO RESHAPE AGAIN?
            #assign this segment to the corresponding batch
            noise_batch[i, :, :] = torch.tensor(noise_segment)

        # return noise_segment
        return noise_batch


