import numpy as np
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor
from transformers.tokenization_utils_base import PaddingStrategy
import torch
from tqdm import tqdm

# Loading Noise Datasets ------------------------------------------------------------------------------------
# FIXME: Removing - not added noise to audio, instead it is a separate audio_batch
precomputed_fsdnoisy = []
# ------------------------------------------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int
    device: torch.device
    padding: Union[bool, str, PaddingStrategy] = True
    
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
            {"input_features": feature[model_input_name]} 
            for feature in features]
        #padding batch[input_features] 
        batch = self.processor.feature_extractor.pad(
            input_features, 
            padding=self.padding, 
            return_tensors="pt",
        )

        # INPUT FEATURES--------------------------------------------------------------------------------
        batch['input_features'] = batch['input_features'].to(self.device)
        #----------------------------------------------------------------------------------------

        # LABELS--------------------------------------------------------------------------------
        label_features = [
            {"input_ids": feature["labels"][0]} 
            for feature in features]
        #Padding batch[labels]
        labels_batch = self.processor.tokenizer.pad(
            label_features, 
            padding=self.padding, 
            return_tensors="pt",
        )
        labels_batch = labels_batch.to(self.device)

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        #----------------------------------------------------------------------------------------

        # ADD NOISE TO INPUT FEATURES-------------------------------------------------------------------------
    
        # if random.random() < 0.5:
        #     # add noise to the audio
        #     # instead of adding noise to each tensor in the batch, basically add noise to the whole batch together
        #     audio_batch = batch["input_features"]
        #     noise_audio = self.get_random_noise(audio_batch.shape)

        #     batch["input_features"] = add_noise(audio_batch, noise_audio)
        #---------------------------------------------------------------------------------------------------

        # MATCHING THE LABELS--------------------------------------------------------------------------------
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        #---------------------------------------------------------------------------------------------------

        return batch
    def get_random_noise(self, target_shape):
        """
        Get a random noise segment from a random noise dataset and preprocess it to match the target shape.
        
        Args:
            target_shape (Tuple): Shape of the target noise segment (batch_size, time_steps, feature_dim).
        
        Returns:
            noise_batch (torch.Tensor): The preprocessed noise segment in the tensor format tensor(batch_size, time_steps, feature_dim).
        """

        # Choose a random noise dataset
        # noise_dataset = random.choice([fsdnoisy_dataset, audioset_dataset]) #TODO: Store Noise Data in a csv path to load on demand
        # precomputed_noise_dataset = random.choice([precomputed_fsdnoisy, precomputed_audioset]) #FIXME: SHOULD WE STORE NOISE DATA IN A CSV FILE?

        batch_size, time_steps, feature_dim = target_shape
        
        #initialize noise batch
        noise_batch = torch.zeros(target_shape)

        for i in tqdm(range(batch_size), desc="Generating noise for audio batch..."):

            #FIXME- Only use FSDNoisy18k for now
            noise_audio = random.choice(precomputed_fsdnoisy) # already precomputed, include: noise_audio, noise_sr, preprocess_noise

            # add padding to noise audio if it's shorter than time_steps
            padding_needed = feature_dim - (len(noise_audio) % feature_dim)
            noise_audio = np.pad(noise_audio, (0, padding_needed), 'constant')

            #then reshape it
            noise_audio = noise_audio.reshape(-1, feature_dim)

            if noise_audio.shape[0] < time_steps:
                repeats = int(np.ceil(time_steps / noise_audio.shape[0])) #repeat noise to match time_steps
                noise_audio = np.tile(noise_audio, (repeats,1))

            start_idx = random.randint(0, noise_audio.shape[0] - time_steps)

            #segment the noise and match to the length of audio (time_steps)
            noise_segment = noise_audio[start_idx:start_idx + time_steps, :] 

            noise_batch[i, :, :] = torch.tensor(noise_segment)

        # return noise_segment
        return noise_batch


