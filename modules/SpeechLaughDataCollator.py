import numpy as np
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor
from transformers.tokenization_utils_base import PaddingStrategy
import torch
from torch.nn.utils.rnn import pad_sequence # using pad_sequence to pad labels
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
    
    # FIXME: TRY USING THIS IF CURRENT ONE IS NOT WORKING --------------------------------
    # def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    #     # Get max length for padding
    #     max_length = max([feat["input_features"].shape[0] for feat in features])
        
    #     # Pad input features
    #     padded_inputs = []
    #     for feat in features:
    #         current_feat = feat["input_features"]
    #         if isinstance(current_feat, list):
    #             current_feat = torch.tensor(current_feat)
            
    #         # Calculate padding needed
    #         pad_length = max_length - current_feat.shape[0]
    #         if pad_length > 0:
    #             # Pad along the time dimension (dim=0)
    #             padding = torch.zeros((pad_length, current_feat.shape[1]))
    #             current_feat = torch.cat([current_feat, padding], dim=0)
            
    #         padded_inputs.append(current_feat)
        
    #     # Stack all padded inputs
    #     input_features = torch.stack(padded_inputs)
        
    #     # Handle labels
    #     labels_list = [feat["labels"] for feat in features]
    #     labels = pad_sequence(
    #         [torch.tensor(label) for label in labels_list],
    #         batch_first=True,
    #         padding_value=self.processor.tokenizer.pad_token_id
    #     )
        
    #     # Create attention mask for labels
    #     labels_attention_mask = (labels != self.processor.tokenizer.pad_token_id)
        
    #     # Replace padding with -100 for loss calculation
    #     labels = labels.masked_fill(~labels_attention_mask, -100)
        
    #     # Remove decoder_start_token_id if present
    #     if labels.size(1) > 0 and (labels[:, 0] == self.decoder_start_token_id).all():
    #         labels = labels[:, 1:]
        
    #     # Move everything to device
    #     batch = {
    #         "input_features": input_features.to(self.device),
    #         "labels": labels.to(self.device)
    #     }
        
    #     return batch
    #-------------------------------------------------------

    def __call__(self, 
                 features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
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
        if not features:
            return None
        # REMARKS: NO NEED TO CHANGE TO TENSOR, AS IT'S ALREADY IN TENSOR FORMAT

        #input_features in tensor format (batch_size, time_steps, feature_dim)
        # input_features = [
        #     {"input_features": feature[model_input_name]} 
        #     for feature in features]
        input_features = []
        for feature in tqdm(features, desc="Padding input_features..."):
            current_features = feature[model_input_name]
            if isinstance(current_features, list):
                current_features = torch.tensor(current_features) #convert list to tensor if not already
            if current_features.ndim == 2:
                current_features = current_features.unsqueeze(0)
            input_features.append({"input_features": current_features})
        
        batch = self.processor.feature_extractor.pad(
            input_features, 
            padding=self.padding, 
            return_tensors="pt",
        ) #padding input_features

        # LABELS--------------------------------------------------------------------------------
        label_features = []
        for feature in tqdm(features, desc="Padding labels..."):
            label = feature["labels"]

            if isinstance(label, list):
                label = torch.tensor(label) #convert list to tensor if not already
            if label.ndim == 1:
                label = label.unsqueeze(0)
            label_features.append({"input_ids": label})
        
        labels_batch = self.processor.tokenizer.pad(
            label_features, # Only padding with input_ids
            padding=self.padding, 
            return_tensors="pt",
        )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            
        # Now check first token
        if labels.size(1) > 0 and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]  # remove the first token
        
        batch["labels"] = labels 
        #---------------------------------------------------------------------------------------------------
        
        # Move batch to GPU after all padding is done
        print("Finished Data Collator for padding, moving to GPU for training...")
        batch = {
            "input_features": batch["input_features"].to(self.device),
            "labels": batch["labels"].to(self.device),
        }
        
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


