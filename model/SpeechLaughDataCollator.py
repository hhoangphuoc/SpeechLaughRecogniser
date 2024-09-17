import numpy as np
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor, DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from datasets import load_dataset
import torch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding(DataCollatorForSeq2Seq):
    processor: WhisperProcessor
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    # max_target_length: Optional[int] = None
    # pad_to_multiple_of: Optional[int] = None
    # pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{"input_features": feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        #Padding the input features and create attention mask
        batch = self.processor.feature_extractor.pad(input_features, padding=self.padding, return_tensors="pt", max_length=self.max_length)
        
        #Padding batch[labels]
        with self.processor.as_target_processor():
            labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Mix speech audio with noise data
        # from FSDNoisy18k and AudioSet
        audio = batch["input_features"]  # Assuming input_features are the audio representations
        noise = get_random_noise(audio.shape)  # Implement this to fetch random noise from your noise datasets
        noise_audio = audio + noise  # Simple mixing, you might want to adjust levels

        batch["input_features"] = noise_audio
        batch["labels"] = labels

        return batch
    
    # def add_noise(self, input_features):
    #     if torch.rand(1).item() < self.noise_prob:
    #         noise = torch.randn_like(input_features) * self.noise_level
    #         return input_features + noise
    #     return input_features

# Create data collator
# data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, padding=True)


fsdnoisy_dataset = load_dataset("sps44/fsdnoisy18k")
audioset_dataset = load_dataset("benjamin-paine/audio-set-16khz")
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