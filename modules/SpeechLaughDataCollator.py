from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor
from transformers.tokenization_utils_base import PaddingStrategy
import torch

# ------------------------------------------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int
    # padding: Union[bool, str] = True

    def __call__(
            self, 
            features
    ):
        """
        Features format:
        [   
            {
                "input_features": torch.Tensor(80, time_steps_i),
                "labels": torch.Tensor(sequence_length) or List[int]
            },
            ...
        ]
        Output format:
        {
            "input_features": torch.Tensor(batch_size, 80, max_time_steps),
            "labels": torch.Tensor(batch_size, max_sequence_length),
        }
        """
        if not features:
            return None
        # try:
        # ==========================================================================
        #                           Process Input Features
        # ==========================================================================

        input_features = [{
            "input_features": feature["input_features"]
        } for feature in features]


        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        ) # FIXME - IF STILL OOM, TRY PAD WITH NUMPY

        # ====================================================================================
        #                                   Process Input Features, Labels
        # ====================================================================================
        labels_features = [{
            "input_ids": feature["labels"]
        } for feature in features]

        labels_batch = self.processor.tokenizer.pad(
            labels_features,
            padding=True,
            return_tensors="pt",
        )

        # Replace padding token id with -100 to ignore in loss
        labels = labels_batch['input_ids'].masked_fill(labels_batch['attention_mask'].ne(1), -100)

        # Remove decoder_start_token_id if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                    labels = labels[:, 1:]
        
        batch["labels"] = labels

        return batch
