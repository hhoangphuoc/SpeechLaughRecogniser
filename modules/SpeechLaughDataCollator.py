import numpy as np
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor
from transformers.tokenization_utils_base import PaddingStrategy
import torch
from tqdm import tqdm

# ------------------------------------------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int
    device: torch.device
    padding: Union[bool, str, PaddingStrategy] = "longest"
    # max_length_input: int = 480000 # 30 seconds @ 16kHz
    # max_length_labels: int = 448 # Whisper's max sequence length

    def __call__(self, 
                 features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Features format:
        [
            {
                "input_features": List[torch.Tensor]; each tensor is (time_steps, n_mels) - batch_size already removed
                "labels": List[int]; each tensor is (sequence_length) : number of tokens in the transcript - batch_size already removed
            },
            ...
        ]
        Output format:
        {
            "input_features": torch.Tensor(batch_size, max_length),
            "labels": torch.Tensor(batch_size, max_length),
        }
        """
        if not features:
            return None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            #input_features in tensor format (batch_size, time_steps, feature_dim)
            input_features = [
                {"input_features": feature["input_features"]} 
                for feature in features]
            
            #-------------- -----------------------------------------
            # # Handle input features - FIXME- FUNCTION TO HANDLE INPUT FEATURES, INCASE ABOVE DOESNT WORK
            # input_features = []
            # for feature in tqdm(features, desc="Processing features..."):
            #     feat = feature["input_features"]
            #     if isinstance(feat, (list, np.ndarray)):
            #         feat = torch.tensor(feat)
            #     input_features.append({"input_features": feat})
            # #-------------------------------------------------------

            batch = self.processor.feature_extractor.pad(
                input_features, 
                return_tensors="pt",
                padding=self.padding,
                # max_length=self.max_length_input, #FIXME - Change to max_length if doesnt work
                truncation=True,
            ) 

            # LABELS--------------------------------------------------------------------------------
            # # Handle labels - FIXME- FUNCTION TO HANDLE LABELS, INCASE IT DOESNT WORK
            # label_features = []
            # for feature in tqdm(features, desc="Processing labels..."):
            #     label = feature["labels"]
            #     if isinstance(label, (list, np.ndarray)):
            #         label = torch.tensor(label)
            #     label_features.append({"input_ids": label})
            #--------------------------------------------------------------------------------

            label_features = [{"input_ids": feature["labels"]} for feature in features]

            #--------------------------------------------------------------------------------
            
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                padding=self.padding,
                # max_length=self.max_length_labels,
                truncation=True,
                return_tensors="pt",
            )
            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
                
            # Now check first token
            if labels.size(1) > 0 and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]  # remove the first token
            
            batch["labels"] = labels 
            #---------------------------------------------------------------------------------------------------
            
            # # Move batch to GPU after all padding is done
            # print("Finished Data Collator for padding, moving to GPU for training...")
            batch = {
                "input_features": batch["input_features"].to(
                    self.device, 
                    non_blocking=True), #FIXME - non_blocking for faster transfer
                "labels": batch["labels"].to(
                    self.device, 
                    non_blocking=True),
            }
            print("Finished Data Collator!, moving to GPU for training")
            return batch
        
        except Exception as e:
            print(f"Error in DataCollator: {e}")
            raise

        finally:
            # Clear GPU cache after data collator is done
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

