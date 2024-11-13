from matplotlib.pylab import pareto
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
    padding: Union[bool, str, PaddingStrategy] = True

    def __call__(
            self, 
            features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
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
            print("Starting Data Collator ------------------------------")

            # Get max length of input features and labels
            max_input_length = max([feature["input_features"].shape[0] for feature in features])
            max_label_length = max([feature["labels"].shape[0] for feature in features])

            max_input_length = (max_input_length + 7) // 8 * 8 # make divisible by 8
            max_label_length = (max_label_length + 7) // 8 * 8 # make divisible by 8

            print(f"Max input features length: {max_input_length} \n")
            print(f"Max labels length: {max_label_length} \n")

            #--------------------------------------------------------------------------------
            # Verify input types and shapes
            for i, feat in enumerate(features):
                assert isinstance(feat["input_features"], torch.Tensor), \
                    f"Feature {i}: input_features is {type(feat['input_features'])}, expected torch.Tensor"
                assert isinstance(feat["labels"], torch.Tensor), \
                    f"Feature {i}: labels is {type(feat['labels'])}, expected torch.Tensor"
                # assert feat["input_features"].size(-1) == 80, \
                #     f"Feature {i}: Expected 80 mel features, got {feat['input_features'].size(-1)}"

            
            #--------------------------------------------------------------------------------

            #input_features in tensor format (time_steps, feature_dim)
            input_features = [
                {"input_features": feature["input_features"]} 
                for feature in features]
            

            batch = self.processor.feature_extractor.pad(
                input_features, 
                padding=True,
                max_length=max_input_length,
                pad_to_multiple_of=8, # set tensor format divisible by 8
                return_tensors="pt",
            ) 

            #--------------------------------------------------------------------------------
            label_features = [
                {"input_ids": feature["labels"]} 
                for feature in features
            ]
            
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                padding=True,
                max_length=max_label_length,
                pad_to_multiple_of=8, # match tensor format that divisible by 8
                return_tensors="pt",
            )
            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
                
            # Now check first token
            if labels.size(1) > 0 and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]  # remove the first token
            
            batch["labels"] = labels 
            #---------------------------------------------------------------------------------------------------
            
            # Verify final shapes
            print("\nBatch shapes after padding:")
            print(f"Input features: {batch['input_features'].shape} (batch_size, time_steps, n_mels)")
            print(f"Labels: {batch['labels'].shape} (batch_size, sequence_length) \n")  
            #---------------------------------------------------------------------------------------------------


            # # Move batch to GPU after all padding is done

            # batch = {
            #     "input_features": batch["input_features"].to(
            #         self.device, 
            #         non_blocking=True), #FIXME - non_blocking for faster transfer
            #     "labels": batch["labels"].to(
            #         self.device, 
            #         non_blocking=True),
            # }
            batch = {
                k : v.to(self.device, non_blocking=True)
                for k, v in batch.items()
            }

            print("Finished Data Collator!, moving to GPU for training")
            return batch
        
        except Exception as e:
            print(f"Error in DataCollator: {e}")
            print("\nDetailed feature information:")
            for i, feat in enumerate(features):
                print(f"\nFeature {i}:")
                for k, v in feat.items():
                    print(f"  {k}: shape={v.shape if isinstance(v, torch.Tensor) else 'N/A'}, "
                        f"type={type(v)}")
            raise

        finally:
            # Clear GPU cache after data collator is done
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

