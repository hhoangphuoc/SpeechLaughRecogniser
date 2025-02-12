import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import gc
import random
import re
import warnings
import numpy as np
import pandas as pd
import torch
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import load_dataset, ClassLabel, load_from_disk

import evaluate

from typing import List, Dict, Union
from dataclasses import dataclass

from huggingface_hub import login, HfApi
from dotenv import load_dotenv

import jiwer
# REMOVE TF WARNINGS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)

# LOGIN TO HUGGINGFACE ===============================================
# load_dotenv()
# hf_token = os.getenv("HUGGINGFACE_TOKEN")

# login(token=hf_token)
#====================================================================

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

# ================================== UTILS ===========================================
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(batch):
    batch["transcript"] = re.sub(chars_to_remove_regex, '', batch["transcript"]).lower()
    return batch

def extract_all_chars(batch):
    """
    Function to extract all characters in the dataset
    and added it to the vocabs.
    """
    all_text = " ".join(batch["transcript"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def show_random_elements(dataset, num_examples=10):
    """
    Function to show random elements in the dataset
    and print it out.
    """
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    print(dataset[picks])

min_audio_length_in_seconds = 0.2  #  Example: Filter out audio < 0.2 seconds
sampling_rate = 16000  # Make sure this matches your feature extractor
def filter_short_audio(example):
    audio_length = len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
    return audio_length >= min_audio_length_in_seconds

#============================================================================================================



# ================================== PATHS ==================================================================
global_checkpoint_path = "../fine-tuned/wav2vec2/finetuned-wav2vec2-buckeye-v4/"
#============================================================================================================


# ================================== LOAD DATASET ============================================================

# Load splitted dataset from disk
# swb_train = load_from_disk("../datasets/switchboard/whisper/swb_train")
data_train = load_from_disk("../datasets/buckeye3/buckeye_train")
data_eval = load_from_disk("../datasets/buckeye3/buckeye_eval")

print("Original Train Dataset (70%):", data_train)
print("Original Validation Dataset (10%):", data_eval)

# =========================================== PROCESS DATASET FOR FINETUNED WITHOUT <LAUGH>  ==================================================
# #Remove <LAUGH> in transcript and filter out the dataset with laughter-only transcript: <LAUGH> 
# data_train = data_train.map(lambda x: {'transcript': x['transcript'].replace('<LAUGH>', '')}, desc="Removing <LAUGH> for NOLAUGH finetuning in Train dataset")
# data_train = data_train.filter(lambda x: len(x["transcript"]) > 0 and x["transcript"] !="<LAUGH>", desc="Filtering out <LAUGH only> and empty transcript in Train dataset")

# data_eval = data_eval.map(lambda x: {'transcript': x['transcript'].replace('<LAUGH>', '').strip()}, desc="Removing <LAUGH> for NOLAUGH finetuning in Eval dataset")
# data_eval = data_eval.filter(lambda x: len(x["transcript"]) > 0 and x["transcript"] !="<LAUGH>", desc="Filtering out <LAUGH only> and empty transcript in Eval dataset")
#=============================================UNCOMMENTED ABOVE CODE IF USE (FINETUNED WITHOUT <LAUGH>)===========================



# # ============================================ PROCESS DATASET FOR FINETUNED WITH <LAUGH> ==================================================
# Remove empty transcript
data_train = data_train.filter(lambda x: len(x["transcript"]) > 0, desc="Filtering out empty transcript in Train dataset")
data_eval = data_eval.filter(lambda x: len(x["transcript"]) > 0, desc="Filtering out empty transcript in Eval dataset")

# Transform the <LAUGH> token into "<" for CTC
data_train = data_train.map(lambda x: {'transcript': x['transcript'].replace('<LAUGH>', '<')}, desc="Replacing <LAUGH> with < for CTC in Train dataset")
data_eval = data_eval.map(lambda x: {'transcript': x['transcript'].replace('<LAUGH>', '<')}, desc="Replacing <LAUGH> with < for CTC in Eval dataset")
#=======================================================UNCOMMENTED ABOVE CODE IF USE (FINETUNED WITH <LAUGH>) ===========================================================


# ============================================ FILTER OUT SHORT AUDIO ==================================================
# Filter out audio < 0.2 seconds
data_train = data_train.filter(filter_short_audio, desc="Filtering out audio < 0.2 seconds in Train dataset")
data_eval = data_eval.filter(filter_short_audio, desc="Filtering out audio < 0.2 seconds in Eval dataset")
#=====================================================================================================================


# =====================================================================================================================
print("Train Dataset (after filtered):", data_train)
print("Validation Dataset (after filtered):", data_eval)
# =====================================================================================================================


# ================================================ PREPROCESSING VOCAB ================================================

# remove special characters
data_train = data_train.map(
    remove_special_characters,
    desc="Removing special characters in data_train"
    )
data_eval = data_eval.map(
    remove_special_characters,
    desc="Removing special characters in data_eval"
    )

# VOCAB MAPPING
vocab_train = data_train.map(
    extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=data_train.column_names,
    desc="Extracting vocab in data_train"
)
vocab_eval = data_eval.map(
    extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=data_eval.column_names,
    desc="Extracting vocab in data_eval"
)

# combined 2 sets of vocab in training and evaluation dataset
vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_eval["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
print("Vocab Dictionary:",vocab_dict)
print(len(vocab_dict))

# Add the " " as a new token class
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

# Add the CTC "blank token" to the vocabulary
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

print("Vocab Dictionary:",vocab_dict)

# Save the vocab
with open("vocab_buckeye_v4.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)

#=====================================================================================================================


# CONFIGURE MODEL COMPONENTS
tokenizer = Wav2Vec2CTCTokenizer("./vocab_buckeye_v4.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Save the tokenizer
print("Saving Wav2Vec2 tokenizer and feature extractor...")
tokenizer.save_pretrained(os.path.join(global_checkpoint_path, "tokenizer"))
feature_extractor.save_pretrained(os.path.join(global_checkpoint_path, "feature_extractor"))

# =============================== PROCESSING DATASET =====================================
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcript"]).input_ids
    
    return batch

data_train = data_train.map(
    prepare_dataset, 
    remove_columns=data_train.column_names, 
    # batch_size=8, num_proc=4
    desc="Preparing dataset for data_train"
)
data_eval = data_eval.map(
    prepare_dataset,
    remove_columns=data_eval.column_names,
    # batch_size=8, num_proc=4
    desc="Preparing dataset for data_eval"
)

# ================================== METRICS ===========================================
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)

    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    eval_transformation = jiwer.Compose([
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemovePunctuation(),
            jiwer.SubstituteWords({
                "uhhuh": "uh-huh",
                "uh huh": "uh-huh",
                "mmhmm": "um-hum",
                "mm hmm": "um-hum",
                "mmhum": "um-hum",
                "mm hum": "um-hum",
                "umhum": "um-hum",
                "um hum": "um-hum",
                "umhmm": "um-hum",
            }),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ToLowerCase()
    ])

    pred_str = eval_transformation(pred_str)
    label_str = eval_transformation(label_str)


    wer = jiwer.wer(reference=label_str, hypothesis=pred_str)

    return {"wer": wer}



# ================================== MODEL ============================================
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-lv60",
    cache_dir="../ref_models/pre_trained",
    local_files_only=True,
    mask_time_length=5,
    mask_time_prob=0.3, #FIXME: change to 0.2
    mask_feature_prob=0.3,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.freeze_feature_encoder()

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,    # Stop if WER doesn't improve for 3 evaluations
    early_stopping_threshold=0.005, # Consider it an improvement if WER decreases by 0.005 (0.5%)
)

# ================================== TRAINING ARGUMENTS ============================================
training_args = TrainingArguments(
    output_dir=global_checkpoint_path,
    group_by_length=True,
    per_device_train_batch_size=16, #16
    gradient_accumulation_steps=2, 
    evaluation_strategy="steps",
    num_train_epochs=40, #FIXME: increase to 30 or even 40 if it is not converge

    gradient_checkpointing=True,
    fp16=True,
    adam_beta2=0.98,

    save_steps=25, #50
    eval_steps=25, #50
    logging_steps=25, #50

    learning_rate=6e-5, #1e-4 - FIXME: use 5e-5 due to small dataset
    weight_decay=0.005,
    warmup_ratio=0.15, #FIXME: change to 0.1 if for
    
    save_total_limit=3,
    
    torch_empty_cache_steps=100, # Force garbage collection if necessary
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=data_train,
    eval_dataset=data_eval,
    tokenizer=processor.feature_extractor,
    # callbacks=[early_stopping_callback]
)

# ================================== TRAINING PROCESS ============================================


#   MONITOR GPU MEMORY
#=============================================
def cleanup_workers():
    """Cleanup function for workers"""
    print("Cleaning CUDA memory cache...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
#=============================================

try:
    print("Training the model...")
    trainer.train()

except Exception as e:
    print(f"Error in training: {e}")
    cleanup_workers() #clear the CUDA memory cache
finally:
    log_history = trainer.state.log_history
    # save the log history to txt file
    with open(os.path.join("../logs/wav2vec2", "log_history_finetuned-wav2vec2-buckeye-v4.txt"), "w") as f:
        for entry in log_history:
            f.write(str(entry) + "\n")
    cleanup_workers() #clear the CUDA memory cache
# trainer.train()

model.save_pretrained(os.path.join(global_checkpoint_path, "model"))
#================================================================================================
# trainer.push_in_progress = None
# trainer.push_to_hub("wav2vec2-large-lv60-speechlaugh-swb")