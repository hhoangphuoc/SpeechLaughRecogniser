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
from IPython.display import display, HTML
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, ClassLabel, load_from_disk
import evaluate

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from huggingface_hub import login, HfApi
from dotenv import load_dotenv

from preprocess import (
    split_dataset, 
    transform_number_words,
    find_total_laughter_speechlaugh
)
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
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

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



# ================================== LOAD DATASET ============================================================
# switchboard = load_dataset("hhoangphuoc/switchboard")
# switchboard = load_from_disk("../datasets/switchboard/swb_all")
# print(switchboard)

# #Shuffle the dataset
# switchboard = switchboard.shuffle(seed=42)


# swb_train, swb_eval, swb_test = split_dataset(
#     switchboard, 
#     split_ratio=0.8, 
#     val_split_ratio=0.1
# )

# Load splitted dataset from disk
swb_train = load_from_disk("../datasets/switchboard/whisper/swb_train")
# Transform the <LAUGH> token into "<" for CTC
swb_train = swb_train.map(lambda x: {'transcript': x['transcript'].replace('<LAUGH>', '<')}, desc="Replacing <LAUGH> with < for CTC in Train dataset")

swb_eval = load_from_disk("../datasets/switchboard/whisper/swb_eval")
swb_eval = swb_eval.map(lambda x: {'transcript': x['transcript'].replace('<LAUGH>', '<')}, desc="Replacing <LAUGH> with < for CTC in Eval dataset")

swb_test = load_from_disk("../datasets/switchboard/whisper/swb_test")
swb_test = swb_test.map(lambda x: {'transcript': x['transcript'].replace('<LAUGH>', '<')}, desc="Replacing <LAUGH> with < for CTC in Test dataset")

print("Train Dataset (70%):", swb_train)
show_random_elements(swb_train, num_examples=10)

print("Validation Dataset (10%):", swb_eval)
show_random_elements(swb_eval, num_examples=10)

print("Test Dataset (20%):", swb_test)
show_random_elements(swb_test, num_examples=10)


# ================================== PREPROCESSING ==============================================
# remove special characters
swb_train = swb_train.map(
    remove_special_characters,
    desc="Removing special characters in swb_train"
    )
swb_eval = swb_eval.map(
    remove_special_characters,
    desc="Removing special characters in swb_eval"
    )

# VOCAB MAPPING
vocab_train = swb_train.map(
    extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=swb_train.column_names,
    desc="Extracting vocab in swb_train"
    )
vocab_eval = swb_eval.map(
    extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=swb_eval.column_names,
    desc="Extracting vocab in swb_eval"
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
with open("vocab_b32.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)

#================================================================================================


# CONFIGURE MODEL COMPONENTS
tokenizer = Wav2Vec2CTCTokenizer("./vocab_b32.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Save the tokenizer
print("Saving Wav2Vec2 tokenizer and feature extractor...")
tokenizer.save_pretrained("../fine-tuned/wav2vec2-batch32/tokenizer")
feature_extractor.save_pretrained("../fine-tuned/wav2vec2-batch32/feature_extractor")

# =============================== PROCESSING DATASET =====================================
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcript"]).input_ids
    
    return batch

swb_train = swb_train.map(
    prepare_dataset, 
    remove_columns=swb_train.column_names, 
    # batch_size=8, num_proc=4
    desc="Preparing dataset for swb_train"
)
swb_eval = swb_eval.map(
    prepare_dataset,
    remove_columns=swb_eval.column_names,
    # batch_size=8, num_proc=4
    desc="Preparing dataset for swb_eval"
)

# ================================== METRICS ===========================================
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)

    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = jiwer.wer(reference=label_str, hypothesis=pred_str)

    return {"wer": wer}



# ================================== MODEL ============================================
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-lv60",
    cache_dir="../ref_models/pre_trained",
    local_files_only=True,
    mask_time_prob=0.3,
    mask_feature_prob=0.3,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.freeze_feature_encoder()

# ================================== TRAINING ARGUMENTS ============================================
training_args = TrainingArguments(
    output_dir="../fine-tuned/wav2vec2-batch32/",
    group_by_length=True,
    per_device_train_batch_size=32, #16
    gradient_accumulation_steps=2, 
    evaluation_strategy="steps",
    num_train_epochs=50,

    gradient_checkpointing=True,
    fp16=True,
    adam_beta2=0.98,

    save_steps=100, #50
    eval_steps=100, #50
    logging_steps=50, #50

    learning_rate=1e-4, #7e-5
    weight_decay=0.005,
    warmup_ratio=0.1,
    
    save_total_limit=2,
    
    torch_empty_cache_steps=100, # Force garbage collection if necessary
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=swb_train,
    eval_dataset=swb_eval,
    tokenizer=processor.feature_extractor,
)

# ================================== TRAINING ============================================

#====================================
#   MONITOR GPU MEMORY
#====================================

def cleanup_workers():
    """Cleanup function for workers"""
    print("Cleaning CUDA memory cache...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

try:
    print("Training the model...")
    trainer.train()

except Exception as e:
    print(f"Error in training: {e}")
    cleanup_workers() #clear the CUDA memory cache
finally:
    log_history = trainer.state.log_history
    # save the log history to txt file
    with open(os.path.join("../logs/wav2vec2", "log_history_batch32.txt"), "w") as f:
        for entry in log_history:
            f.write(str(entry) + "\n")
    cleanup_workers() #clear the CUDA memory cache
trainer.train()

trainer.save_model("../fine-tuned/wav2vec2-batch32/wav2vec2-large-lv60-speechlaugh-swb")
# trainer.push_in_progress = None
# trainer.push_to_hub("wav2vec2-large-lv60-speechlaugh-swb")