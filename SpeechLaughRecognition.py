import argparse
# import torchaudio
import librosa
import soundfile as sf
import pandas as pd
import numpy as np

# For dataset processing and caching--------------------------------
from datasets.fingerprint import Hasher
from joblib import Memory
#----------------------------------------------------------

# For Fine-tuned Model--------------------------------
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
#import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from modules.SpeechLaughDataCollator import DataCollatorSpeechSeq2SeqWithPadding
#----------------------------------------------------------

from utils.preprocess import process_dataset
import evaluate
#----------------------------------------------------------

"""
This is the fine-tuning Whisper model 
for the specific task of transcribing recognizing speech laughter, fillers, 
pauses, and other non-speech sounds in conversations.
"""
# notebook_login()

metric = evaluate.load("wer")

# Set up a cache directory for caching the processed datasets
# cache_dir = "./cache_dir"
# memory = Memory(cache_dir, verbose=0)

def SpeechLaughWhisper(args):
    """
    This function is used to recognize speech laughter, fillers, 
    pauses, and other non-speech sounds in conversations.
    Args:
        df: pandas dataframe - contains all audios and transcripts

    """
    # MODEL CONFIGS
    #----------------------------------------------------------
    #Processor and Tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_path) #feature extractor
    processor = WhisperProcessor.from_pretrained(args.model_path) # processor - combination of feature extractor and tokenizer
    tokenizer = WhisperTokenizer.from_pretrained(args.model_path) #tokenizer
    # special_tokens = ["[LAUGHTER]", "[COUGH]", "[SNEEZE]", "[THROAT-CLEARING]", "[SIGH]", "[SNIFF]", "[UH]", "[UM]", "[MM]", "[YEAH]", "[MM-HMM]"]
    special_tokens = ["[LAUGHTER]", "[SPEECH_LAUGH]", "[VOCALIZE-NOISE]", "[UH]", "[UM]", "[MM]", "[YEAH]", "[UM-HUM]", "[UH-HUH]", "[OH]"]
    tokenizer.add_tokens(special_tokens)

    # Load the fine-tuned Whisper model
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))

    model.generation_config.forced_decoder_ids = None

    #----------------------------------------------------------
    #Data collator for random noise
    speech_laugh_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, 
        decoder_start_token_id=model.config.decoder_start_token_id,
        padding=True,
    )

    #----------------------------------------------------------
    # @memory.cache
    # def cached_feature_extractor(*args, **kwargs):
    #     return feature_extractor(*args, **kwargs)

    #----------------------------------------------------------
    def prepare_dataset(examples):
        """
        Batched Dataset Format
        Examples:  {'audio': [
        {'path': '/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data/audio_segments/sw03111A_196534375_200339875.wav', 'array': array([-0.00018135, -0.00039364, -0.00035248, ..., -0.00202596,
        0.00115121,  0.00038809]), 'sampling_rate': 16000}, 
        {'path': '/deepstore/datasets/hmi/speechlaugh-corpus/switchboard_data/audio_segments/sw03111A_200339875_204145375.wav', 'array': array([-0.00018135, -0.00039364, -0.00035248, ..., -0.00202596,
        0.00115121,  0.00038809]), 'sampling_rate': 16000},
        ...
        ],
        'transcript': [
        'there is [A] lot in the society where things have changed', 
        "now [I] guess that's that that's become less so [I] noted that [UM] [UH] the last couple times that [I] had to go by jury they've actually selected intelligent jurors"
        ...
        ]}

        RETURN:
        Batched Dataset Format
        Examples: {
        'input_features': [
        array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ...,
        ...),
        array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ...,
        ...),
        ],
        'labels': [
        array([  0,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,..),
        array([  0,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,...),
        ]}

        Single Example format:
        {
            "audio": {
                "path": "path/to/audio",
                "array": [1, 2, 3, ...]
            },
            "sampling_rate": 16000,
            "transcript": "..."
        }

        RETURN:
        Example format: {
            "input_features": [1, 2, 3, ...],
            "labels": [1, 2, 3, ...]
        }
        """
        # #vectorise the audio arrays and sampling rates
        # audio_arrays = [example["array"] for example in examples["audio"]]
        # # sampling_rates = [example["sampling_rate"] for example in examples["audio"]]
        

        # # Convert audio arrays to NumPy arrays
        # audio_arrays = [np.array(audio) if not isinstance(audio, np.ndarray) else audio for audio in audio_arrays]

        # # or squeeze if larger dimensions
        # audio_arrays = [audio.squeeze() if audio.ndim > 1 else audio for audio in audio_arrays]

        # # Batch feature extraction with caching
        # input_features = feature_extractor(
        #     audio_arrays, sampling_rate=16000, return_tensors="pt"
        # ).input_features

        # # Batch tokenization
        # labels = tokenizer(examples["transcript"], return_tensors="pt").input_ids

        # # Create a new batch with processed data


        batch = {
            "input_features": [],
            "labels": []
        }
        for i in range(len(examples)):
            example = {}

            example_audio = examples["audio"][i]
            example_transcript = examples["transcript"][i]

            audio = example_audio["array"] 
            if type(audio) is not np.ndarray:
                audio = np.array(audio)
            if audio.ndim > 1:
                audio = audio.squeeze()
            
            sampling_rate = example_audio["sampling_rate"]

            example["input_features"] = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features[0].numpy()
            example["labels"] = tokenizer(example_transcript, return_tensors="pt").input_ids

            batch["input_features"].append(example["input_features"])
            batch["labels"].append(example["labels"])
        return batch
        
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    #----------------------------------------------------------
    train_dataset = process_dataset(args.input_file_path)
    print("Train dataset: ", train_dataset)
    train_dataset = train_dataset.map(
        prepare_dataset, 
        remove_columns=train_dataset.column_names,
        # num_proc=2,
        batched=True,
        batch_size=8,
        load_from_cache_file=True
        )
    # save the processed dataset
    train_dataset.save_to_disk("./datasets/processed_dataset/train")

    #----------------------------------------------------------
    eval_dataset = process_dataset(args.eval_file_path)
    print("Eval dataset: ", eval_dataset)
    eval_dataset = eval_dataset.map(
        prepare_dataset, 
        remove_columns=eval_dataset.column_names,
        # num_proc=2,
        batched=True,
        batch_size=8,
        load_from_cache_file=True
        )
    eval_dataset.save_to_disk("./datasets/processed_dataset/eval")
    #----------------------------------------------------------

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_output_dir,
        per_device_train_batch_size=args.batch_size, #4 - default batch size = 4, could add up to 16 based on the GPU max memory
        gradient_accumulation_steps=args.grad_steps, #increase the batch size by accumulating gradients (add 8 per cycle), could change to 2
        learning_rate=args.lr, #1e-5
        num_train_epochs=args.num_train_epochs, #default = 3 - change between 2 -5 based on overfitting
        warmup_steps=args.warmup_steps, #800
        max_steps=args.max_steps, #5000
        logging_dir=args.log_dir,
        gradient_checkpointing=True,
        fp16=True, #use mixed precision training
        evaluation_strategy="steps",
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers, #default = 16 - can change to 24 if the GPU has less memory, now compatible with 72 cores GPU
        logging_steps=25,
        save_steps=1000,
        eval_steps=1000,
        report_to=["tensorboard"], #enable tensorboard for logging
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    writer = SummaryWriter(log_dir=args.log_dir)

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3, #stop training if the model is not improving
        early_stopping_threshold=0.01 #consider improve if WER decrease by 0.01
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        # eval_dataset=dataset, #TODO- create validation dataset for evaluation instead
        eval_dataset=eval_dataset,
        data_collator=speech_laugh_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )
    
    # trainer.train()

    # Training loop with TensorBoard logging
    for epoch in range(int(training_args.num_train_epochs)):
        trainer.train()
        # Log metrics to TensorBoard after each epoch
        metrics = trainer.evaluate()
        for key, value in metrics.items():
            writer.add_scalar(f"eval/{key}", value, epoch)
    writer.close()  # Close the TensorBoard writer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Speech Laugh Recognition")
    parser.add_argument("--input_file_path", default="./datasets/train.csv", type=str, required=False, help="Path to the train.csv file")
    parser.add_argument("--eval_file_path", default="./datasets/val.csv", type=str, required=False, help="Path to the val.csv file")
    parser.add_argument("--model_path", default="openai/whisper-small", type=str, required=False, help="Select pretrained model")
    parser.add_argument("--model_output_dir", default="./vocalwhisper/vocalspeech-whisper-small", type=str, required=False, help="Path to the output directory")
    parser.add_argument("--log_dir", default="./log", type=str, required=False, help="Path to the log directory")
    parser.add_argument("--batch_size", default=2, type=int, required=False, help="Batch size for training")
    parser.add_argument("--grad_steps", default=8, type=int, required=False, help="Number of gradient accumulation steps, which increase the batch size without extend the memory usage")
    parser.add_argument("--num_train_epochs", default=3, type=int, required=False, help="Number of training epochs")
    parser.add_argument("--num_workers", default=16, type=int, required=False, help="number of workers to use for data loading, can change based on the number of cores")
    parser.add_argument("--warmup_steps", default=800, type=int, required=False, help="Number of warmup steps")
    parser.add_argument("--max_steps", default=5000, type=int, required=False, help="Maximum number of training steps")
    parser.add_argument("--lr", default=1e-5, type=float, required=False, help="Learning rate for training")
    #----------------------------------------------------------
    args = parser.parse_args()

    result = SpeechLaughWhisper(args)
