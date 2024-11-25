import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2FeatureExtractor,
    WavLMForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    AutoTokenizer,
    AutoProcessor
)
from datasets import load_dataset
import evaluate
import multiprocessing
import numpy as np
import argparse

# Initialize multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# Memory callback to clear CUDA cache
class MemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            torch.cuda.empty_cache()

def SpeechLaughWavLM(args):
    """
    Fine-tune WavLM model for speech laugh and laughter recognition
    """
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Model Configuration
    processor = AutoProcessor.from_pretrained(
        "microsoft/wavlm-base-plus", 
        cache_dir=args.pretrained_model_dir
    )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "microsoft/wavlm-base-plus",
        cache_dir=args.pretrained_model_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/wavlm-base-plus",
        cache_dir=args.pretrained_model_dir
    )

    # Add special tokens for laughter
    special_tokens = ["[LAUGHTER]", "[SPEECH_LAUGH]"]
    tokenizer.add_tokens(special_tokens)

    # Load pre-trained model
    model = WavLMForSequenceClassification.from_pretrained(
        "microsoft/wavlm-base-plus",
        cache_dir=args.pretrained_model_dir,
        num_labels=len(special_tokens) + 1  # +1 for regular speech
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Clear GPU cache
    torch.cuda.empty_cache()
    model.config.use_cache = False

    # Prepare dataset
    def prepare_dataset(batch):
        audio = batch["audio"]
        transcript = batch["transcript"]

        # Process audio
        input_features = feature_extractor(
            raw_speech=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]

        # Process transcript
        labels = tokenizer(
            transcript,
            return_tensors="pt"
        ).input_ids

        batch["input_features"] = input_features
        batch["labels"] = labels

        return batch

    # Load and prepare dataset
    dataset = load_dataset(args.dataset_name)
    processed_dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names,
        num_proc=args.num_proc
    )

    # Evaluation metrics
    wer_metric = evaluate.load("wer", cache_dir=args.evaluate_dir)
    f1_metric = evaluate.load("f1", cache_dir=args.evaluate_dir)

    def compute_metrics(pred):
        print("Computing Metrics...")
        
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Decode predictions and references
        pred_transcripts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref_transcripts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Calculate metrics
        wer = 100 * wer_metric.compute(predictions=pred_transcripts, references=ref_transcripts)
        f1 = 100 * f1_metric.compute(predictions=pred_transcripts, references=ref_transcripts)

        return {
            "wer": wer,
            "f1": f1
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        num_train_epochs=args.num_epochs,
        fp16=True,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        save_total_limit=3,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[MemoryCallback]
    )

    # Train model
    trainer.train()

    return model, processor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="switchboard")
    args = parser.parse_args()
    model, processor = SpeechLaughWavLM(args)