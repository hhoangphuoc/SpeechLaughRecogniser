import os
import evaluate
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    

)

if __name__ == "__main__":
    print("Downloading pre-trained model...")
    # Download the pre-trained model
    try:
        processor = WhisperProcessor.from_pretrained(
            "openai/whisper-large-v2",
        cache_dir="../ref_models/pre_trained",
            force_download=True,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v2",
        cache_dir="../ref_models/pre_trained",
            force_download=True,
        )
    except Exception as e:
        print(f"Error downloading pre-trained model: {e}")
        exit(1)
    print("Pre-downloading completed!")

    # print("Downloading evaluation metrics...")
    # # Get the cache directory
    # cache_dir = evaluate.get_cache_dir()
    # print(f"Evaluation metrics cache directory: {cache_dir}")

    # # Download the evaluation metrics
    # evaluate.load("wer")
    # evaluate.load("f1")
    # evaluate.load("exact_match")

    # # List cached metrics
    # cached_metrics = os.listdir(cache_dir)
    # print("\nCached evaluation metrics:")
    # for metric in cached_metrics:
    #     print(f"- {metric}")

    # print("\nPre-downloading completed.")
    # Get the current cache directory
    # current_cache = HfFolder.get_cache_dir()
    # print(f"Current cache directory: {current_cache}")