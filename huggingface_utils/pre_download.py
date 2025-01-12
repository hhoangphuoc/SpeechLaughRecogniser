import os
import evaluate
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    # Wav2Vec2ProcessorWithLM,
    Wav2Vec2ForCTC,
)
import evaluate

if __name__ == "__main__":
    print("Downloading pre-trained model...")
    # Download the pre-trained model
    try:
        # processor = WhisperProcessor.from_pretrained(
        #     "openai/whisper-large-v2",
        # cache_dir="../ref_models/pre_trained",
        #     force_download=True,
        # )
        # model = WhisperForConditionalGeneration.from_pretrained(
        #     "openai/whisper-large-v2",
        # cache_dir="../ref_models/pre_trained",
        #     force_download=True,
        # )
        model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60", #FIXME: this model loaded for testing
            # "facebook/wav2vec2-large-lv60", # THIS MODEL USED FOR FINE-TUNING
            cache_dir="../ref_models/pre_trained",
            force_download=True,
        )
        processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60", #FIXME: this model loaded for testing
            # "facebook/wav2vec2-large-lv60", # THIS MODEL USED FOR FINE-TUNING
            cache_dir="../ref_models/pre_trained",
            force_download=True,
        )

        # wer_metrics = evaluate.load("wer", cache_dir="../ref_models/metrics")
        # f1_metrics = evaluate.load("f1", cache_dir="../ref_models/metrics")

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