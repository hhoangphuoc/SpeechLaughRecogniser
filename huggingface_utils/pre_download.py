import os
import evaluate
from pathlib import Path
from huggingface_hub import HfFolder
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Pretraining Download")

    # parser.add_argument("--pretrained_model_dir", default="./ref_models/pre_trained/", type=str, required=False, help="Name of the model")
    # parser.add_argument("--evaluate_dir", default="./evaluate", type=str, required=False, help="Path to the evaluation directory")
    # parser.add_argument("--model_path", default="openai/whisper-small", type=str, required=False, help="Specify pretrained model")

    # args = parser.parse_args()
    # #-----------------------------------------PRE-DOWNLOADING-----------------------------------------

    # if not os.path.exists(args.pretrained_model_dir):
    #     os.makedirs(args.pretrained_model_dir)
    # if not os.path.exists(args.evaluate_dir):
    #     os.makedirs(args.evaluate_dir)
    print("Downloading pre-trained model...")
    # Download the pre-trained model
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-large",
        force_download=True,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large",
        force_download=True,
    )
    print("Pre-downloading completed.")

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