import os
import argparse
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration
import evaluate

# pretrained_model_dir = "./ref_models/pre_trained"  # Specify your desired cache directory
# evaluate_dir = "./evaluate"



# # Download the pre-trained model
# processor = WhisperProcessor.from_pretrained("openai/whisper-small", cache_dir=pretrained_model_dir) 
# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", cache_dir=pretrained_model_dir) #feature extractor
# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", cache_dir=pretrained_model_dir) #tokenizer
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", cache_dir=pretrained_model_dir)
# print(f"Model downloaded to: {pretrained_model_dir}")


# # Download the evaluation metrics
# evaluate.load("wer", cache_dir=evaluate_dir)
# evaluate.load("f1", cache_dir=evaluate_dir)
# evaluate.load("exact_match", cache_dir=evaluate_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretraining Download")

    parser.add_argument("--pretrained_model_dir", default="./ref_models/pre_trained/", type=str, required=False, help="Name of the model")
    parser.add_argument("--evaluate_dir", default="./evaluate", type=str, required=False, help="Path to the evaluation directory")
    parser.add_argument("--model_path", default="openai/whisper-small", type=str, required=False, help="Specify pretrained model")

    args = parser.parse_args()
    #-----------------------------------------PRE-DOWNLOADING-----------------------------------------

    if not os.path.exists(args.pretrained_model_dir):
        os.makedirs(args.pretrained_model_dir)
    if not os.path.exists(args.evaluate_dir):
        os.makedirs(args.evaluate_dir)
    print("Downloading pre-trained model...")
    # Download the pre-trained model
    processor = WhisperProcessor.from_pretrained(args.model_path, cache_dir=args.pretrained_model_dir)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_path, cache_dir=args.pretrained_model_dir) #feature extractor
    tokenizer = WhisperTokenizer.from_pretrained(args.model_path, cache_dir=args.pretrained_model_dir) #tokenizer
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path, cache_dir=args.pretrained_model_dir)
    print(f"Model downloaded to: {args.pretrained_model_dir}")

    print("Downloading evaluation metrics...")
    # Download the evaluation metrics
    evaluate.load("wer", cache_dir=args.evaluate_dir)
    evaluate.load("f1", cache_dir=args.evaluate_dir)
    evaluate.load("exact_match", cache_dir=args.evaluate_dir)

    print("Pre-downloading completed.")
