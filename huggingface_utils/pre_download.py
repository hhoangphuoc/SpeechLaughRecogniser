import os
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
from huggingface_hub import HfApi, HfFolder, create_repo
import evaluate

def push_model_to_hub(model_type, model_name, model_path, token=None, private=True):
    """
    Push a model to Huggingface Hub

    Args:
        model_type (str): Type of model that being pushed to Huggingface Hub (either "whisper" or "wav2vec2")
        model_name (str): Name of the model checkpoint (e.g. "wav2vec2-speechlaugh-recogniser").  This  name will be the name to be displayed on Huggingface
        model_path (str): Actual path to the (finetuned) model. This will be the checkpoint dir (e.g. `fine-tuned/wav2vec2/checkpoint-id`)
        token (str): Huggingface API token
        private (bool): Whether the model is private or not
    """
    # Push the model to Huggingface Hub
    if model_type == "whisper":
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
    elif model_type == "wav2vec2":
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
    
    repo_id = f"hhoangphuoc/{model_name}"
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            token=token
        )
        print(f"Created new dataset repository: {repo_id}")
    except Exception as e:
        print(f"Repository {repo_id} already exists or error occurred: {e}")
    
    try:
        print(f"Pushing model to Huggingface Hub: {model_name} to  repository {repo_id}")
        model.push_to_hub(
            repo_id=repo_id,
            # use_temp_dir=True,
            private=private,
            token=token,
        )
    except Exception as e:
        print(f"Error pushing model to Huggingface Hub: {e}")
        exit(1)
    

if __name__ == "__main__":

#================================================================================================
#                     DOWNLOAD PRE-TRAINED MODEL TO LOCAL DIRECTORY
#================================================================================================
    # print("Downloading pre-trained model...")
    # # Download the pre-trained model
    # try:
    #     # processor = WhisperProcessor.from_pretrained(
    #     #     "openai/whisper-large-v2",
    #     # cache_dir="../ref_models/pre_trained",
    #     #     force_download=True,
    #     # )
    #     # model = WhisperForConditionalGeneration.from_pretrained(
    #     #     "openai/whisper-large-v2",
    #     # cache_dir="../ref_models/pre_trained",
    #     #     force_download=True,
    #     # )
    #     model = Wav2Vec2ForCTC.from_pretrained(
    #         "facebook/wav2vec2-large-960h-lv60", #FIXME: this model loaded for testing
    #         # "facebook/wav2vec2-large-lv60", # THIS MODEL USED FOR FINE-TUNING
    #         cache_dir="../ref_models/pre_trained",
    #         force_download=True,
    #     )
    #     processor = Wav2Vec2Processor.from_pretrained(
    #         "facebook/wav2vec2-large-960h-lv60", #FIXME: this model loaded for testing
    #         # "facebook/wav2vec2-large-lv60", # THIS MODEL USED FOR FINE-TUNING
    #         cache_dir="../ref_models/pre_trained",
    #         force_download=True,
    #     )

    #     # wer_metrics = evaluate.load("wer", cache_dir="../ref_models/metrics")
    #     # f1_metrics = evaluate.load("f1", cache_dir="../ref_models/metrics")

    # except Exception as e:
    #     print(f"Error downloading pre-trained model: {e}")
    #     exit(1)
    # print("Pre-downloading completed!")
#================================================================================================


#================================================================================================
#                           PUSH A MODEL TO HUGGINGFACE HUB
#================================================================================================
    print("Pushing a model to Huggingface Hub...")
    try:
        push_model_to_hub(
            model_type="wav2vec2",
            model_name="wav2vec2-nolaugh-7epochs",
            model_path="../fine-tuned/wav2vec2-nolaugh/checkpoint-20100",
            private=True
        )
    except Exception as e:
        print(f"Error pushing a model to Huggingface Hub: {e}")
        exit(1)
    print("Model pushed to Huggingface Hub!")
#================================================================================================