import argparse
import numpy as np
import os
import warnings
import torch
import multiprocessing
import torch.multiprocessing as mp
import gc
from dotenv import load_dotenv

# For Fine-tuned Model--------------------------------
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig
from transformers.trainer_callback import EarlyStoppingCallback
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel   #for distributed training
from datasets import load_from_disk
from huggingface_hub import login
#-----------------------------------------------------------------

# Custom Modules
from modules import (
    DataCollatorSpeechSeq2SeqWithPadding,
    MemoryEfficientCallback,
    MetricsCallback,
    MultiprocessingCallback,
    MemoryEfficientTrainer,
)
from preprocess import (
    split_dataset, 
    transform_number_words, 
    clean_transcript_sentence
)
from utils import track_laugh_word_alignments

# Evaluation Metrics------
import pandas as pd
import evaluate
import jiwer
#====================================================================================================================================================


#===================================================================
# REMOVE TF WARNINGS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)

#===================================================================

"""
This is the fine-tuning Whisper model 
for the specific task of transcribing recognizing speech laughter, fillers, 
pauses, and other non-speech sounds in conversations.
"""
#==========================================================================================================
#   Initialise Multiprocessing
#==========================================================================================================


#==========================================================================================================
#           HELPER FUNCTIONS TO LOAD AND SAVE MODEL COMPONENTS
#========================================================================================================== 
def initialize_model_config(model_path, cache_dir):
    """
    Initialize the model configuration specified for A40 GPU
    """
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, 
        cache_dir=cache_dir,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    #===================================================================
    # CuDNN configuration for A40 GPU
    #===================================================================
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    #===================================================================
    model.config.use_cache = False
    

    return model

def save_model_components(
        model, 
        tokenizer, 
        generation_config,  
        output_dir
    ):
    """Save all model components to the specified directory"""
    # Save main components
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    generation_config.save_pretrained(output_dir)
    
    print("Successfully saved model components to: ", output_dir)

def setup_tokenizer_and_generation_config(
        model_path="openai/whisper-small", 
        cache_dir="./ref_models/pre_trained/"
    ):
    
    tokenizer = WhisperTokenizer.from_pretrained(
        model_path, 
        cache_dir=cache_dir
    )
    
    # 2. Add special tokens
    special_tokens = ["[LAUGHTER]", "[SPEECH_LAUGH]"]
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    
    # 3. Load and modify generation config
    generation_config = GenerationConfig.from_pretrained(
        model_path, 
        cache_dir=cache_dir
    )
    
    # 4. Update suppress_tokens to exclude your new special tokens
    new_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in special_tokens]
    current_suppress = generation_config.suppress_tokens
    
    # Remove your special token IDs from suppress_tokens if they're in there
    generation_config.suppress_tokens = [
        token_id for token_id in current_suppress 
        if token_id not in new_token_ids
    ]
    
    return tokenizer, generation_config, num_added_tokens


#==========================================================================================================
#                               SPEECH LAUGH RECOGNITION USING WHISPER
#==========================================================================================================
def SpeechLaughWhisper(args):
    """
    This function is used to recognize speech laughter, fillers, 
    pauses, and other non-speech sounds in conversations.
    Args:
        df: pandas dataframe - contains all audios and transcripts

    """
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    #==========================================================================================================
    #                                       ENABLE MULTIPROCESSING
    #==========================================================================================================
    n_proc = init_multiprocessing()

    #==========================================================================================================
    #                                           MODEL CONFIGURATION 
    #==========================================================================================================
    tokenizer, generation_config, num_added_tokens = setup_tokenizer_and_generation_config(
        args.model_path, 
        args.pretrained_model_dir)

    #=========================================
    #               Processor
    #=========================================
    processor = WhisperProcessor.from_pretrained(
        args.model_path, 
        cache_dir=args.pretrained_model_dir
    ) # processor - combination of feature extractor and tokenizer

    #=========================================
    #           Feature Extractor
    #=========================================
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        args.model_path, 
        cache_dir=args.pretrained_model_dir
    ) #feature extractor

    #=====================================================================
    #                                   MODELS
    #=====================================================================
    
    # PRE-TRAINED MODEL
    model = initialize_model_config(
        model_path=args.model_path, 
        cache_dir=args.pretrained_model_dir
    )

    # MODEL FROM CHECKPOINT
    # model = WhisperForConditionalGeneration.from_pretrained(
    #     args.model_output_dir + "fine-tuned-2000steps-oom",
    #     device_map="auto",
    #     low_cpu_mem_usage=True,
    # )

    #=====================================================================

    # UPDATE CONFIGS
    model.resize_token_embeddings(len(tokenizer))
    model.generation_config = generation_config
    model.to(device) # move model to GPUs
    #---------------------------------------------------------------------

    #=========================================
    #       DATA COLLATOR FOR PADDING
    #=========================================
    speech_laugh_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, 
        decoder_start_token_id=model.config.decoder_start_token_id,
        device=device
    )

    def prepare_dataset_with_a40(
        dataset,
        num_proc=16 #default number of processes to multiprocessing
    ):
        """
        Process dataset with A40-specific optimizations and 
        enabled with multiprocessing for faster processing
        """
        print(f"Prepare dataset: ========================================= \n{dataset}")
    
        @torch.amp.autocast("cuda")
        def process_single(example):
            """Process a single example to avoid length mismatches"""
            try:
                # Process audio
                audio_array = example['audio']['array']
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.squeeze()
                
                # Extract features
                inputs = feature_extractor(
                    audio_array,
                    sampling_rate=16000,
                    return_tensors="pt",
                ).input_features[0]
                
                # Process text
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(
                        example["transcript"],
                        return_tensors="pt",    
                    ).input_ids[0]
                
                # Remove batch dimension since we're processing single examples
                return {
                    "input_features": inputs,
                    "labels": labels
                }
                
            except Exception as e:
                print(f"Error processing example: {str(e)}")
                raise


        # Process dataset with multiprocessing
        try:
            # Process single example each process with 16 processes running in parallel
            processed_dataset = dataset.map(
                process_single,
                num_proc=num_proc,
                remove_columns=dataset.column_names,
                desc=f"Processing dataset (num_proc={num_proc})",
                load_from_cache_file=False
            )

            #==============================================================
            #                       Debug Info
            #============================================================== 
            print("\nProcessed Dataset Info:-------------------------------------------")
            sample = processed_dataset[0]
            print(f"Type of input_features: {type(sample['input_features'])}")
            print(f"Type of sample labels: {type(sample['labels'])}")

            return processed_dataset.with_format("torch", device="cpu")
            
        except Exception as e:
            print(f"Dataset processing error: {str(e)}")
            # Fallback to single process if multiprocessing fails
            print("Falling back to single process...")
            raise
        

    #=================================================================================================
    #                       COMPUTE METRICS 
    #=================================================================================================   
    wer_metric = evaluate.load("wer") #Word Error Rate between the hypothesis and the reference transcript
    f1_metric = evaluate.load("f1") #F1 score between the hypothesis and the reference transcript

    #===============================================================================================
    #                       COMPUTE METRICS WITH Seq2SeqTrainer
    #===============================================================================================    
    # def compute_metrics(pred):
    #     """
    #     This function is used to compute the metrics for the model
    #     on every evaluation step (`eval_steps`) and pass the metrics to the 
    #     `metrics_callback` for logging to tensorboard and saving to a csv file
    #     Args:
    #         pred: predictions from the model
    #     Returns:
    #         metrics: dictionary of metrics
    #     """
    #     print("Computing metrics: =========================================\n")
    #     pred_ids = pred.predictions.cpu() if isinstance(pred.predictions, torch.Tensor) else pred.predictions
    #     label_ids = pred.label_ids.cpu() if isinstance(pred.label_ids, torch.Tensor) else pred.label_ids

    #     # Replace -100 with pad_token_id
    #     label_ids[label_ids == -100] = tokenizer.pad_token_id

    #     # Decode predictions and references
    #     ref_transcripts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    #     pred_transcripts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        
    #     # Clean and transform transcripts
    #     ref_transcripts = clean_transcript_sentence(ref_transcripts)
    #     pred_transcripts = transform_number_words(pred_transcripts, reverse=True)
    #     pred_transcripts = clean_transcript_sentence(pred_transcripts)

    #     # Initialize aggregate metrics
    #     batch_metrics = {
    #         'wer': 0,
    #         'f1': 0,
    #         'lwhr': 0,  # Laugh Word Hit Rate
    #         'lthr': 0,  # Laughter Token Hit Rate
    #         'lwsr': 0,  # Laugh Word Substitution Rate
    #         'ltsr': 0,  # Laughter Token Substitution Rate
    #         'lwdr': 0,  # Laugh Word Deletion Rate
    #         'ltdr': 0,  # Laughter Token Deletion Rate
    #         'lwir': 0,  # Laugh Word Insertion Rate
    #         'ltir': 0   # Laughter Token Insertion Rate
    #     }

    #     # Calculate metrics for each transcript pair
    #     batch_size = len(ref_transcripts)
    #     for ref, hyp in zip(ref_transcripts, pred_transcripts):
    #         # Get alignments using jiwer
    #         alignments = jiwer.process_words(reference=ref, hypothesis=hyp)
            
    #         # Calculate laugh-specific metrics
    #         laugh_stats = track_laugh_word_alignments(ref, hyp, alignments)
    #         if laugh_stats is not None:     
    #             # Accumulate metrics
    #             # batch_metrics['wer'] += alignments.wer
    #             batch_metrics['lwhr'] += laugh_stats['lwhr']
    #             batch_metrics['lthr'] += laugh_stats['lthr']
    #             batch_metrics['lwsr'] += laugh_stats['lwsr']
    #             batch_metrics['ltsr'] += laugh_stats['ltsr']
    #             batch_metrics['lwdr'] += laugh_stats['lwdr']
    #             batch_metrics['ltdr'] += laugh_stats['ltdr']
    #             batch_metrics['lwir'] += laugh_stats['lwir']
    #             batch_metrics['ltir'] += laugh_stats['ltir']
    #         else:
    #             print("No laughter stats was computed!")

    #     batch_metrics['wer'] = 100 * wer_metric.compute(predictions=pred_transcripts, references=ref_transcripts)
    #     # Calculate F1 score separately (as it's already batch-computed)
    #     batch_metrics['f1'] = 100 * f1_metric.compute(predictions=pred_transcripts, references=ref_transcripts)

    #     # Average all metrics over batch
    #     for key in batch_metrics:
    #         if key != 'f1':  # Skip F1 as it's already properly computed
    #             batch_metrics[key] = (batch_metrics[key] / batch_size) * 100

    #     # Debug print
    #     print("Computed batch metrics:")
    #     for key, value in batch_metrics.items():
    #         print(f"{key}: {value:.2f}")

    #     return batch_metrics
    #===============================================================================================

    
    #===============================================================================================
    #                       COMPUTE METRICS WITH MemoryEfficientTrainer
    #===============================================================================================
    def compute_metrics(eval_pred, eval_loss=None):
        """Memory-efficient metric computation"""
        try:
            print("\nComputing metrics...")
            
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids

            # Replace -100 with pad_token_id
            # labels[labels == -100] = tokenizer.pad_token_id

            # Process in batches
            batch_size = 16
            metrics_sum = {
                'wer': 0, 'f1': 0, 'lwhr': 0, 'lthr': 0,
                'lwsr': 0, 'ltsr': 0, 'lwdr': 0, 'ltdr': 0,
                'lwir': 0, 'ltir': 0
            }
            batch_count = 0

            for i in range(0, len(predictions), batch_size):
                # Get batch
                batch_preds = predictions[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                
                # Decode
                refs = tokenizer.batch_decode(batch_labels, skip_special_tokens=True)
                hyps = tokenizer.batch_decode(batch_preds, skip_special_tokens=True)
                
                # Clean transcripts
                refs = clean_transcript_sentence(refs)
                hyps = transform_number_words(hyps, reverse=True)
                hyps = clean_transcript_sentence(hyps)
                
                # Compute batch metrics
                batch_wer = wer_metric.compute(predictions=hyps, references=refs)
                batch_f1 = f1_metric.compute(predictions=hyps, references=refs)
                
                #Compute WER and F1
                metrics_sum['wer'] += batch_wer
                metrics_sum['f1'] += batch_f1
                
                # Compute laugh metrics
                for ref, hyp in zip(refs, hyps):
                    alignments = jiwer.process_words(reference=ref, hypothesis=hyp)
                    laugh_stats = track_laugh_word_alignments(ref, hyp, alignments)
                    
                    if laugh_stats:
                        for key in laugh_stats:
                            metrics_sum[key] += laugh_stats[key]
                
                batch_count += 1

                # Periodic cleanup during the evaluation
                if i % (batch_size * 5) == 0:
                    print(f"Clean up memory during evaluation at step:{i}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                

            # Average metrics
            metrics = {
                k: (v / batch_count) * 100 
                for k, v in metrics_sum.items()
            }
            
            if eval_loss is not None:
                metrics['loss'] = eval_loss

            # Debug print
            print("\nComputed metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v:.2f}")

            return metrics
        
        except Exception as e:
            print(f"Error in compute_metrics: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise e


    #===============================================================================================
    #                           LOAD DATASET AND PROCESSING 
    #===============================================================================================
    if args.processed_as_dataset:
        # Load the dataset
        print("Loading the dataset as HuggingFace Dataset...")
        switchboard_dataset = load_from_disk(args.processed_file_path)
        # Split the dataset into train and validation
        train_dataset, test_dataset = split_dataset(
            switchboard_dataset, 
            subset_ratio=0.1, #TODO: Given subset ration < 1 to get smaller dataset for testing
            split_ratio=0.9, 
            split="both"
        )
    print("Dataset Loaded....\n")
    print(f"Train Dataset: {train_dataset}")
    print(f"Validation Dataset: {test_dataset}")
    print("------------------------------------------------------")
    
    #===============================================================================================
    #                       DATASET MAPPING TO TENSORS
    #===============================================================================================
    train_dataset = prepare_dataset_with_a40(
        train_dataset,  
        num_proc=n_proc # Use 16 CPU cores for multiprocessing
    )
    test_dataset = prepare_dataset_with_a40(
        test_dataset, 
        num_proc=n_proc # Use 16 CPU cores for multiprocessing
    )
    # Verify dataset size
    print(f"Process dataset size: {len(train_dataset)}")
    # Also verify the dataset format
    print("Dataset features:", train_dataset.features)

    # ---------------------------------------------------- end of prepare dataset --------------------------------------------


    #===============================================================================================
    #                           TRAINING CONFIGURATION 
    #===============================================================================================

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_output_dir,
        logging_dir=args.log_dir,
        per_device_train_batch_size=4,  # Reduced batch size
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=2,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=100,
        learning_rate=1e-5,
        # num_train_epochs=5,
        max_steps=5000,
        warmup_steps=500,
        fp16=True,
        tf32=True,
        dataloader_num_workers=1,        # Reduced workers
        dataloader_pin_memory=True,
        gradient_checkpointing=True,     # Enable checkpointing
        generation_max_length=225,
        predict_with_generate=True,
        include_inputs_for_metrics=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        load_best_model_at_end=True,
        # Add memory optimization flags
        optim="adamw_torch_fused",
        max_grad_norm=0.5,
    )


    #================================================================
    #                   CALLBACKS FUNCTIONS
    #================================================================
    writer = SummaryWriter(log_dir=args.log_dir)

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3, #stop training if the model is not improving
        early_stopping_threshold=0.01 #consider improve if WER decrease by 0.01
    )
    memory_callback = MemoryEfficientCallback(num_proc=n_proc)
    metrics_callback = MetricsCallback(
        output_dir=args.log_dir, # ./logs
        save_steps=1000, #save the model every 1000 steps - SAVE LESS FREQUENTLY
        model_name="speechlaugh_subset10" #model name to saved to corresponding checkpoint
    )
    multiprocessing_callback = MultiprocessingCallback(
        num_proc=n_proc
    )
    #--------------------------------------------------------------------------------------------

    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     tokenizer=tokenizer,
    #     train_dataset=train_dataset,
    #     eval_dataset=test_dataset,
    #     data_collator=speech_laugh_collator,
    #     compute_metrics=compute_metrics,
    #     callbacks=[
    #         early_stopping, 
    #         memory_callback,
    #         metrics_callback,
    #         multiprocessing_callback
    #         ],
    # )
    #=====================================================
    #FIXME: Try using MemoryEfficientTrainer
    #=====================================================  
    trainer = MemoryEfficientTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=speech_laugh_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            early_stopping,
            memory_callback,
            metrics_callback,
            multiprocessing_callback
        ],
    )
    #===============================================================================================
    #                           TRAINING 
    #===============================================================================================

    try:
        # # Add evaluation memory management
        # def evaluate_with_memory_management():
        #     torch.cuda.empty_cache()
        #     gc.collect()
            
        #     initial_memory = torch.cuda.memory_allocated() / 1e9
        #     print(f"Memory before evaluation: {initial_memory:.2f}GB")
            
        #     results = trainer.evaluate()
            
        #     torch.cuda.empty_cache()
        #     gc.collect()
            
        #     final_memory = torch.cuda.memory_allocated() / 1e9
        #     print(f"Memory after evaluation: {final_memory:.2f}GB")
            
        #     return results
            
        # # Modify trainer to use memory-managed evaluation
        # trainer.evaluate = evaluate_with_memory_management()
        
        # # Start training
        # trainer.train()

        # Start training with memory monitoring
        print("\nStarting training with memory monitoring...")
        # Initial memory check
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1e9
            print(f"Initial GPU memory: {initial_memory:.2f}GB")
        
        trainer.train()
        
        # Save the model after successful training
        #=============================================================================
        #                           SAVE MODEL TO CHECKPOINT 
        #=============================================================================

        output_dir = args.model_output_dir + "fine-tuned"
        save_model_components(
            model=model, 
            tokenizer=tokenizer, 
            generation_config=generation_config, 
            output_dir=output_dir
        )
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise e


    #-----------------------------------------end of training ------------------------------


    # PUSH MODEL TO HUB ----------------------------------------------------------
    # kwargs = {
    #     "dataset_tags": "hhoangphuoc/switchboard",
    #     "dataset": "Switchboard",
    #     # "dataset_args": "config: hi, split: test",
    #     "model_name": "Speech Laugh Whisper - Phuoc Ho",
    #     "finetuned_from": "openai/whisper-large-v2",
    #     "tasks": "automatic-speech-recognition",
    # }
    # trainer.push_to_hub(**kwargs)
    #-------------------------------------------------------------------------------------------


# MAIN FUNCTION ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Laugh Recognition")
    
    # Data Configs
    parser.add_argument("--processed_as_dataset", default=False, type=bool, required=False, help="Whether or not process as Huggingface dataset")
    parser.add_argument("--processed_file_path", default="./datasets/processed_dataset/", type=str, required=False, help="Path to the test.csv file")
    
    #FIXME: IF LOAD FROM CSV FILE, NEED THESE FILE PATH ----------------------------------
    parser.add_argument("--train_file_path", default="./datasets/train_switchboard.csv", type=str, required=False, help="Path to the train.csv file")
    parser.add_argument("--eval_file_path", default="./datasets/val_switchboard.csv", type=str, required=False, help="Path to the val.csv file")
    #-------------------------------------------------------------------------------------

    # Model Configs
    parser.add_argument("--model_path", default="openai/whisper-small", type=str, required=False, help="Select pretrained model")
    parser.add_argument("--pretrained_model_dir", default="./ref_models/pre_trained/", type=str, required=False, help="Name of the model")
    parser.add_argument("--model_output_dir", default="./vocalwhisper/speechlaugh-whisper-small/", type=str, required=False, help="Path to the output directory")
    parser.add_argument("--log_dir", default="./logs", type=str, required=False, help="Path to the log directory")
    parser.add_argument("--evaluate_dir", default="./evaluate", type=str, required=False, help="Path to the evaluation directory")
    #------------------------------------------------------------------------------
    args = parser.parse_args()

    load_dotenv()

    try:
        SpeechLaughWhisper(args)
    except OSError as error:
        print(error)

