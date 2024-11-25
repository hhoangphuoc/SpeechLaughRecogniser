import multiprocessing
import os
import torch
import torch.multiprocessing as mp
import gc

import jiwer

from .text_utils import clean_transcript_sentence, transform_number_words
from .metrics import track_laugh_word_alignments, load_metrics

#=================================================================================================      
def init_multiprocessing():
    """Initialize multiprocessing settings"""
    # Set start method
    mp.set_start_method('spawn', force=True)
    
    # Get number of available CPUs from SLURM
    n_cores = os.environ.get('SLURM_CPUS_PER_TASK')
    print(f"Number of available CPUs: {n_cores}")
    if n_cores is not None:
        n_cores = int(n_cores)
    else:
        n_cores = max(1, multiprocessing.cpu_count() // 2)
    
    print(f"Using multiprocessing workers: {n_cores}")

    # Set the number of threads for PyTorch
    torch.set_num_threads(n_cores)
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    print(f"PyTorch threads: {torch.get_num_threads()}")
    
    return n_cores

def save_model_components(
        model, 
        tokenizer, 
        generation_config,  
        output_dir
    ):
    """
    Save all model components to the specified directory
    """
    # Save main components
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    generation_config.save_pretrained(output_dir)
    
    print("Successfully saved model components to: ", output_dir)

#=================================================================================================  

def prepare_dataset_with_a40(
        dataset,
        feature_extractor,
        tokenizer,
        num_proc=8, #default number of processes to multiprocessing

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
            print(f"Type of input_features example: {type(sample['input_features'])}")
            print(f"Type of labels example: {type(sample['labels'])}")

            return processed_dataset.with_format("torch", device="cpu")
            
        except Exception as e:
            print(f"Dataset processing error: {str(e)}")
            # Fallback to single process if multiprocessing fails
            print("Falling back to single process...")
            raise

#=================================================================================================  
def gpu_config():
    """
    GPU configuration for A40 GPU. Added to faster training, in details:
    - CuDNN enabled
    - CuDNN allow_tf32 enabled
    - CuDNN matmul allow_tf32 enabled
    """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    #===================================================================    

#=================================================================================================      
#===============================================================================================
#                       COMPUTE METRICS WITH MemoryEfficientTrainer
#===============================================================================================
def compute_metrics(
        eval_pred, 
        tokenizer,
        eval_loss=None,
    ):
    """Memory-efficient metric computation"""
    try:
        print("\nComputing metrics...")
        
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        # Load metrics
        wer_metric, f1_metric = load_metrics()

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
    
#=================================================================================================          


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
