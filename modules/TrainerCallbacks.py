import os
import pandas as pd
import torch
import gc
import multiprocessing
import time
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
#------------------------------------------------------------------------------------------------

#========================================================================================================================

#========================================================================================================================
#                   MULTIPROCESSING CALLBACK
#========================================================================================================================
class MultiprocessingCallback(TrainerCallback):
    def __init__(self, num_proc):
        self.num_proc = num_proc
        
    def on_train_begin(self, args, state, control, **kwargs):
        torch.set_num_threads(self.num_proc)
        print(f"\nTraining started with {self.num_proc} processes")
        print(f"CPU cores available: {multiprocessing.cpu_count()}")
        print(f"PyTorch threads: {torch.get_num_threads()}")
        
    def on_train_end(self, args, state, control, **kwargs):
        # Cleanup multiprocessing resources
        torch.set_num_threads(1)
#========================================================================================================================


#========================================================================================================================
#                   Memory Efficient Callback
#========================================================================================================================
class A40MemoryMonitor:
    def __init__(self, threshold_gb=20):  # Set high threshold for A40
        self.threshold_gb = threshold_gb
        self.peak_memory = 0
        self.allocation_threshold = 0.9 #~90% of allocated memory
        
    def check_memory(self):
        """
        Cleanup the memory with respect to certain threshold.
        Using more frequent memory check (every 50 steps) and cleanup.
        And only clean if the memory usage is above the threshold.
        """
        current_memory = torch.cuda.memory_allocated() / 1e9
        current_cached_memory = torch.cuda.memory_reserved() / 1e9
        max_memory_allocated = torch.cuda.max_memory_allocated() / 1e9

        self.peak_memory = max(self.peak_memory, current_memory)
        
        print(f"\nCurrent GPU Memory: {current_memory:.2f}GB")
        print(f"GPU Memory cached: {current_cached_memory:.2f}GB")
        print(f"Peak GPU Memory: {self.peak_memory:.2f}GB")
        
        allocated_ratio = current_memory / max_memory_allocated
        # Warning if approaching threshold
        if current_memory > self.threshold_gb:
            print(f"WARNING: High memory usage ({current_memory:.2f}GB > {self.threshold_gb}GB) - Cleaning up memory ...")
            self.soft_cleanup()
        elif allocated_ratio > self.allocation_threshold:
            print(f"CRITICAL: High memory usage ({allocated_ratio:.2f} > {self.allocation_threshold:.2f}) - Deep cleaning up memory ...")
            self.deep_cleanup()
    
    def soft_cleanup(self):
        torch.cuda.empty_cache()
        # gc.collect() #~soft cleanup don't need to call this   
    def deep_cleanup(self):
        """
        Function to clean up the memory without any threshold.
        Therefore this function will only be called when necessary,
        not every `check_memory()` call (every 50 steps), or `save_metrics()` call (every 1000 steps).
        
        This function is called when the memory monitor detects high memory usage. For example,
        when `torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()` > 90%

        """
        print("Deep cleaning up memory ...")
        torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()
            
        # Print memory stats for debugging
        print(f"After Deep Cleanup:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f}GB")
class MemoryEfficientCallback(TrainerCallback):
    """
    Callback function to manage GPU memory every `50 steps` and before evaluation.
    The memory monitor is used to check the memory usage and clean up if necessary.
    based on the certain `threshold = 30GB`, the memory monitor will clean up the memory.
    """
    def __init__(self, num_proc):
        self.memory_monitor = A40MemoryMonitor()
        self.num_proc = num_proc
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:  # Check CUDA memory every 100 steps - more frequent
            # manage_memory()
            self.memory_monitor.check_memory() #check memory every 100 steps and clean up if necessary

            #-------------------------------------------------------------------
            # ADD GRADUAL MEMORY CLEANUP
            #-------------------------------------------------------------------
            if hasattr(state, "trainer") and state.trainer is not None:
                if state.trainer.optimizer is not None:
                    state.trainer.optimizer.zero_grad(set_to_none=True)
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # manage_memory() # Check memory before and after evaluation
        """
        Clean up memory after evaluation.
        """
        self.memory_monitor.deep_cleanup()
    

    def on_save(self, args, state, control, **kwargs):
        """
        Check memory before saving the model checkpoint.
        Consider calling `deep_cleanup()` here if memory is a critical issue.
        Or call `soft_cleanup()` if memory is above soft threshold.
        """	
        self.memory_monitor.check_memory()
    
    def on_train_end(self, args, state, control, **kwargs):
        """
        Check memory before training ends.
        Consider calling `deep_cleanup()` here when training ends.
        """
        self.memory_monitor.deep_cleanup() #deep cleanup finally when training ends


#========================================================================================================================
#                   Metrics Callback with MemoryEfficientTrainer
#========================================================================================================================
class MetricsCallback(TrainerCallback):
    def __init__(self, 
                 output_dir, 
                 save_steps=1000, 
                 model_name="speechlaugh_whisper"):
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.model_name = model_name
        self.last_eval_step = 0
        
        # Initialize DataFrame with columns
        self.metrics_df = pd.DataFrame(columns=[
            "step", "epoch", "timestamp",
            "training_loss", "validation_loss",
            "wer", "f1",
            "lwhr", "lthr", "lwsr", "ltsr", 
            "lwdr", "ltdr", "lwir", "ltir"
        ])
        
        # Create tensorboard writer with memory-efficient settings
        self.writer = SummaryWriter(
            log_dir=os.path.join(output_dir, "tensorboard"),
            max_queue=10,
            flush_secs=60
        )
        os.makedirs(output_dir, exist_ok=True)
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Enhanced evaluation metrics handling"""
        if metrics is None:
            return
            
        current_step = state.global_step
        
        try:
            # Only process if we haven't recently saved at this step
            if current_step != self.last_eval_step:
                self.last_eval_step = current_step
                
                # Get training loss from the latest log history
                latest_logs = state.log_history[-1] if state.log_history else {}
                training_loss = latest_logs.get("loss", 0)
                
                # Prepare metrics dictionary
                new_metrics = {
                    "step": current_step,
                    "epoch": state.epoch,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "training_loss": training_loss,
                    "validation_loss": metrics.get("eval_loss", 0),
                }
                
                # Add evaluation metrics with proper prefix handling
                metric_mappings = {
                    "wer": "wer",
                    "f1": "f1",
                    "lwhr": "lwhr",
                    "lthr": "lthr",
                    "lwsr": "lwsr",
                    "ltsr": "ltsr",
                    "lwdr": "lwdr",
                    "ltdr": "ltdr",
                    "lwir": "lwir",
                    "ltir": "ltir"
                }
                
                # Update metrics with proper prefix handling
                for metric_key, df_key in metric_mappings.items():
                    # Check both with and without eval_ prefix
                    value = metrics.get(f"eval_{metric_key}", metrics.get(metric_key, 0))
                    new_metrics[df_key] = value

                # Debug prints with memory info
                print("\nEvaluation Step Summary:")
                print(f"Step: {current_step}, Epoch: {state.epoch:.2f}")
                if torch.cuda.is_available():
                    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                
                # Append to DataFrame
                self.metrics_df.loc[len(self.metrics_df)] = new_metrics
                
                # Log to TensorBoard with memory cleanup
                for key, value in new_metrics.items():
                    if isinstance(value, (int, float)) and key not in ["step", "epoch"]:
                        self.writer.add_scalar(f"eval/{key}", value, current_step)
                self.writer.flush()  # Force write to disk
                
                # Save metrics to CSV
                self.save_metrics()
                
                # Print metrics summary
                print("\n" + "="*50)
                print("Metrics Summary:")
                print(f"Training Loss: {new_metrics['training_loss']:.4f}")
                print(f"Validation Loss: {new_metrics['validation_loss']:.4f}")
                print(f"WER: {new_metrics['wer']:.2f}")
                print(f"F1: {new_metrics['f1']:.2f}")
                print("\nLaughter Metrics:")
                print(f"Word Hit Rate: {new_metrics['lwhr']:.2f}")
                print(f"Token Hit Rate: {new_metrics['lthr']:.2f}")
                print("="*50 + "\n")

                # Save checkpoint with memory management
                self.save_checkpoint(state, new_metrics)
                
        except Exception as e:
            print(f"Error in metrics callback: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise e

    def save_checkpoint(self, state, metrics):
        """Memory-efficient checkpoint saving"""
        try:
            checkpoint_dir = os.path.join(self.output_dir, "save_models")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_name = f"{self.model_name}_step_{state.global_step}"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            
            if hasattr(state, "trainer") and state.trainer is not None:
                # Save model with memory cleanup
                state.trainer.save_model(checkpoint_path)
                
                # Save training state efficiently
                torch.save({
                    'step': state.global_step,
                    'model_state_dict': state.trainer.model.state_dict(),
                    'optimizer_state_dict': state.trainer.optimizer.state_dict(),
                    'metrics': metrics,
                }, f"{checkpoint_path}/checkpoint.pt", 
                   _use_new_zipfile_serialization=True)
                
                print(f"\nCheckpoint saved: {checkpoint_path}")
                
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def on_train_end(self, args, state, control, **kwargs):
        """Cleanup on training end"""
        try:
            self.save_metrics()
            self.writer.flush()
            self.writer.close()
            
            if len(self.metrics_df) > 0:
                print("\nTraining Complete - Final Metrics:")
                print(f"Total Steps: {state.global_step}")
                print(f"Final Validation Loss: {self.metrics_df['validation_loss'].iloc[-1]:.4f}")
                print(f"Final WER: {self.metrics_df['wer'].iloc[-1]:.2f}")
                print(f"Final F1: {self.metrics_df['f1'].iloc[-1]:.2f}")
                print("\nFinal Laughter Metrics:")
                print(f"Word Hit Rate: {self.metrics_df['lwhr'].iloc[-1]:.2f}")
                print(f"Token Hit Rate: {self.metrics_df['lthr'].iloc[-1]:.2f}")
                
        except Exception as e:
            print(f"Error in training end cleanup: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
#========================================================================================================================   


#========================================================================================================================
#                   Metrics Callback
#========================================================================================================================
# class MetricsCallback(TrainerCallback):
#     """
#     Callback to save training and evaluation metrics to CSV every 1000 steps.
#     """
#     def __init__(self, 
#                  output_dir, 
#                  save_steps=1000, 
#                  model_name="speechlaugh_whisper"):
#         self.output_dir = output_dir
#         self.save_steps = save_steps
#         self.model_name = model_name
#         self.last_eval_step = 0
        
#         # Initialize DataFrame with columns
#         self.metrics_df = pd.DataFrame(columns=[
#             "step", "epoch", "timestamp",
#             "training_loss", "validation_loss",
#             "wer", "f1",
#             "lwhr", "lthr", "lwsr", "ltsr", 
#             "lwdr", "ltdr", "lwir", "ltir"
#         ])
        
#         # Create tensorboard writer
#         self.writer = SummaryWriter(
#             log_dir=os.path.join(output_dir, "tensorboard"))
#         os.makedirs(output_dir, exist_ok=True)
        
#     def on_evaluate(self, args, state, control, metrics=None, **kwargs):
#         """Called after evaluation - This is where we'll compute and save metrics"""
#         if metrics is None:
#             return
            
#         current_step = state.global_step
        
#         # Only process if we haven't recently saved at this step
#         if current_step != self.last_eval_step:
#             self.last_eval_step = current_step
            
#             # Get current metrics
#             new_metrics = {
#                 "step": current_step,
#                 "epoch": state.epoch,
#                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#                 "training_loss": state.log_history[-1].get("loss", 0) if state.log_history else 0,
#                 "validation_loss": metrics.get("eval_loss", 0),
                
#                 # Evaluation metrics
#                 "wer": metrics.get("wer", 0),
#                 "f1": metrics.get("f1", 0),
#                 "lwhr": metrics.get("lwhr", 0),
#                 "lthr": metrics.get("lthr", 0),
#                 "lwsr": metrics.get("lwsr", 0),
#                 "ltsr": metrics.get("ltsr", 0),
#                 "lwdr": metrics.get("lwdr", 0),
#                 "ltdr": metrics.get("ltdr", 0),
#                 "lwir": metrics.get("lwir", 0),
#                 "ltir": metrics.get("ltir", 0)
#             }

#             # Debug prints
#             print("\nEvaluation metrics received:", metrics)
#             print("Metrics being saved:", new_metrics)

#             # Append to DataFrame
#             self.metrics_df.loc[len(self.metrics_df)] = new_metrics
            
#             # Log to TensorBoard
#             for key, value in new_metrics.items():
#                 if isinstance(value, (int, float)) and key not in ["step", "epoch"]:
#                     self.writer.add_scalar(f"eval/{key}", value, current_step)
            
#             # Save metrics to CSV
#             self.save_metrics()
            
#             # Print metrics summary
#             print("\n ==============================================")
#             print(f"Metrics saved at step {current_step}")
#             print(f"Validation Loss: {new_metrics['validation_loss']:.4f}")
#             print(f"WER: {new_metrics['wer']:.2f}")
#             print(f"F1: {new_metrics['f1']:.2f}")
#             print(f"Laugh Word Hit Rate: {new_metrics['lwhr']:.2f}")
#             print(f"Laughter Token Hit Rate: {new_metrics['lthr']:.2f}")
#             print("==============================================\n")

#             # Save model checkpoint after evaluation
#             self.save_checkpoint(state, new_metrics)

#     def save_checkpoint(self, state, metrics):
#         """Save model checkpoint with current metrics"""
#         checkpoint_dir = os.path.join(self.output_dir, "save_models")
#         os.makedirs(checkpoint_dir, exist_ok=True)
        
#         checkpoint_name = f"{self.model_name}_step_{state.global_step}"
#         checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
#         if hasattr(state, "trainer") and state.trainer is not None:
#             # Save model
#             state.trainer.save_model(checkpoint_path)
            
#             # Save additional training state
#             torch.save({
#                 'step': state.global_step,
#                 'model_state_dict': state.trainer.model.state_dict(),
#                 'optimizer_state_dict': state.trainer.optimizer.state_dict(),
#                 'metrics': metrics,
#             }, f"{checkpoint_path}/checkpoint.pt")

#     def save_metrics(self):
#         """Save metrics to CSV files"""
#         metrics_dir = os.path.join(self.output_dir, "metrics")
#         os.makedirs(metrics_dir, exist_ok=True)
        
#         # Save timestamped version
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         self.metrics_df.to_csv(
#             os.path.join(metrics_dir, f"training_metrics_{timestamp}.csv"), 
#             index=False
#         )
        
#         # Save latest version
#         self.metrics_df.to_csv(
#             os.path.join(metrics_dir, "training_metrics_latest.csv"), 
#             index=False
#         )

#     def on_train_begin(self, args, state, control, **kwargs):
#         print("\nMetrics tracking started")
#         print(f"Metrics will be saved after each evaluation")
#         print(f"Output directory: {self.output_dir}/metrics/")
    
#     def on_train_end(self, args, state, control, **kwargs):
#         self.save_metrics()
#         self.writer.close()
        
#         print("\nTraining completed - Final metrics saved")
#         print(f"Total steps: {state.global_step}")
#         if len(self.metrics_df) > 0:
#             print(f"Final Validation Loss: {self.metrics_df['validation_loss'].iloc[-1]:.4f}")
#             print(f"Final WER: {self.metrics_df['wer'].iloc[-1]:.2f}")
#             print(f"Final F1: {self.metrics_df['f1'].iloc[-1]:.2f}")
#             print(f"Final Laugh Metrics:")
#             print(f"  Word Hit Rate: {self.metrics_df['lwhr'].iloc[-1]:.2f}")
#             print(f"  Token Hit Rate: {self.metrics_df['lthr'].iloc[-1]:.2f}")

