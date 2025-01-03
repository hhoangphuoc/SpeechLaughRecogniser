import os
import pandas as pd
import csv
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
    def __init__(self, threshold_gb=20):  # Set high threshold for A100
        self.threshold_gb = threshold_gb
        self.peak_memory = 0
        self.allocation_threshold = 0.9 #~90% of allocated memory
        
    def check_memory(self):
        """
        Cleanup the memory with respect to certain threshold.
        Using more frequent memory check (every 100 steps) and cleanup.
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
        
        # Print memory stats for debugging
        print(f"Checked Memory:  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB ; Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    
    def soft_cleanup(self):
        torch.cuda.empty_cache()
        gc.collect() #~soft cleanup don't need to call this   
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

class MemoryEfficientCallback(TrainerCallback):
    """
    Callback function to manage GPU memory every `50 steps` and before evaluation.
    The memory monitor is used to check the memory usage and clean up if necessary.
    based on the certain `threshold = 30GB`, the memory monitor will clean up the memory.
    """
    def __init__(self):
        self.memory_monitor = A40MemoryMonitor()
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:  # Check CUDA memory every 100 steps - more frequent
            # manage_memory()
            self.memory_monitor.check_memory() #check memory every 100 steps and clean up if necessary

            #-------------------------------------------------------------------
            # ADD GRADUAL MEMORY CLEANUP
            #-------------------------------------------------------------------
            if hasattr(state, "optimizer") and state.optimizer is not None:
                state.optimizer.zero_grad(set_to_none=True)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.memory_monitor.check_memory()

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


class MetricsCallback(TrainerCallback):
    def __init__(self, file_path):
        self.file_path = file_path
        if not os.path.exists(file_path):
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['step', 'epoch', 'training_loss', 'validation_loss', 'wer', 'f1'])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print(f"Metrics callback: {metrics}")
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                state.global_step, #step
                state.epoch, #epoch
                state.log_history[-1].get('loss',0), #training loss
                metrics.get('eval_loss', ''), #validation loss
                metrics.get('eval_wer', ''), #wer
                metrics.get('eval_f1', ''), #f1
            ])


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
#         self.last_save_step = 0
        
#         # Initialize DataFrame with columns
#         self.metrics_df = pd.DataFrame(columns=[
#             # training metrics
#             "step", 
#             "epoch", 
#             "timestamp",
#             "training_loss", 
#             "validation_loss",

#             # general word-level metrics
#             "wer",
#             "f1",
#             # "th",
#             # "ts",
#             # "td",
#             # "ti",
#         ])
        
        
#         # Create tensorboard writer
#         self.writer = SummaryWriter(
#             log_dir=os.path.join(output_dir, "tensorboard"))
        
#         # Ensure output directory exists
#         os.makedirs(output_dir, exist_ok=True)
#         self.current_eval_metrics = {}  # Add this to store latest evaluation metrics
    
#     def on_evaluate(self, args, state, control, metrics=None, **kwargs):
#         """Called after evaluation."""
#         if metrics is None:
#             return
#         print(f"Evaluation metrics: {metrics}")

#         new_metrics = {
#             "step": state.global_step,
#             "epoch": state.epoch,
#             "training_loss": state.log_history[-1].get('loss',0),
#             # "validation_loss": state.eval_loss,
#             "timestamp": time.strftime("%Y%m%d_%H%M%S"),
#             "wer": metrics.get('eval_wer',0),
#             "f1": metrics.get('eval_f1',0),
#             # "th": metrics.get('eval_th',0),
#             # "ts": metrics.get('eval_ts',0),
#             # "td": metrics.get('eval_td',0),
#             # "ti": metrics.get('eval_ti',0),
#         }

#         self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([new_metrics])], ignore_index=True)

#         # Store the evaluation metrics
#         self.current_eval_metrics = metrics

#         # save evaluation metrics to csv
#         self.save_metrics(metrics)
        
#         # Update TensorBoard with evaluation metrics
#         for key, value in metrics.items():
#             if isinstance(value, (int, float)):
#                 self.writer.add_scalar(f"eval/{key}", value, state.global_step)
        
    # def on_step_end(self, args, state, control, **kwargs):
    #     """
    #     Called at the end of each step.
    #     and save metrics every save_steps (1000 steps)
    #     and save model checkpoint every save_steps (1000 steps)
        
    #     """
    #     current_step = state.global_step
        
    #     if current_step % self.save_steps == 0 and current_step != self.last_save_step:
    #         print(f"Saving model checkpoint at step {current_step}")
    #         self.last_save_step = current_step
    #         #=======================================================================================
    #         #  SAVE MODEL CHECKPOINTS EVERY SAVE_STEPS (1000 STEPS)
    #         #=======================================================================================
            
    #         # Add model saving with custom name
    #         checkpoint_dir = os.path.join(self.output_dir, "save_models")
    #         os.makedirs(checkpoint_dir, exist_ok=True)
            
    #         checkpoint_name = f"{self.model_name}_step_{current_step}"
    #         checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            
    #         # Save model checkpoint
    #         if hasattr(state, "trainer") and state.trainer is not None:
    #             state.trainer.save_model(checkpoint_path) #~trainer.save_model()
                
    #             # Optionally save optimizer and scheduler
    #             torch.save({
    #                 'step': current_step,
    #                 'model_state_dict': state.trainer.model.state_dict(),
    #                 'optimizer_state_dict': state.trainer.optimizer.state_dict(),
    #             }, f"{checkpoint_path}/checkpoint.pt")

    # #===============================================================================================


    # #===============================================================================================
    # #               WRITE TO TENSORBOARD 
    # #===============================================================================================
    # def _log_to_tensorboard(self, metrics, step):
    #     """Helper method to log metrics to TensorBoard."""
    #     for key, value in metrics.items():
    #         if isinstance(value, (int, float)) and key != "step" and key != "epoch":
    #             self.writer.add_scalar(f"train/{key}", value, step)

    # #--------------------------------------------------------------------------------------------

    # def save_metrics(self):
    #     """Save metrics to CSV file with timestamp in filename."""
    #     # Create metrics directory if it doesn't exist
    #     metrics_dir = os.path.join(self.output_dir, "metrics")
    #     os.makedirs(metrics_dir, exist_ok=True)
        
    #     # Save with timestamp
    #     timestamp = time.strftime("%Y%m%d_%H%M%S")
    #     csv_path = os.path.join(metrics_dir, f"training_metrics_{timestamp}.csv")
        
    #     # Save current metrics
    #     self.metrics_df.to_csv(csv_path, index=False)
        
    #     # Also save to a fixed filename for easy access to latest metrics
    #     latest_csv_path = os.path.join(metrics_dir, "training_metrics_latest.csv")
    #     self.metrics_df.to_csv(latest_csv_path, index=False)
    # #--------------------------------------------------------------------------------------------



    # #===============================================================================================
    # #               SAVE FINAL METRICS AND CLOSE TENSORBOARD WRITER
    # #===============================================================================================
    # def on_train_begin(self, args, state, control, **kwargs):
    #     """Called at the beginning of training."""
    #     print("\nMetrics tracking started.")
    #     print(f"Saving metrics every {self.save_steps} steps")
    #     print(f"Metrics will be saved to: {self.output_dir}/metrics/") # ./logs/metrics/
    
    # def on_train_end(self, args, state, control, **kwargs):
    #     """Called at the end of training."""
    #     # Save final metrics
    #     self.save_metrics()
        
    #     # Close TensorBoard writer
    #     self.writer.close()
        
    #     print("\nTraining completed. Final metrics saved -----------------------------------")
    #     print(f"Total steps: {state.global_step}")
    #     print(f"Final WER: {self.metrics_df['wer'].iloc[-1]:.2f}")
    #     print(f"Final F1: {self.metrics_df['f1'].iloc[-1]:.2f}")
    #     # print(f"Final Laugh Word Hit Rate: {self.metrics_df['lwhr'].iloc[-1]:.2f}")
    #     # print(f"Final Laughter Token Hit Rate: {self.metrics_df['lthr'].iloc[-1]:.2f}")
