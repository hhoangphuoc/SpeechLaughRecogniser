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
#                   Memory Efficient Callback
#========================================================================================================================
class A40MemoryMonitor:
    def __init__(self, threshold_gb=30):  # Set high threshold for A100
        self.threshold_gb = threshold_gb
        self.peak_memory = 0
        self.allocation_threshold = 0.8 #~80% of allocated memory
        
    def check_memory(self):
        """
        Cleanup the memory with respect to certain threshold.
        Using more frequent memory check (every 100 steps) and cleanup.
        And only clean if the memory usage is above the threshold.
        """
        allocated_memory = torch.cuda.memory_allocated() / 1e9
        max_memory_allocated = torch.cuda.max_memory_allocated() / 1e9

        self.peak_memory = max(self.peak_memory, allocated_memory)
        
        # Print memory stats for debugging
        print(f"Checked Memory:  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB ; Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
        
        allocated_ratio = allocated_memory / max_memory_allocated

        # Warning if approaching threshold
        if allocated_memory > self.threshold_gb:
            print(f"WARNING: High memory usage ({allocated_memory:.2f}GB > {self.threshold_gb}GB) - Cleaning up memory ...")
            self.soft_cleanup()
        elif allocated_ratio > self.allocation_threshold:
            print(f"CRITICAL: High memory usage ({allocated_ratio:.2f} > {self.allocation_threshold:.2f}) - Deep cleaning up memory ...")
            self.deep_cleanup()
        
    
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
        if state.global_step % 100 == 0:  # Check CUDA memory every 100 steps - more frequent
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
                writer.writerow(['step', 'epoch', 'training_loss', 'validation_loss', 'wer'])

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
            ])
