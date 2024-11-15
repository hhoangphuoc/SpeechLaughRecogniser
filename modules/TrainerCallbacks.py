import os
import pandas as pd
import torch
import gc
import time
import psutil
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback

#------------------------------------------------------------------------------------------------
def monitor_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    process = psutil.Process()
    print(f"RAM Memory: {process.memory_info().rss / 1024**2:.2f} MB")


class MemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:  # Check CUDA memory every 100 steps
            torch.cuda.empty_cache()
            gc.collect() #FIXME - Not using gc.collect() as it slows down the training

class MetricsCallback(TrainerCallback):
    """
    Callback to save training and evaluation metrics to CSV every 1000 steps.
    """
    def __init__(self, output_dir, save_steps=1000):
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.last_save_step = 0
        
        # Initialize DataFrame with columns
        self.metrics_df = pd.DataFrame(columns=[
            "step", 
            "epoch", 
            "timestamp",
            "training_loss", 
            "validation_loss",
            "wer",
            "f1"
        ])
        
        # Create tensorboard writer
        self.writer = SummaryWriter(
            log_dir=os.path.join(output_dir, "tensorboard"))
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each step."""
        current_step = state.global_step
        
        # Check if it's time to save metrics
        if current_step % self.save_steps == 0 and current_step != self.last_save_step:
            self.last_save_step = current_step
            
            # Get current metrics
            new_metrics = {
                "step": current_step,
                "epoch": state.epoch,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "training_loss": state.log_history[-1].get("loss", 0) if state.log_history else 0,
                "validation_loss": state.log_history[-1].get("eval_loss", 0) if state.log_history else 0,
                "wer": state.log_history[-1].get("eval_wer", 0) if state.log_history else 0,
                "f1": state.log_history[-1].get("eval_f1", 0) if state.log_history else 0
            }

            # Append to DataFrame
            self.metrics_df.loc[len(self.metrics_df)] = new_metrics
            
            # Log to TensorBoard
            self._log_to_tensorboard(new_metrics, current_step)
            
            # Save to CSV
            self.save_metrics()
            
            print(f"\nMetrics saved at step {current_step}")
            print(f"Current WER: {new_metrics['wer']:.2f}")
            print(f"Current F1: {new_metrics['f1']:.2f}\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics is None:
            return
        
        # Update TensorBoard with evaluation metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"eval/{key}", value, state.global_step)

    def _log_to_tensorboard(self, metrics, step):
        """Helper method to log metrics to TensorBoard."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key != "step" and key != "epoch":
                self.writer.add_scalar(f"train/{key}", value, step)

    def save_metrics(self):
        """Save metrics to CSV file with timestamp in filename."""
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Save with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(metrics_dir, f"training_metrics_{timestamp}.csv")
        
        # Save current metrics
        self.metrics_df.to_csv(csv_path, index=False)
        
        # Also save to a fixed filename for easy access to latest metrics
        latest_csv_path = os.path.join(metrics_dir, "training_metrics_latest.csv")
        self.metrics_df.to_csv(latest_csv_path, index=False)

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        # Save final metrics
        self.save_metrics()
        
        # Close TensorBoard writer
        self.writer.close()
        
        print("\nTraining completed. Final metrics saved.")
        print(f"Total steps: {state.global_step}")
        print(f"Final WER: {self.metrics_df['wer'].iloc[-1]:.2f}")
        print(f"Final F1: {self.metrics_df['f1'].iloc[-1]:.2f}")

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        print("\nMetrics tracking started.")
        print(f"Saving metrics every {self.save_steps} steps")
        print(f"Metrics will be saved to: {self.output_dir}/metrics/")