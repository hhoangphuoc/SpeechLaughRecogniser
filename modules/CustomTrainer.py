from transformers import Seq2SeqTrainer, EvalPrediction
import torch
import gc
from typing import Optional, Dict, Union, Any
import numpy as np

class MemoryEfficientTrainer(Seq2SeqTrainer):
    """
    A memory-efficient trainer that handles evaluation data loading separately
    and implements memory cleanup during training and evaluation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_dataloader = None
        self.last_eval_memory = 0
        
    def create_eval_dataloader(self):
        """Create evaluation dataloader with memory-efficient settings"""
        if self.eval_dataset is None:
            return None
            
        # Memory-efficient dataloader settings
        return self.get_eval_dataloader(
            self.eval_dataset,
            num_workers=1,  # Reduce worker count
            pin_memory=True,
            prefetch_factor=1,
            drop_last=False
        )

    def evaluation_loop(
        self,
        dataloader = None,
        description: str = "Evaluation",
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list] = None,
        metric_key_prefix: str = "eval",
    ) -> Union[Dict[str, float], Dict[str, Any]]:
        """Memory-efficient evaluation loop with proper metrics handling"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated() / 1e9
                print(f"\nMemory before evaluation: {initial_memory:.2f}GB")

            if dataloader is None:
                dataloader = self.create_eval_dataloader()

            # Modified prediction collection
            all_preds = []
            all_labels = []
            eval_losses = []

            self.model.eval()
            
            for batch in dataloader:
                # Move batch to device
                batch = self._prepare_inputs(batch)
                
                with torch.no_grad():
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    eval_losses.append(loss.detach().cpu())
                    
                    # Store predictions and labels
                    if not prediction_loss_only:
                        # Get full sequence predictions instead of just logits
                        predictions = outputs.logits.argmax(dim=-1)
                        labels = batch["labels"]
                        
                        # Store full sequences
                        all_preds.append(predictions)
                        all_labels.append(labels)
                
                del outputs, batch
                torch.cuda.empty_cache()

            # Create EvalPrediction object
            eval_loss = torch.stack(eval_losses).mean().item()
            metrics = {f"{metric_key_prefix}_loss": eval_loss}
            
            if not prediction_loss_only:
                # Stack predictions and labels
                predictions = torch.cat(all_preds, dim=0)
                labels = torch.cat(all_labels, dim=0)
                
                # Create EvalPrediction object
                eval_pred = EvalPrediction(
                    predictions=predictions,
                    label_ids=labels
                )
                
                # Compute metrics
                if self.compute_metrics is not None:
                    computed_metrics = self.compute_metrics(eval_pred)
                    metrics.update(computed_metrics)
            
            # Final memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated() / 1e9
                print(f"Memory after evaluation: {final_memory:.2f}GB")
                
            return metrics
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            torch.cuda.empty_cache()
            gc.collect()
            raise e

    def train(self, *args, **kwargs):
        """
        Memory-efficient training with periodic evaluation
        """
        try:
            # Initial memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated() / 1e9
                print(f"Memory at training start: {initial_memory:.2f}GB")
            
            # Training loop
            result = super().train(*args, **kwargs)
            
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated() / 1e9
                print(f"Memory at training end: {final_memory:.2f}GB")
                print(f"Total memory change: {final_memory - initial_memory:.2f}GB")
            
            return result
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise e
