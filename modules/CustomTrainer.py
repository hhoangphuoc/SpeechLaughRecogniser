from transformers import Trainer, EvalPrediction
import torch
from torch.utils.data import DataLoader
import gc
from typing import Optional, Dict, Union, Any
import numpy as np




#===============================================================================================
#                           CUSTOM TRAINER
#===============================================================================================
class MemoryEfficientTrainer(Trainer):
    """
    A memory-efficient trainer that handles evaluation data loading separately
    and implements memory cleanup during training and evaluation.
    """
    def __init__(self, compute_metrics=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = compute_metrics
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

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
            
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=1,  # Reduced for memory efficiency
            pin_memory=True,
            drop_last=False
        )

    def evaluation_loop(
        self,
        dataloader=None,
        description: str = "Evaluation",
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """Memory-efficient evaluation loop with proper metrics handling"""
        try:
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"\nInitial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

            # Get dataloader
            eval_dataloader = dataloader if dataloader is not None else self.get_eval_dataloader()
            print(f"Evaluation batch size: {eval_dataloader.batch_size}")

            # Initialize collections
            all_preds = []
            all_labels = []
            eval_losses = []

            self.model.eval()
            
            for step, batch in enumerate(eval_dataloader):
                # Move batch to device
                batch = self._prepare_inputs(batch)
                # batch = {
                #     k: v.to(self.args.device)
                #     for k, v in batch.items()
                # }
                
                with torch.no_grad():
                    outputs = self.model(**batch)

                    # Calculate validation loss
                    loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
                    eval_losses.append(loss.detach().cpu())
                    
                    if not prediction_loss_only:
                        # Get predictions
                        preds = outputs.logits.argmax(dim=-1)
                        # Convert to numpy and store
                        all_preds.extend(preds.cpu().numpy())

                        # Get labels
                        labels = batch["labels"]
                        labels[labels == -100] = self.tokenizer.pad_token_id
                        all_labels.extend(labels.cpu().numpy())
                
                # Cleanup
                del outputs, batch
                if step % 1000 == 0:  # Periodic cleanup
                    torch.cuda.empty_cache()
                    gc.collect()

            # Compute average loss
            eval_loss = torch.stack(eval_losses).mean().item()
            metrics = {f"{metric_key_prefix}_loss": eval_loss}

            # Compute other metrics
            if not prediction_loss_only and self.compute_metrics is not None:
                print("Compute evaluation...")
                # Convert to numpy arrays
                all_preds = [np.array(pred) for pred in all_preds]
                all_labels = [np.array(label) for label in all_labels]

                all_preds = np.array(all_preds, dtype=object)
                all_labels = np.array(all_labels, dtype=object)
                
                eval_pred = EvalPrediction(
                    predictions=all_preds,
                    label_ids=all_labels
                )
                
                computed_metrics = self.compute_metrics(eval_pred, eval_loss)
                self.log(computed_metrics) #Log metrics to the output

                metrics.update({
                    f"{metric_key_prefix}_{k}": v 
                    for k, v in computed_metrics.items()
                })
                print(f"Evaluation metrics: {metrics}")
            else:
                print(f"Unable to compute evaluation")
                
            return metrics

        except Exception as e:
            print(f"Error in evaluation_loop: {str(e)}")
            if torch.cuda.is_available():
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
