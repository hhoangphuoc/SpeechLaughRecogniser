from transformers import Seq2SeqTrainer, EvalPrediction
from torch.utils.data import DataLoader, SequentialSampler
import torch
import numpy as np
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataset = self.train_dataset
        self.eval_dataset = self.eval_dataset

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.args.train_batch_size, 
            # sampler=self._get_train_sampler(),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,  # Use multiple workers
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            # sampler=SequentialSampler(eval_dataset),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,  # Use multiple workers
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )
    
    def evaluate(self, *args, **kwargs):
        print("Evaluating eval dataset...")
        # return super().evaluate(*args, **kwargs)
        eval_dataset = kwargs.get('eval_dataset')
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        all_preds = []
        all_labels = []
        val_losses = []

        for step, batch in enumerate(eval_dataloader):
            
            # Process in Dataloader beforehands to ensure data only move to device when computed
            #move to device
            batch = {k: v.to(self.args.device) for k, v in batch.items()} # batch = {input_ids: tensor, attention_mask: tensor, labels: tensor}

            #forward pass
            with torch.no_grad():
                outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                all_preds.extend(predictions)
            
                # Handle label padding
                labels = batch['labels']
                labels[labels == -100] = self.tokenizer.pad_token_id
                all_labels.extend(labels)
                

                #compute loss
                val_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
                val_losses.append(val_loss.item())
            
            avg_loss = np.mean(val_losses)
            print(f"Average loss: {avg_loss:.4f}")

            if self.compute_metrics is not None:
                print("Computing metrics...")
                # Convert predictions and labels to numpy arrays
                all_preds = [np.array(pred) for pred in all_preds]
                all_labels = [np.array(label) for label in all_labels]
                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                #compute metrics

                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, labels=all_labels), avg_loss)
                prefix = "eval_"
                for k, v in metrics.items():
                    metrics[f"{prefix}_{k}"] = v
                
                print(f"Metrics: {metrics}")
            elif self.compute_metrics is None:
                print("No metrics to compute")
                return {}

        return metrics

