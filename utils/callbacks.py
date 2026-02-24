import torch
import pytorch_lightning as pl
import os

class SaveLatestBackboneCallback(pl.callbacks.Callback):
    def __init__(self, checkpoint_callback=None):
        super().__init__()
        self.save_dir = None
        self.latest_backbone_path = None
        self.checkpoint_callback = checkpoint_callback 

    def on_train_end(self, trainer, pl_module):
        if self.checkpoint_callback and self.checkpoint_callback.dirpath:
               self.save_dir = self.checkpoint_callback.dirpath
        else:
            self.save_dir = '.'
            print('Checkpoint callback not provided, using current directory.')
            
        os.makedirs(self.save_dir, exist_ok=True)
        self.latest_backbone_path = os.path.join(self.save_dir, "latest_backbone.pt")
        torch.save(pl_module.backbone.state_dict(), self.latest_backbone_path)


class SaveBestBackboneCallback(pl.callbacks.Callback):
    def __init__(self, monitor="val_loss", mode="min", checkpoint_callback=None):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.checkpoint_callback = checkpoint_callback
        self.best_backbone_path = None  #set on_validation_end

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)

        if current_score is None:
            return

        if self.best_score is None or (
            self.mode == "max" and current_score > self.best_score
        ) or (
            self.mode == "min" and current_score < self.best_score
        ):  
            # Get the directory from checkpoint callback
            if self.checkpoint_callback and self.checkpoint_callback.dirpath:
               save_dir = self.checkpoint_callback.dirpath
            else:
                save_dir = '.'
                print('Checkpoint callback not provided, using current directory.')
            
            # Ensure the directory exists
            os.makedirs(save_dir, exist_ok=True)
            self.best_backbone_path = os.path.join(save_dir, "best_backbone.pt")
            torch.save(pl_module.backbone.state_dict(), self.best_backbone_path)

            self.best_score = current_score
            print(f"Best backbone saved to {self.best_backbone_path} with {current_score} {self.monitor}.")