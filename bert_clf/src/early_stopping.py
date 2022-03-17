import json
import os

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, config=None):
        """
        Args:
            config (dict): Parameters for training the model
        """
        self.patience = config['training']['patience']
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = config['training']['delta']
        self.config = config

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""

        if self.config['training']['save_state_dict']:
            torch.save(
                model.state_dict(),
                os.path.join(self.config["training"]["output_dir"], "model.pth"),
            )

            with open(
                    os.path.join(self.config["training"]["output_dir"], 'label_mapper.json'),
                    mode='w',
                    encoding='utf-8'
            ) as f:
                json.dump(model.mapper, f, indent=4, ensure_ascii=False)
        else:
            torch.save(
                model,
                os.path.join(self.config["training"]["output_dir"], "model.pth"),
            )

        self.val_loss_min = val_loss
