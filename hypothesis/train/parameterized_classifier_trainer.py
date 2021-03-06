"""
Utilities for training parameterized classifiers.
"""

import torch
import hypothesis
import os

from hypothesis.train import Trainer
from torch.utils.data import DataLoader



class ParameterizedClassifierTrainer(Trainer):
    r"""Training interface for parameterized classifiers."""

    def __init__(self, dataset, allocate_optimizer, epochs=1, data_workers=2,
                 batch_size=32, checkpoint=None, validate=None,
                 allocate_scheduler=None, criterion=torch.nn.BCELoss(reduction="sum"),
                 pin_memory=False):
        # Initialize the parent object.
        super(ParameterizedClassifierTrainer, self).__init__(
            dataset, allocate_optimizer, epochs, data_workers,
            batch_size, checkpoint, validate, allocate_scheduler, pin_memory)
        self.epoch_iterations = int(len(dataset) / batch_size / 2)
        self.criterion = criterion.to(hypothesis.device)
        self.zeros = torch.zeros(self.batch_size, 1).to(hypothesis.device)
        self.ones = torch.ones(self.batch_size, 1).to(hypothesis.device)

    def dataset_iterations(self):
        return self.epoch_iterations

    def step(self, loader):
        try:
            # Prepare the batch.
            thetas, x_thetas = next(loader)
            thetas = thetas.to(hypothesis.device, non_blocking=True)
            x_thetas = x_thetas.to(hypothesis.device, non_blocking=True)
            thetas_hat, x_thetas_hat = next(loader)
            thetas_hat = thetas_hat.to(hypothesis.device, non_blocking=True)
            x_thetas_hat = x_thetas_hat.to(hypothesis.device, non_blocking=True)
            # First pass
            y = self.model(x_thetas, thetas)
            y_hat = self.model(x_thetas_hat, thetas)
            # Second pass
            y_reverse = self.model(x_thetas_hat, thetas_hat)
            y_hat_reverse = self.model(x_thetas, thetas_hat)
            # Combine losses and backpropagate
            loss_a = self.criterion(y, self.zeros) + self.criterion(y_hat, self.ones)
            loss_b = self.criterion(y_reverse, self.zeros) + self.criterion(y_hat_reverse, self.ones)
            loss = loss_a + loss_b
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        except Exception as e:
            hypothesis.call_hooks(hypothesis.hooks.exception, self, exception=e)
            loss = None

        return loss
