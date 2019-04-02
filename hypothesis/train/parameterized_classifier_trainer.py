import hypothesis
import torch

from hypothesis.train import Trainer



class ParameterizedClassifierTrainer(Trainer):
    r""""Training interface for parameterized classifiers.

    Args:
        ...
    Hooks:
        ...
    """

    def __init__(self, model, dataset, optimizer, epochs=1, data_workers=2,
                 batch_size=32, scheduler= None, pin_memory=False, shuffle=False,
                 criterion=torch.nn.BCELoss(reduction="sum")):
        # Initialize the parent object.
        super(ParameterizedClassifierTrainer, self).__init__(
            model, dataset, optimizer, epochs, data_workers,
            batch_size, scheduler, pin_memory, shuffle)
        # Set the number of epoch iterations.
        self.epoch_iterations = int(len(dataset) / batch_size / 2)
        self.criterion = criterion.to(hypothesis.device)
        self.has_targets = (len(self.dataset[0]) > 0)
        if not self.has_targets:
            self.zeros = torch.zeros(self.batch_size, 1).to(hypothesis.device)
            self.ones = torch.ones(self.batch_size, 1).to(hypothesis.device)
            self.step_f = self._step_without_targets
        else:
            self.step_f = self._step_with_targets

    def dataset_iterations(self):
        n = len(self.dataset) // self.batch_size
        if not self.has_targets:
            n /= 2

        return int(n)

    def _step_with_targets(self, loader):
        thetas, x_thetas, targets = next(loader)
        thetas = thetas.to(hypothesis.device, non_blocking=True)
        x_thetas = x_thetas.to(hypothesis.device, non_blocking=True)
        targets = targets.to(hypothesis.device, non_blocking=True)
        y = self.model(x_thetas, thetas)
        loss = self.criterion(y, targets)

        return loss

    def _step_without_targets(self, loader):
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
        # Combine losses and backpropagate.
        loss_a = self.criterion(y, self.zeros) + self.criterion(y_hat, self.ones)
        loss_b = self.criterion(y_reverse, self.zeros) + self.criterion(y_hat_reverse, self.ones)

        return (loss_a + loss_b) / 4

    def step(self, loader):
        loss = self.step_f(loader)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
