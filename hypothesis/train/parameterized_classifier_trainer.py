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
                 batch_size=32, scheduler= None, pin_memory=False,
                 shuffle=False, criterion=torch.nn.BCELoss(reduction="sum")):
        # Initialize the parent object.
        super(ParameterizedClassifierTrainer, self).__init__(
            model, dataset, optimizer, epochs, data_workers,
            batch_size, scheduler, pin_memory)
        # Set the number of epoch iterations.
        self.epoch_iterations = int(len(dataset) / batch_size / 2)
        self.criterion = criterion.to(hypothesis.device)
        self.zeros = torch.zeros(self.batch_size, 1).to(hypothesis.device)
        self.ones = torch.ones(self.batch_size, 1).to(hypothesis.device)

    def step(self, loader):
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
        loss = loss_a + loss_b
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
