import hypothesis
import torch

from torch.utils.data import DataLoader



class Trainer:
    r"""Base `Trainer` interface.

    Args:
        ....
    Hooks:
        hypothesis.tags.epoch
        hypothesis.tags.step
        hypothesis.tags.validate
    """

    def __init__(self, model, dataset, optimizer, epochs=1,
                 data_workers=2, batch_size=32, scheduler=None,
                 pin_memory=False, shuffle=False):
        self.batch_size = batch_size
        self.data_workers = data_workers
        self.dataset = dataset
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self.scheduler = scheduler
        self.shuffle = shuffle
        # Allocate the hooks.
        hypothesis.hook.add_tag("epoch")
        hypothesis.hook.add_tag("step")
        hypothesis.hook.add_tag("validate")

    def dataset_iterations(self):
        return int(len(self.dataset) // self.batch_size)

    def scheduler_step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def epoch(self, epoch):
        self.scheduler_step()
        loader = iter(DataLoader(self.dataset, num_workers=self.data_workers,
            batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=self.shuffle))
        num_iterations = self.dataset_iterations()
        for iteration in range(num_iterations):
            try:
                loss = self.step(loader)
                hypothesis.hook_call(hypothesis.tags.step, self, loss=loss)
            except Exception as e:
                hypothesis.hook_call(hypothesis.tags.exception, self, exception=e)
        hypothesis.hook_call(hypothesis.tags.epoch, self, model=self.model, epoch=epoch)
        del loader # Free up the loader.

    def step(self, loader):
        raise NotImplementedError

    def train(self):
        hypothesis.hook_call(hypothesis.tags.start, self)
        # Seed the initial validation score.
        hypothesis.hook_call(hypothesis.tags.validate, self, model=self.model, epoch=-1)
        # Start the training procedure.
        for epoch in range(self.epochs):
            self.epoch(epoch)
            hypothesis.hook_call(hypothesis.tags.validate, self, model=self.model, epoch=epoch)
        hypothesis.hook_call(hypothesis.tags.end, self)
