import hypothesis
import torch

from torch.utils.data import DataLoader



class Trainer:
    r"""Base `Trainer` interface.

    Args:
        ....
    Hooks:
        hypothesis.tags.checkpoint
        hypothesis.tags.epoch
        hypothesis.tags.step
        hypothesis.tags.validate
    """

    def __init__(self, dataset, allocate_optimizer, epochs=1, data_workers=2,
                 batch_size=32, allocate_scheduler=None,
                 pin_memory=False, shuffle=False):
        self.allocate_optimizer = allocate_optimizer
        self.allocate_scheduler = allocate_scheduler
        self.batch_size = batch_size
        self.data_workers = data_workers
        self.dataset = dataset
        self.epochs = epochs
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        # Allocate the hooks.
        hypothesis.hook.add_tag("checkpoint")
        hypothesis.hook.add_tag("epoch")
        hypothesis.hook.add_tag("step")
        hypothesis.hook.add_tag("validate")

    def _reset(self):
        if self.allocate_optimizer is not None:
            self.optimizer = self.allocate_optimizer(self.model)
        else:
            self.optimizer = None
        if self.allocate_scheduler is not None:
            self.scheduler = self.allocate_scheduler(self.optimizer)
        else:
            self.scheduler = None

    def scheduler_step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def epoch(self, epoch):
        self.scheduler_step()
        loader = iter(DataLoader(self.dataset, num_workers=self.data_workers,
            batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=self.shuffle))
        num_iterations = len(self.dataset)
        for iteration in range(num_iterations):
            try:
                loss = self.step(loader)
                hypothesis.hook_call(hypothesis.tags.step, self, loss=loss)
            except Exception as e:
                hypothesis.hook_call(hypothesis.tags.exception, self, exception=e)
        # Check if the training supports checkpointing.
        hypothesis.hook_call(hypothesis.tags.checkpoint, self, model=model, epoch=epoch)
        del loader # Free up the loader.

    def step(self, loader):
        raise NotImplementedError

    def train(self, model):
        self.model = model
        self._reset()
        hypothesis.hook_call(hypothesis.tags.start, self)
        # Seed the initial validation score.
        hypothesis.hook_call(hypothesis.tags.validate, self, model=model, epoch=-1)
        # Start the training procedure.
        for epoch in range(self.epochs):
            self.epoch(epoch)
            hypothesis.hook_call(hypothesis.tags.validate, self, model=model, epoch=epoch)
        hypothesis.hook_call(hypothesis.tags.end, self)
