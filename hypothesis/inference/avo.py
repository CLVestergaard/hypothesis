"""Adversarial Variational Optimization"""

import hypothesis
import torch

from hypothesis.inference import Method
from hypothesis.util.common import load_argument
from hypothesis.util.common import sample
from hypothesis.util.constant import epsilon



class AdversarialVariationalOptimizationPlus(Method):
    r"""Adversarial Variational Optimization (AVO) +

    A modified and faster implementation of arxiv.org/abs/1707.07113
    """
    DEFAULT_BATCH_SIZE = 32
    KEY_BATCH_SIZE = "batch_size"
    KEY_PROPOSAL = "proposal"
    KEY_STEPS = "steps"

    def __init__(self, implicit_model, discriminator, lr_discriminator=.001,
                 lr_proposal=.01, criterion=torch.nn.BCELoss(), gamma=5):
        super(AdversarialVariationalOptimizationPlus, self).__init__()
        self.batch_size = self.DEFAULT_BATCH_SIZE
        self.discriminator = discriminator.to(hypothesis.device)
        self.gamma = gamma
        self.lr_discriminator = lr_discriminator
        self.lr_proposal = lr_proposal
        self.model = implicit_model
        self.optimizer_discriminator = None
        self.optimizer_proposal = None
        self.proposal = None
        self.ones = torch.ones(self.batch_size, 1).to(hypothesis.device)
        self.zeros = torch.zeros(self.batch_size, 1).to(hypothesis.device)
        self.criterion = criterion.to(hypothesis.device)

    def reset(self):
        # Allocate the optimizers.
        self.allocate_optimizers()

    def allocate_optimizers(self):
        # Discriminator optimizer.
        self.optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr_discriminator)
        # Proposal optimizer.
        proposal_loc_parameter = [self.proposal.parameters()[0]]
        self.optimizer_proposal = torch.optim.Adam(
            proposal_loc_parameter, lr=self.lr_proposal)

    def update_discriminator(self, observations, inputs, outputs):
        x_real = sample(observations, self.batch_size).view(self.batch_size, -1)
        x_real.requires_grad = True
        x_fake = outputs.view(self.batch_size, -1)
        y_real = self.discriminator(x_real)
        y_fake = self.discriminator(x_fake)
        loss = (self.criterion(y_real, self.ones) + self.criterion(y_fake, self.zeros))
        loss = loss + self.gamma * r1_regularization(y_real, x_real).mean()
        self.optimizer_discriminator.zero_grad()
        loss.backward()
        self.optimizer_discriminator.step()
        x_real.requires_grad = False

    def update_proposal(self, observations, inputs, outputs):
        with torch.no_grad():
            loc = self.proposal.parameters()[0]
            repeated_loc = loc.repeat(1, self.batch_size).view(self.batch_size, -1)
            gradients = (inputs - repeated_loc)
            subsampled = sample(observations, self.batch_size).view(self.batch_size, -1)
            score = (self.discriminator(subsampled).mean() - self.discriminator(outputs))
            gradient = (gradients * score).mean(dim=0).view(-1).squeeze()
        loc.grad = gradient
        self.optimizer_proposal.step()

    def sample_and_simulate(self):
        inputs = self.proposal.sample(self.batch_size).view(self.batch_size, -1)
        outputs = self.model(inputs)

        return inputs, outputs

    def step(self, observations):
        inputs, outputs = self.sample_and_simulate()
        self.update_discriminator(observations, inputs, outputs)
        self.update_proposal(observations, inputs, outputs)

    def infer(self, observations, **kwargs):
        # Fetch the desired number of optimization.
        steps = load_argument(self.KEY_STEPS, **kwargs)
        proposal = load_argument(self.KEY_PROPOSAL, **kwargs)
        batch_size = load_argument(self.KEY_BATCH_SIZE, **kwargs, default=self.DEFAULT_BATCH_SIZE)
        # Check if a proper number of steps and proposal has been specified.
        if steps is None or steps <= 0:
            raise ValueError("Please specify 'steps' > 0.")
        steps = int(steps)
        # Check if a proper proposal has been specified.
        if proposal is None:
            raise ValueError("Please specify a 'proposal'.")
        self.proposal = proposal
        self.batch_size = int(batch_size)
        self.allocate_optimizers()
        for step in range(steps):
            self.step(observations)
        final_proposal = self.proposal.clone()

        return final_proposal



def r1_regularization(y_hat, x):
    """R1 regularization from Mesheder et al, 2017."""
    batch_size = x.size(0)
    grad_y_hat = torch.autograd.grad(
        outputs=y_hat.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    regularizer = grad_y_hat.pow(2).view(batch_size, -1).sum()

    return regularizer
