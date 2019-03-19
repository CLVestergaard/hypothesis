"""Proposals"""

import torch

from torch.distributions.normal import Normal as NormalDistribution
from torch.distributions.multivariate_normal import MultivariateNormal as MultivariateNormalDistribution



class Proposal:
    r"""Abstract base class for a proposal with parameters $\theta$."""

    def clone(self):
        raise NotImplementedError

    def log_prob(self, thetas):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def sample(self, num_samples):
        raise NotImplementedError



class Normal(Proposal):
    r"""An univariate normal proposal."""

    def __init__(self, mu=0, sigma=1):
        super(Normal, self).__init__()
        self.mu = torch.tensor(mu).float().squeeze().detach()
        self.sigma = torch.tensor(sigma).float().squeeze().detach()
        self.distribution = NormalDistribution(self.mu, self.sigma)
        self.mu.requires_grad = True
        self.sigma.requires_grad = True
        self.params = [self.mu, self.sigma]

    def clone(self):
        return Normal(self.mu.item(), self.sigma.item())

    def log_prob(self, thetas):
        return self.distribution.log_prob(thetas).view(-1)

    def parameters(self):
        return self.params

    def sample(self, num_samples):
        return self.distribution.sample(torch.Size([num_samples]))



class MultivariateNormal(Proposal):
    r"""A multivariate normal proposal."""

    def __init__(self, mu, sigma):
        super(MultivariateNormal, self).__init__()
        self.mu = torch.tensor(mu).float()
        self.mu.requires_grad = True
        self.sigma = torch.tensor(sigma).float()
        self.sigma.requires_grad = True
        self.distribution = MultivariateNormalDistribution(self.mu, self.sigma)
        self.params = [self.mu, self.sigma]
        self.mask = torch.eye(self.sigma.shape[0])

    def clone(self):
        return MultivariateNormal(self.mu.detach(), self.sigma.detach())

    def log_prob(self, thetas):
        return self.distribution.log_prob(thetas)

    def parameters(self):
        return self.params

    def sample(self, num_samples):
        return self.distribution.sample(torch.Size([num_samples]))
