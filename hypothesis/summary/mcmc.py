"""Summaries and statistic for MCMC methods."""

import numpy as np
import torch



class Chain:
    r"""Summary of a Markov chain from an MCMC sampler.

    Args:
        chain (sequence): sequence of MCMC states.
        probabilities (sequence): sequence of proposal probabilities.
        acceptances (sequence): sequence of accept, reject flags.
    """

    def __init__(self, chain,
                 probabilities,
                 acceptances):
        # Initialize the main chain states.
        chain = torch.cat(chain, dim=0).squeeze()
        d = chain[0].dim()
        if d == 0:
            chain = chain.view(-1, 1)
        probabilities = torch.tensor(probabilities).squeeze()
        acceptances = acceptances
        self.chain = chain
        self.probabilities = probabilities
        self.acceptances = acceptances

    def mean(self, parameter_index=None, burnin=False):
        return self.chain[:, parameter_index].mean(dim=0).squeeze()

    def variance(self, parameter_index=None):
        return (self.chain[:, parameter_index].std(dim=0) ** 2).squeeze()

    def monte_carlo_error(self):
        variance = self.variance()
        effective_sample_size = self.effective_size()

        return (variance / effective_sample_size).sqrt()

    def size(self):
        return len(self.chain)

    def min(self):
        return self.chain.min()

    def max(self):
        return self.chain.max()

    def state_dim(self):
        return self.chain[0].view(-1).dim()

    def acceptances(self):
        return self.acceptances

    def acceptance_ratio(self):
        raise NotImplementedError

    def get(self, parameter_index=None, burnin=False):
        return self.chain[:, parameter_index].squeeze().clone()

    def probabilities(self):
        return self.probabilities.squeeze()

    def autocorrelation(self, lag, parameter_index=None):
        with torch.no_grad():
            num_parameters = self.state_dim()
            thetas = self.chain.clone()
            sample_mean = self.mean(parameter_index)
            if lag > 0:
                padding = torch.zeros(lag, num_parameters)
                lagged_thetas = thetas[lag:, parameter_index].view(-1, num_parameters).clone()
                lagged_thetas -= sample_mean
                padded_thetas = torch.cat([lagged_thetas, padding], dim=0)
            else:
                padded_thetas = thetas
            thetas -= sample_mean
            rhos = thetas * padded_thetas
            rho = rhos.sum(dim=0).squeeze()
            rho *= (1. / (self.size() - lag))
        del thetas
        del padded_thetas
        del rhos

        return rho

    def autocorrelation_function(self, parameter_index=None, interval=1, max_lag=None):
        if not max_lag:
            max_lag = self.size() - 1
        x = np.arange(0, max_lag + 1, interval)
        y_0 = self.autocorrelation(lag=0, parameter_index=parameter_index)
        y = [self.autocorrelation(lag=tau, parameter_index=parameter_index) / y_0 for tau in x]

        return x, y

    def integrated_autocorrelation(self, M=None, interval=1):
        int_tau = 0.
        if not M:
            M = self.size() - 1
        c_0 = self.autocorrelation(0)
        for index in range(M):
            int_tau += self.autocorrelation(index) / c_0

        return int_tau

    def efficiency(self):
        return self.effective_size() / self.size()

    def effective_size(self):
        # TODO Support multi-dimensional
        y_0 = self.autocorrelation(0)
        M = 0
        for lag in range(self.size()):
            y = self.autocorrelation(lag)
            p = y / y_0
            if p <= 0:
                M = lag - 1
                break
        tau = self.integrated_autocorrelation(M)
        effective_size = (self.size() / tau)

        return int(abs(effective_size))

    def thin(self):
        chain = []
        p = self.efficiency()
        probabilities = []
        acceptances = []
        for index in range(self.size()):
            u = np.random.uniform()
            if u <= p:
                chain.append(self.chain[index])
                probabilities.append(self.probabilities[index])
                acceptances.append(self.acceptances[index])

        return Chain(chain, probabilities, acceptances)
