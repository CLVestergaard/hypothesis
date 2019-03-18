"""Markov chain Monte Carlo for inference.
"""

import hypothesis
import numpy as np
import torch

from hypothesis.inference import Method
from hypothesis.inference.approximate_likelihood_ratio import likelihood_to_evidence_ratio
from hypothesis.inference.approximate_likelihood_ratio import log_likelihood_ratio
from hypothesis.summary.mcmc import Chain
from hypothesis.util.common import load_argument
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal



class MarkovChainMonteCarlo(Method):
    r"""General interface for Markov chain Monte Carlo (MCMC) samplers.

    Hooks:
        hypothesis.hook.tags.start
        hypothesis.hook.tags.pre_step
        hypothesis.hook.tags.post_step
        hypothesis.hook.tags.end
    """

    KEY_INITIAL_THETA = "theta_0"
    KEY_SAMPLES = "samples"
    KEY_BURNIN_SAMPLES = "burnin_samples"

    def __init__(self):
        super(MarkovChainMonteCarlo, self).__init__()
        # Add the default MCMC hooks.
        hypothesis.hook.add_tag("pre_step")
        hypothesis.hook.add_tag("post_step")

    def step(self, observations, theta):
        r"""Computes a single step of an MCMC chain."""
        raise NotImplementedError

    def sample(self, observations, theta, num_samples):
        r"""Samples the specified number of MCMC samples."""
        samples = []
        probabilities = []
        acceptances = []

        for sample_index in range(num_samples):
            # Call the pre-hook step.
            hypothesis.call_hook(hypothesis.hook.tag.pre_step, self, theta=theta)
            # Apply the MCMC step.
            theta, probability, accepted = self.step(observations, theta)
            # Call the post-step hook.
            hypothesis.call_hook(hypothesis.hook.tag.post_step, self,
                                 theta=theta, probability=probability,
                                 accepted=accepted)
            with torch.no_grad():
                samples.append(theta.squueze().cpu()) # Move sample to CPU.
                probabilities.append(probability)
                acceptances.append(accepted)

        return samples, probabilities, acceptances


    def infer(self, observations, **kwargs):
        r"""Samples the specified number of MCMC samples given an initial
        starting point.

        Args:
            observations (tensor): set of observations.
            samples (int): number of MCMC samples.
            theta_0 (tensor): initial theta to start sampling from.
            burnin_samples (int, optional): number of burnin samples.
        """
        burnin_samples = None
        burnin_probabilities = None
        burnin_acceptances = None

        # Call the start hook.
        hypothesis.call_hook(hypothesis.hook.tags.start, self)
        # Fetch the procedure arguments.
        theta_0 = load_argument(self.KEY_INITIAL_THETA, **kwargs)
        num_samples = load_argument(self.KEY_SAMPLES, **kwargs, default=0)
        burnin_num_samples = load_argument(self.KEY_BURNIN_SAMPLES, **kwargs, default=0)
        # Check if the initial theta has been specified.
        if theta_0 is None:
            raise ValueError("Initial model parameter 'theta_0' has not been specified.")
        # Move the observations and theta_0 to the appropriate device.
        observations = observations.to(hypothesis.device)
        theta_0 = observations.to(hypothesis.device)
        # Check if the burnin-chain needs to be samples.
        if burnin_num_samples > 0:
            # Sample the burnin chain.
            samples, probabilities, acceptances = self.sample(observations, theta_0, burnin_num_samples)
            chain_burnin = Chain(samples, probabilities, acceptances)
            theta_0 = samples[-1]
        else:
            chain_burnin = None
        # Sample the main chain.
        samples, probabilities, acceptances = self.sample(observations, theta_0, num_samples)
        chain_main = Chain(samples, probabilities, acceptances)
        # Call the end-hook.
        hypothesis.call_hook(hypothesis.hook.tags.end, self)
        # Check if a burnin-chain has been sampled.
        if chain_burnin is not None:
            return chain_burnin, chain_main
        else:
            return chain_main



class RatioMetropolisHastings(MarkovChainMonteCarlo):
    r"""Metropolis-Hastings MCMC sampler accepting a custom lambda function which
    returns the likelihood-ratio given a set of observations and two model parameters.

    Args:
        ratio (lambda): custom likelihood-ratio function.
        transition (Transition): transition distribution.

    Hooks: (extending from MarkovChainMonteCarlo)
    """

    def __init__(self, lambda_likelihood_ratio, transition):
        super(RatioMetropolisHastings, self).__init__()
        self.likelihood_ratio = lambda_likelihood_ratio
        self.transition = transition

    def step(self, observations, theta):
        accepted = False

        with torch.no_grad():
            theta_next = self.transition.sample(theta).view(-1)
            lr = self.likelihood_ratio(observations, theta, theta_next)
            if not self.transition.is_symmetric():
                t_next = self.transition.log_prob(theta, theta_next).exp()
                t_current = self.transition.log_prob(theta_next, theta).exp()
                p = (t_next / (t_current + epsilon)).item()
            else:
                p = 1
            probability = min([1, lr * p])
            u = np.random.uniform()
            if u <= probability:
                accepted = True
                theta = theta_next

        return theta, probability, accepted



class MetropolisHastings(RatioMetropolisHastings):
    r"""Metropolis Hastings MCMC sampler.

    Args:
        log_likelihood (lambda): function yielding the log-likelihood given a
        set of observations and a specific model parameter.
        transition (Transition): transition distribution.

    Hooks: (extending from MarkovChainMonteCarlo)
    """

    def __init__(self, log_likelihood, transition):
        # Define the ratio function in terms of the log-likelihood.
        def likelihood_ratio(observations, theta, theta_next):
            likelihood_current = log_likelihood(observations, theta)
            likelihood_next = log_likelihood(observations, theta_next)
            lr = likelihood_next - likelihood_current

            return lr.exp()
        # Initialize the parent with the ratio-method.
        super(MetropolisHastings, self).__init__(likelihood_ratio, transition)



class ParameterizedClassifierMetropolisHastings(RatioMetropolisHastings):
    r"""Likelihood-free parameterized classifier Metropolis-Hastings MCMC.

    Args:
        classifier (ParameterizedClassifier): a pretrained parameterized classifier.
        transition (Transition): a transition distribution.
    """

    def __init__(self, classifier, transition):
        # Define the ratio-function using the specified classifier.
        def likelihood_ratio(observations, theta, theta_next):
            return classifier.log_likelihood_ratio(observations, theta, theta_next).exp()
        # Initialize the parent.
        super(ParameterizedClassifierMetropolisHastings, self).__init__(likelihood_ratio, transition)
