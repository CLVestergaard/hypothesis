"""Approximate Bayesian Computation"""

import hypothesis
import numpy as np
import torch

from hypothesis.inference import Method
from hypothesis.util.common import load_argument



class ApproximateBayesianComputation(Method):
    r"""Vanilla Approximate Bayesian Computation.

    Args:
        TODO

    Hooks:
        TODO
    """

    KEY_NUM_SAMPLES = "samples"

    def __init__(self, prior, model, summary, distance, epsilon=.01):
        super(ApproximateBayesianComputation, self).__init__()
        self.distance = distance
        self.epsilon = epsilon
        self.model = model
        self.prior = prior
        self.summary = summary
        self.summary_observations = None
        self.num_observations = None

    def sample(self, observations):
        accepted = False

        while not accepted:
            theta = self.prior.sample()
            inputs = theta.repeat(self.num_observations)
            try:
                outputs = self.model(inputs)
            except Exception as e:
                hypothesis.hook_call(hypothesis.hook.tags.exception, self, exception=e)
                continue
            summary_outputs = self.summary(outputs)
            distance = self.distance(self.summary_observations, summary_outputs)
            if (distance < self.epsilon).all():
                accepted = True
                sample = theta

        return sample

    def infer(self, observations, **kwargs):
        samples = []

        # Initialize the summary statistic of the current observations.
        self.summary_observations = self.summary(observations)
        self.num_observations = observations.size(0)
        num_samples = load_argument(self.KEY_NUM_SAMPLES, **kwargs)
        for sample_index in range(num_samples):
            sample = self.sample(observations)
            samples.append(sample)

        return samples
