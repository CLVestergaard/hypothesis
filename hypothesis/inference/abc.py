"""Approximate Bayesian Computation"""

import hypothesis
import numpy as np
import torch

from hypothesis.inference import Method



class ApproximateBayesianComputation(Method):
    r"""Vanilla Approximate Bayesian Computation

    Arguments:
        prior (Distribution): Prior distribution over the parameters of interest
        model (Simulator): TODO
        summary (lambda): Function to generate the summary statistic
        distance (lambda): Function expressing the distance between two given summary statistics
        epsilon (float): Acceptance threshold

    Hooks:
        hypothesis.hooks.start
        hypothesis.hooks.completed
        hypothesis.hooks.pre_step
        hypothesis.hooks.post_step
        hypothesis.hooks.pre_simulation
        hypothesis.hooks.post_simulation
        hypothesis.hooks.pre_inference
        hypothesis.hooks.post_inference
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

    def sample(self, observations):
        accepted = False
        num_observations = observations.size(0)
        sample = None
        while not accepted:
            theta = self.prior.sample()
            inputs = theta.repeat(num_observations)
            hypothesis.call_hooks(hypothesis.hooks.pre_simulation, self, inputs=inputs)
            try:
                outputs = self.model(inputs)
            except Exception as e:
                hypothesis.call_hooks(hypothesis.hooks.exception, self, exception=e)
                continue
            hypothesis.call_hooks(hypothesis.hooks.post_simulation, self, inputs=inputs, outputs=outputs)
            summary_outputs = self.summary(outputs)
            distance = self.distance(self.summary_observations, summary_outputs)
            if distance < self.epsilon:
                accepted = True
                sample = theta

        return sample

    def infer(self, observations, **kwargs):
        samples = []
        self.summary_observations = self.summary(observations)
        num_samples = int(kwargs[self.KEY_NUM_SAMPLES])
        hypothesis.call_hooks(hypothesis.hooks.pre_inference, self)
        counter = 0
        for sample_index in range(num_samples):
            hypothesis.call_hooks(hypothesis.hooks.pre_step, self)
            sample = self.sample(observations)
            counter += 1
            print(counter)
            hypothesis.call_hooks(hypothesis.hooks.post_step, self, sample=sample)
            samples.append(sample)
        hypothesis.call_hooks(hypothesis.hooks.post_inference, self, samples=samples)

        return samples
