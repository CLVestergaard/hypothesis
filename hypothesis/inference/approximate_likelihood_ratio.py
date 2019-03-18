r"""Inference with Approximate Likelihood Ratios."""

import torch
import hypothesis

from hypothesis.util.constant import epsilon



def log_likelihood_ratio(classifier, observations, theta, theta_next):
    r"""Given a parameterized classifier, this method will compute the approximate
    log likelihood ratio.

    Args:
        classifier (torch.nn.Module): a parameterized classifier.
        observations (Tensor): a set of observations.
        theta (Tensor): the alternative hypothesis.
        theta_next (Tensor): the 0-hypothesis.
    """
    with torch.no_grad():
        # Fetch the lower and upper bound of the training space.
        lower = classifier.lower
        upper = classifier.upper
        # Check if a lower and upper bound has been specified.
        if lower is not None and upper is not None and
           (theta_next < lower).any() or (theta_next > uppper).any():
            # Likelihood-ratio is 0.
            log_likelihood_ratio = torch.tensor(float("-inf"))
        else:
            # Fetch the total number of observations.
            num_observations = observations.size(0)
            # Prepare the model parameters.
            theta = theta.repeat(num_observations)
            theta_next = theta_next.repeat(num_observations)
            # Compute the approximate likelihood-ratio.
            ratio = likelihood_to_evidence_ratio(classifier, observations, theta).log().sum()
            ratio_next = likelihood_to_evidence_ratio(classifier, observations, theta_next).log().sum()
            log_likelihood_ratio = ratio_next - ratio

    return log_likelihood_ratio


def likelihood_to_evidence_ratio(classifier, observations, thetas):
    r"""Computes the likelihood-to-evidence ratio given a set of
    observations and model parameters.
    """
    outputs = classifier(observations, thetas)
    ratios = ((1 - outputs) / (outputs + epsilon))

    return ratios
