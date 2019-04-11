r"""Defines a `parameterized classifier` used in likelihood ratio estimation.

Conventions:
    -
"""

import torch
import hypothesis

from hypothesis.inference.approximate_likelihood_ratio import likelihood_to_evidence_ratio
from hypothesis.inference.approximate_likelihood_ratio import log_likelihood_ratio
from hypothesis.util.common import tensor_initialize



class AbstractParameterizedClassifier(torch.nn.Module):
    r"""Abstract parameterized classifier.

    This class provides a base-definition of an abstract
    parameterized classifier, and should be used when
    """

    def __init__(self, lower=None, upper=None, reduction="mean"):
        super(AbstractParameterizedClassifier, self).__init__()
        # Define the lower bound of the parameterized classifier.
        if lower is not None:
            lower = tensor_initialize(lower).float()
            lower = lower.to(hypothesis.device)
        # Define the upper bound of the parameterized classifier.
        if upper is not None:
            upper = tensor_initialize(upper).float()
            upper = upper.to(hypothesis.device)
        # Set the processed bounds.
        self.lower = lower
        self.upper = upper
        # Set the reduction model.
        if reduction == "mean":
            self.reduce = self._reduce_mean
        else:
            self.reduce = self._reduce_sum

    def _reduce_mean(x):
        return x.mean(dim=0)

    def _reduce_sum(x):
        return x.sum(dim=0)

    def has_bounds(self):
        r"""Checks if the parameterized classifier has proper lower
        and upper bounds.
        """
        return self.lower is not None and self.upper is not None

    def grad_log_likelihood(self, observations, theta):
        r"""Returns the gradient of the log likelihood with respect
        to the model parameter theta.

        Args:
            observations (tensor): set of observations.
            theta (tensor): model parameter to compute the grad against.
            reduction (str, optional): specified the reduction of the gradients.
            By default this is 'mean'.
        """
        with torch.no_grad():
            num_observations = observations.size(0)
            thetas = thetas.repeat(num_observations)
        # Compute the gradient with respect to theta.
        thetas.requires_grad = True
        ratios = self.likelihood_to_evidence_ratio(observations, thetas)
        torch.autograd.backward(ratios.split(1), None)
        with torch.no_grad():
            gradients = (-thetas.grad / (ratios + epsilon))
            gradient = self.reduce(gradients)

        return gradient

    def log_likelihood_to_evidence_ratio(self, observations, theta):
        r"""Computes log p(x|theta) / p(x) for the specified observations.

        Args:
            observations (tensor): set of observations.
            theta (tensor): model parameter.
        """
        n = observations.size(0)
        thetas = theta.repeat(n)
        return likelihood_to_evidence_ratio(self, observations, thetas).log().sum()

    def likelihood_to_evidence_ratio(self, observations, theta):
        r"""Computes log p(x|theta) / p(x) for the specified observations.

        Args:
            observations (tensor): set of observations.
            theta (tensor): model parameter.
        """
        return (-self.log_likelihood_to_evidence_ratio(observations, theta)).exp()

    def log_likelihood_ratio(self, observations, theta, theta_next):
        r"""Computes p(x|theta_next) / p(x|theta) for the
        specified observations.

        Args:
            observations (tensor): set of observations.
            theta (tensor): the model parameter of the alternative hypothesis.
            theta_next (tensor): the model parameter of the 0-hypothesis.
        """
        return log_likelihood_ratio(self, observations, theta, theta_next)

    def forward(self, observations, thetas):
        r""""""
        raise NotImplementedError


class ParameterizedClassifier(AbstractParameterizedClassifier):

    def __init__(self, classifier, lower=None, upper=None):
        super(ParameterizedClassifier, self).__init__(lower, upper)
        self.classifier = classifier

    def parameters(self):
        r"""Returns the trainable parameters of the parameterized classifier."""
        return self.classifier.parameters()

    def forward(self, observations, thetas):
        num_observations = observations.size(0)
        observations = observations.view(num_observations, -1)
        thetas = thetas.view(num_observations, -1)
        inputs = torch.cat([observations, thetas], dim=1)

        return self.classifier(inputs)


class ParameterizedClassifierEnsemble(AbstractParameterizedClassifier):

    def __init__(self, classifiers, lower=None, upper=None):
        super(ParameterizedClassifierEnsemble, self).__init__(lower, upper)
        self.classifiers = classifiers

    def forward_all(self, observations, thetas):
        outputs = []
        for classifier in self.classifiers:
            out = classifier(observations, thetas).view(-1, 1)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=1)

        return outputs

    def forward(self, observations, thetas):
        return self.forward_all(observations, thetas).mean(dim=1).view(-1, 1)
