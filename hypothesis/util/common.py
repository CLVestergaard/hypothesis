"""Common utility method in Hypothesis.
"""

import numpy as np
import torch



def load_argument(key, **kwargs, default=None):
    r"""Loads the specified keys from kwargs.

    This procedure returns 'None' when the key or its
    argument has not been specified.

    Args:
        key (str): variable name.
        **kwargs (dict): dictionary of possible arguments.
        default: the default value to set. Default = None.
    """
    value = default

    # Check if the specified key is present.
    if key in kwargs.keys():
        value = kwargs[key]

    return value


def tensor_initialize(data):
    r"""Accepts a list, numpy arrow or PyTorch tensor, and
    converts the structure into an indepedent PyTorch tensor.

    Args:
        data (object): the data object to initialize the tensor with.
    """
    if type(data) == torch.Tensor:
        data = data.clone()
    else:
        data = torch.tensor(data)

    return data
