"""Base inference module."""

import numpy as np
import torch


class Method:
    r""""""

    def infer(self, observations, **kwargs):
        raise NotImplementedError
