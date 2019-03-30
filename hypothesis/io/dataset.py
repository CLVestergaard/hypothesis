"""Hypothesis custom datasets.

Based on the PyTorch dataset abstraction.
"""

import glob
import hypothesis
import numpy as np
import os
import torch

from torch.utils.data import Dataset



class NPSimulationDataset(Dataset):
    r"""Dataset accepting a a numpy data array for model parameters and associated observations.

    Args:
        path_inputs (str): the np-datafile with the inputs.
        path_outputs (str): the np-datafile with the outputs.
        memmap_inputs (bool, optional): memory-maps the inputs datafile (default: False).
        memmap_outputs (bool, optional): memory-maps the outputs datafile (default: True).
    """

    def __init__(self, path_inputs, path_outputs, memmap_inputs=False, memmap_outputs=False):
        super(NPSplittedSimulationDataset, self).__init__()
        # Check if the specified paths exist.
        if not os.path.exists(path_inputs) or not os.path.exists(path_outputs):
            raise ValueError("Please specify a path to a inputs and outputs numpy data file.")
        # Set the memory-mapping arguments.
        if memmap_inputs:
            memmap_inputs = 'r' # Read-only
        else:
            memmap_inputs = None
        if memmap_outputs:
            memmap_outputs = 'r' # Read-only.
        else:
            memmap_outputs = None
        # Load the associated datafiles.
        self.data_inputs = np.load(path_inputs, mmap_mode=memmap_inputs)
        self.data_outputs = np.load(path_outputs, mmap_mode=memmap_outputs)

    def __getitem__(self, index):
        inputs = self.data_inputs[index]
        outputs = self.data_outputs[index]

        return torch.tensor(inputs), torch.tensor(outputs)

    def __len__(self):
        return len(self.data_inputs)


class GeneratorDataset(Dataset):
    r"""
    A dataset sampling directly from a generative model or simulator
    under a given prior.

    Args:
        model (torch.nn.Module): Generative model to sample from
        prior (torch.distributions.Distribution): Probability distribution over the inputs
        size (int): Assumed number of samples in the dataset.

    .. note::
        When iterating over multiple epochs, this dataset will have
        different contents as we will sample from the prior in
        an online fashion. Furthermore, the potential stochasticity
        of the model also contributes to this fact.
    """

    def __init__(self, model, prior, size=100000):
        super(GeneratorDataset, self).__init__()
        self.prior = prior
        self.model = model
        self.size = size

    def _sample(self):
        x = self.prior.sample()
        y = self.model(x)

        return x, y

    def __getitem__(self, index):
        x, y = self._sample()
        return x, y

    def __len__(self):
        return self.size
