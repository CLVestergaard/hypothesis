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

    def __init__(self, path_inputs, path_outputs, path_targets=None,
                 memmap_inputs=False, memmap_outputs=True, memmap_targets=False,
                 transform_inputs=None, transform_outputs=None, transform_targets=None):
        super(NPSimulationDataset, self).__init__()
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
        if memmap_targets:
            memmap_targets = 'r' # Read-only
        else:
            memmap_targets = none
        # Load the associated datafiles.
        self.data_inputs = np.load(path_inputs, mmap_mode=memmap_inputs)
        self.data_outputs = np.load(path_outputs, mmap_mode=memmap_outputs)
        if path_targets is not None:
            self.data_targets = np.load(path_targets, mmap_mode=memmap_targets)
        else:
            self.data_targets = None
        # Set the transforms.
        self.transform_inputs = transfrom_inputs
        self.transform_outputs = transform_outputs
        self.transform_targets = transform_targets

    def __getitem__(self, index):
        # Load the inputs.
        inputs = self.data_inputs[index]
        if self.transform_inputs is not None:
            inputs = self.transform_inputs(inputs)
        outputs = self.data_outputs[index]
        if self.transform_outputs is not None:
            outputs = self.data_outputs[index]
        if data_targets is not None:
            targets = self.data_targets[index]
            if self.transform_targets is not None:
                targets = self.transform_targets(targets)
            return inputs, outputs, targets
        return inputs, outputs

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
