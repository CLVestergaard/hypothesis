"""Hypothesis custom datasets.

Based on the PyTorch dataset abstraction.
"""

import glob
import hypothesis
import numpy as np
import os
import torch

from torch.utils.data import Dataset



class NPZDataset(Dataset):
    r"""Dataset accepting npz data arrays.

    Args:
        path (str): the path to the npz dataset.
        inputs (str, optional): key for the model parameters (default: 'inputs')
        outputs (str, optional): key of the generated observations (default: 'outputs')
    """

    def __init__(self, path, inputs="inputs", outputs="outputs"):
        super(NPZDataset, self).__init__()
        # Check if the specified path exists.
        if not os.path.exists(path) and os.path.isdir(path):
            raise ValueError("Please specify a path to the NPZ directory.")
        # Basic dataset parameters.
        self.base = path
        self.key_inputs = inputs
        self.key_outputs = outputs
        self.block_names = self._fetch_block_names()
        # Main dataset properties.
        self.num_blocks = self._inspect_num_blocks()
        self.block_elements = self._inspect_block_elements()
        self.size = self.num_blocks * self.block_elements
        # Buffer block.
        self.buffer_block_index = 0
        self.buffer_block = self._load_block(0)

    def _fetch_block_names(self):
        return os.listdir(self.base)

    def _inspect_num_blocks(self):
        return len(glob.glob(self.base + "/*.npz"))

    def _inspect_block_elements(self):
        data = np.load(self.base + "/" + self.block_names[-1])
        return len(data[self.key_inputs])

    def _load_block(self, block_index):
        data = np.load(self.base + "/" + self.block_names[block_index])

        return data

    def __getitem__(self, index):
        # Check if the block is buffered in memory.
        block_index = int(index / self.block_elements)
        if block_index != self.buffer_block_index:
            self.buffer_block = self._load_block(block_index)
        # Load the requested data from the buffer.
        data_index = index % self.block_elements
        inputs = self.buffer_block[self.key_inputs][data_index]
        outputs = self.buffer_block[self.key_outputs][data_index]

        return inputs, outputs

    def __len__(self):
        return self.size



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
