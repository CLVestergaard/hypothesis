"""
Hypothesis is a Python module for likelihood-free inference.
"""

__version__ = "0.0.3"
__author__ = [
    "Joeri Hermans",
    "Volodimir Begy"]
__email__ = [
    "joeri.hermans@doct.uliege.be",
    "volodmir.begy@cern.ch"]

import torch

# Check the availablity of a GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hypothesis hooks.
from .engine import hook
from .engine.hook import call_hook as hook_call
from .engine.hook import clear_hook as hook_clear
from .engine.hook import clear_hook as hooks_clear
from .engine.hook import register_hook as hook_register
from .engine.hook import tags
