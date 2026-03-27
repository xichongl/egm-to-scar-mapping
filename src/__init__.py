"""
EGM to Scar Mapping: Machine Learning Framework for Cardiac EGM Analysis

This package provides tools for:
1. Loading and preprocessing cardiac electrogram (EGM) data
2. Training neural network models (CNN-STFT and Transformer-based)
3. Evaluating model performance on scar localization tasks
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from . import config

__all__ = ["config"]
