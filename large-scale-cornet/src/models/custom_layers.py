# src/models/custom_layers.py

"""
Custom neural network layers for modeling ASD characteristics
"""

import torch
import torch.nn as nn


class EIRectifiedLinear(nn.Module):
    """
    PyTorch custom activation for E/I imbalance and internal noise.

    Implements: y = alpha * ReLU(x) + Noise

    Parameters:
        alpha: E/I gain modulation (slope)
        noise_std: standard deviation of Gaussian noise
    """
    def __init__(self, alpha=1.0, noise_std=0.0):
        super().__init__()
        self.alpha = alpha
        self.noise_std = noise_std

    def forward(self, x):
        # Apply E/I slope
        out = torch.nn.functional.relu(x) * self.alpha

        # Inject internal noise
        if self.noise_std > 0:
            noise = torch.randn_like(out) * self.noise_std
            out = out + noise

        return out
