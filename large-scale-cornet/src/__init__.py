# src/__init__.py

"""
ASD Face Recognition Modeling with CNN
A neurocomputational basis of face recognition changes in ASD: E/I balance, internal noise, and weak neural representations
"""

__version__ = "1.0.0"

# Import main modules for easier access
from . import models
from . import data
from . import analysis
from . import utils

__all__ = [
    'models',
    'data',
    'analysis',
    'utils',
]

