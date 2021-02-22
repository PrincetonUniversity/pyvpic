"""pyvpic

Python tools for accessing and analyzing VPIC data.
"""
from .data import *
from .readers import ParticleReader

__all__ = (
    'open',
    'ParticleReader'
)
