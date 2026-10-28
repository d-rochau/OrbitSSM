
from .layers.tails import *
from .layers import tails as _canonical

def __getattr__(name):
    return getattr(_canonical, name)
