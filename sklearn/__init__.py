"""
Stub mínimo de scikit-learn para que librosa importe sin instalar sklearn real.
Este proyecto NO usa scikit-learn.
"""
from . import decomposition  # noqa: F401

__all__ = ["decomposition"]
__version__ = "0.0-stub"
