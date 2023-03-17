"""
This file contains the interface for the base algorithm used in the package.
"""
from abc import ABC, abstractmethod
import numpy as np


class Algorithm(ABC):

    @abstractmethod
    def transform(self, time_series: np.ndarray):
        raise NotImplementedError
