import numpy as np
from numpy.typing import NDArray


class BaseInitializer:
    def __init__(self, shape: NDArray):
        self.shape = shape

    def initialize(self) -> NDArray:
        raise NotImplementedError


class HomogenousRandomInitializer(BaseInitializer):
    def initialize(self) -> NDArray:
        return 0.09 * np.random.rand(*self.shape)
