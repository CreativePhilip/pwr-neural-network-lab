import numpy as np
from numpy.typing import NDArray


class BaseInitializer:
    def __init__(self, shape: NDArray, scale: float = 0.0009):
        self.shape = shape
        self.scale = scale

    def initialize(self) -> NDArray:
        raise NotImplementedError


class HomogenousRandomInitializer(BaseInitializer):
    def initialize(self) -> NDArray:
        return self.scale * np.random.rand(*self.shape)


class GeohotInitializer(BaseInitializer):
    """
        This is stolen from George Hotz's livestream about nns from scratch,
         he has also blessed us with this knowledge from his jupyter



    https://github.com/geohot/ai-notebooks/blob/master/mnist_from_scratch.ipynb

    # protip: if you like accuracy like 96 not like 93, next time through the notebook, consider
    # CHAD MODE WEIGHT INIT WITH NUMPY
    # instead of virgin torch init mode
    # TODO: why is torch linear init bad?

    """

    def initialize(self) -> NDArray:
        return np.random.uniform(
            -1.0,
            1.0,
            size=(self.shape[0], self.shape[1]),
        ) / np.sqrt(self.shape[0] * self.shape[1])
