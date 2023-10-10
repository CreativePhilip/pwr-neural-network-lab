import numpy as np
from numpy.typing import NDArray


class BaseActivationFunction:
    def __init__(self):
        self.input = np.array([])
        self.output = np.array([])

        self.d_inputs = np.array([])

    def forward(self, inputs: NDArray) -> NDArray:
        raise NotImplementedError

    def backward(self, d_values: NDArray) -> None:
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class ReLU(BaseActivationFunction):
    def forward(self, inputs: NDArray) -> NDArray:
        self.input = inputs
        self.output = np.maximum(0, inputs)

        return self.output

    def backward(self, d_values: NDArray) -> None:
        self.d_inputs = d_values.copy()

        # Zero the gradient for negative values
        self.d_inputs[self.input <= 0] = 0


class SoftMax(BaseActivationFunction):
    def forward(self, inputs: NDArray) -> NDArray:
        exponential_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)

        self.output = probabilities

        return self.output

    def backward(self, d_values: NDArray) -> None:
        d_inputs = np.empty_like(d_values)

        for idx, (output, d_value) in enumerate(zip(self.output, d_values)):
            # Flatten output
            output = output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(output) - np.dot(output, output.T)

            d_inputs[idx] = np.dot(jacobian_matrix, d_value)

        self.d_inputs = d_inputs
