import math
from typing import Type

import numpy as np
from numpy.typing import NDArray
from numpy.lib.stride_tricks import as_strided

from nn_labs.nn.initializers import BaseInitializer, HomogenousRandomInitializer, GeohotInitializer


class BaseLayer:
    weights: NDArray
    biases: NDArray

    output: NDArray

    d_weights: NDArray
    d_bias: NDArray
    d_inputs: NDArray

    l1_weight: float
    l1_bias: float
    l2_weight: float
    l2_bias: float

    def forward(self, inputs: NDArray) -> NDArray:
        raise NotImplementedError

    def backward(self, d_values: NDArray) -> None:
        raise NotImplementedError


class DenseLayer(BaseLayer):
    def __init__(
        self,
        input_count: int,
        neurons_count: int,
        weights_initializer: Type[BaseInitializer] = HomogenousRandomInitializer,
        l1_weight: float = 0.0,
        l1_bias: float = 0.0,
        l2_weight: float = 0.0,
        l2_bias: float = 0.0,
    ):
        self.neurons_count = neurons_count
        self.input_count = input_count
        self.initializer = weights_initializer(np.array([self.input_count, self.neurons_count]))

        self.weights = self.initializer.initialize()
        self.biases = np.zeros((1, self.neurons_count))

        self.input = np.zeros((1, self.input_count))
        self.output = np.zeros((1, self.neurons_count))

        self.d_weights = np.zeros((1, self.neurons_count))
        self.d_bias = np.zeros((1, self.neurons_count))
        self.d_inputs = np.zeros((1, self.input_count))

        self.l1_weight = l1_weight
        self.l1_bias = l1_bias
        self.l2_weight = l2_weight
        self.l2_bias = l2_bias

    def forward(self, inputs: NDArray) -> NDArray:
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

        return self.output

    def backward(self, d_values: NDArray) -> None:
        # Set gradient on layer params
        self.d_weights = np.dot(self.input.T, d_values)
        self.d_bias = np.sum(d_values, axis=0, keepdims=True)

        # Set gradient on inputs
        self.d_inputs = np.dot(d_values, self.weights.T)

        if self.l1_weight > 0:
            d_l1 = np.ones_like(self.weights)
            d_l1[self.weights < 0] = -1
            self.d_weights += self.l1_weight * d_l1

        if self.l2_weight > 0:
            self.d_weights += 2 * self.l2_weight * self.weights

        if self.l1_bias > 0:
            d_l1 = np.ones_like(self.biases)
            d_l1[self.biases] = -1
            self.d_bias += self.l1_bias * d_l1

        if self.l2_bias > 0:
            self.d_bias += 2**self.l2_weight * self.biases

    def __str__(self):
        return f"DenseLayer({self.input_count} x {self.neurons_count})"


class Conv2D(BaseLayer):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        kernel_count: int,
        stride=1,
        padding=0,
    ):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel_count = kernel_count
        self.kernel = np.random.randn(
            self.kernel_count, self.in_channels, self.kernel_size, self.kernel_size
        ) * np.sqrt(2.0 / (self.in_channels * self.kernel_size * self.kernel_size))
        self.biases = np.zeros(self.kernel_count)

        self.input = None

    def forward(self, inputs: NDArray) -> NDArray:
        self.input = np.pad(
            inputs,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode="constant",
        )
        in_strides = self.input.strides

        bs, in_channels, input_h, input_w = self.input.shape
        output_h = math.ceil((input_h - self.kernel_size) / self.stride)
        output_w = math.ceil(input_w - self.kernel_size / self.stride)

        self.output = np.zeros((bs, self.kernel_count, output_h, output_w))

        stride_view = as_strided(
            self.input,
            shape=(bs, in_channels, output_h, output_w, self.kernel_size, self.kernel_size),
            strides=(*in_strides[:2], in_strides[2] * self.stride, in_strides[3] * self.stride, *in_strides[2:]),
        )

        for kernel_idx, kernel in enumerate(self.kernel):
            self.output[:, kernel_idx, :, :] = np.tensordot(stride_view, kernel, axes=([1, 4, 5], [0, 1, 2]))

        self.output = np.maximum(0, self.output)

        return self.output

    def backward(self, d_values: NDArray) -> None:
        pass


class MaxPool2D(BaseLayer):
    def __init__(
        self,
        kernel_size: int,
        stride=1,
    ):
        self.kernel_size = kernel_size
        self.stride = stride

        self.input = None

    def forward(self, inputs: NDArray) -> NDArray:
        self.input = inputs
        in_strides = self.input.strides

        bs, kernels, input_h, input_w = inputs.shape

        output_h = math.ceil((input_h - self.kernel_size) / self.stride)
        output_w = math.ceil((input_w - self.kernel_size) / self.stride)

        stride_view = as_strided(
            self.input,
            shape=(bs, kernels, output_h, output_w, self.kernel_size, self.kernel_size),
            strides=(
                *in_strides[:2],
                in_strides[2] * self.kernel_size,
                in_strides[3] * self.kernel_size,
                *in_strides[2:],
            ),
        )

        self.output = np.max(stride_view, axis=(4, 5))

        return self.output


class Flatten(BaseLayer):
    def __init__(self):
        self.input = None
        self.in_shape_cache = None

    def forward(self, inputs: NDArray) -> NDArray:
        self.input = inputs
        self.in_shape_cache = inputs.shape
        self.output = inputs.reshape(inputs.shape[0], -1)

        return self.output

    def backward(self, d_values: NDArray) -> None:
        self.d_inputs = d_values.reshape(self.in_shape_cache)
