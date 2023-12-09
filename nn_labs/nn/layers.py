from typing import Type

import numpy as np
from numba import njit
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
        out_channels: int,
        kernel_size: int,
        kernel_count: int,
        stride=1,
        padding=0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        self.kernel_count = kernel_count
        self.kernel = (
            GeohotInitializer(np.array([kernel_size * kernel_size, kernel_count]))
            .initialize()
            .reshape((kernel_size, kernel_size, -1))
        )

        self.input = None

    def forward(self, inputs: NDArray) -> NDArray:
        self.input = inputs
        view_shape = tuple(np.subtract(inputs[0].shape[:2], self.kernel_size) + 1) + self.kernel_size

        strides = self._conv2d__generate_strides(inputs, view_shape, self.kernel_size)
        initial_strides_shape = strides.shape

        strides = strides.reshape((*initial_strides_shape[:3], initial_strides_shape[4] * 2))
        strides = np.tile(strides, self.kernel_count)
        strides = strides.reshape(
            (
                *initial_strides_shape[:3],
                *self.kernel_size,
                self.kernel_count,
            )
        )

        out = strides * self.kernel
        self.output = np.mean(out, axis=(1, 2))

        return self.output

    def backward(self, d_values: NDArray) -> None:
        pass

    @staticmethod
    def _conv2d__generate_strides(
        inputs: NDArray,
        view_shape: tuple[int, ...],
        kernel_size: tuple[int, int],
        stride_offset: int = 1,
    ) -> NDArray:
        out_strides = np.zeros((inputs.shape[0], view_shape[0], view_shape[1], *kernel_size))

        for idx, image in enumerate(inputs):
            arr_view = as_strided(
                image,
                shape=view_shape,
                strides=(image.strides[0] * stride_offset, image.strides[1] * stride_offset) + image.strides,
            )

            out_strides[idx] = arr_view

        return out_strides


class MaxPool2D(BaseLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        padding=0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        self.input = None

    def forward(self, inputs: NDArray) -> NDArray:
        self.input = inputs

        view_shape = (
            (
                inputs.shape[1] // self.kernel_size[0],
                inputs.shape[2] // self.kernel_size[0],
            )
            + self.kernel_size
            + (inputs.shape[-1],)
        )

        strides = self._generate_strides(
            inputs,
            view_shape,
            stride_offset=self.stride,
        )

        return np.max(strides, axis=(-1, -2))

    @staticmethod
    def _generate_strides(
        inputs: NDArray,
        view_shape: tuple[int, ...],
        stride_offset: int = 1,
    ) -> NDArray:
        out_strides = np.zeros((inputs.shape[0], *view_shape))

        for idx, image in enumerate(inputs):
            arr_view = as_strided(
                image,
                shape=view_shape,
                strides=(image.strides[0] * stride_offset, image.strides[1] * stride_offset) + image.strides,
            )

            out_strides[idx] = arr_view

        return out_strides
