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
            # add bias
            self.output[:, kernel_idx, :, :] = np.tensordot(stride_view, kernel, axes=([1, 4, 5], [0, 1, 2]))

        self.output = np.maximum(0, self.output)

        return self.output

    def backward(self, d_values: NDArray) -> None:
        d_values = d_values * (self.output > 0)

        bs, input_channels, input_h, input_w = self.input.shape
        _, _, output_h, output_w = d_values.shape
        in_strides = self.input.strides

        self.d_weights = np.zeros_like(self.kernel)
        self.d_bias = np.zeros_like(self.biases)
        self.d_inputs = np.zeros_like(self.input)

        stride_view_kernel = as_strided(
            self.input,
            shape=(bs, input_channels, output_h, output_w, self.kernel_size, self.kernel_size),
            strides=(*in_strides[:2], in_strides[2] * self.stride, in_strides[3] * self.stride, *in_strides[2:]),
        )

        for idx in range(self.kernel_count):
            kern_d_values = d_values[:, idx, :, :].reshape(bs, 1, output_h, output_w, 1, 1)
            self.d_weights[idx] = np.sum(stride_view_kernel * kern_d_values, axis=(0, 2, 3))
            self.d_bias[idx] = np.sum(d_values[:, idx, :, :], axis=(0, 1, 2))

        inv_kernel = self.kernel[:, :, ::-1, ::-1]
        d_values_with_pad = np.pad(
            d_values,
            (
                (0, 0),
                (0, 0),
                (self.kernel_size - 1, self.kernel_size - 1),
                (self.kernel_size - 1, self.kernel_size - 1),
            ),
            mode="constant",
        )
        d_values_with_pad_strides = d_values_with_pad.strides

        out_stride = as_strided(
            d_values_with_pad,
            shape=(bs, self.kernel_count, input_h, input_w, self.kernel_size, self.kernel_size),
            strides=(
                *d_values_with_pad_strides[:2],
                d_values_with_pad_strides[2] * self.stride,
                d_values_with_pad_strides[3] * self.stride,
                *d_values_with_pad_strides[2:],
            ),
        )

        print(out_stride.shape)
        print(inv_kernel.shape)

        self.d_inputs = np.tensordot(out_stride, inv_kernel, axes=((1, 4, 5), (0, 2, 3)))

        print(self.d_inputs.shape)
        print("-" * 69)
        if self.padding != 0:
            self.d_inputs = self.d_inputs[:, :, self.padding : -self.padding, self.padding : -self.padding]


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
                in_strides[2] * self.stride,
                in_strides[3] * self.stride,
                *in_strides[2:],
            ),
        )

        self.output = np.max(stride_view, axis=(4, 5))

        return self.output

    def backward(self, d_values: NDArray) -> None:
        bs, kernels, input_h, input_w = self.input.shape
        in_strides = self.input.strides

        output_h = d_values.shape[2]
        output_w = d_values.shape[3]

        stride_view = as_strided(
            self.input,
            shape=(bs, kernels, output_h, output_w, self.kernel_size, self.kernel_size),
            strides=(
                *in_strides[:2],
                in_strides[2] * self.stride,
                in_strides[3] * self.stride,
                *in_strides[2:],
            ),
        )

        self.d_inputs = np.zeros_like(self.input)

        for h in range(output_h):
            for w in range(output_w):
                max_indices = np.argmax(stride_view[:, :, h, w].reshape(bs, kernels, -1), axis=2)
                max_coords = np.unravel_index(max_indices, (self.kernel_size, self.kernel_size))

                for batch in range(bs):
                    for kernel in range(kernels):
                        coord_1 = max_coords[0][batch, kernel]
                        coord_2 = max_coords[1][batch, kernel]

                        h_start = h * self.stride
                        w_start = w * self.stride

                        local_err = d_values[batch, kernel, h, w]
                        self.d_inputs[batch, kernel, h_start + coord_1, w_start + coord_2] += local_err


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
