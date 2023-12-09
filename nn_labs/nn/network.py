import numpy as np
from numpy.typing import NDArray

from nn_labs.nn.activation_functions import ReLU, BaseActivationFunction, SoftMax, Sigmoid
from nn_labs.nn.initializers import GeohotInitializer, HomogenousRandomInitializer
from nn_labs.nn.layers import DenseLayer, BaseLayer
from nn_labs.nn.perf import CategoricalCrossEntropyLoss


class NeuralNetwork:
    def __init__(
        self,
        layers: list[BaseLayer],
        activation_fns: list[BaseActivationFunction],
    ):
        self.layers: list[BaseLayer] = layers
        self.activation_fns: list[BaseActivationFunction] = activation_fns

    def forward(self, inputs: NDArray) -> NDArray:
        current_vec = inputs

        layer: BaseLayer
        activation_fn: BaseActivationFunction

        for layer, activation_fn in zip(self.layers, self.activation_fns, strict=True):
            layer.forward(current_vec)
            activation_fn.forward(layer.output)

            current_vec = activation_fn.output

        return current_vec

    def backwards(self, d_loss: NDArray):
        current_gradient = d_loss

        layer: BaseLayer
        activation_fn: BaseActivationFunction

        for layer, activation_fn in zip(
            reversed(self.layers),
            reversed(self.activation_fns),
            strict=True,
        ):
            activation_fn.backward(current_gradient)
            layer.backward(activation_fn.d_inputs)

            current_gradient = layer.d_inputs

    def __str__(self):
        value = f"Network >> \n"
        for layer, activation_fn in zip(self.layers, self.activation_fns):
            value += f"\t{layer} ->> {activation_fn}\n"

        return value


class FFDenseNetwork(NeuralNetwork):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        number_of_hidden_layers: int,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.number_of_hidden_layers = number_of_hidden_layers

        super().__init__(self._initialize_layers(), self._initialize_activation_functions())

    def _initialize_layers(self) -> list[BaseLayer]:
        layers: list[BaseLayer] = [DenseLayer(self.input_dim, self.hidden_dim)]

        for _ in range(self.number_of_hidden_layers):
            layer = DenseLayer(self.hidden_dim, self.hidden_dim, weights_initializer=GeohotInitializer)
            layers.append(layer)

        layers.append(DenseLayer(self.hidden_dim, self.output_dim, weights_initializer=HomogenousRandomInitializer))

        return layers

    def _initialize_activation_functions(self) -> list[BaseActivationFunction]:
        fns: list[BaseActivationFunction] = []

        for _ in range(self.number_of_hidden_layers + 1):
            fns.append(ReLU())

        fns.append(SoftMax())

        return fns


class AutoEncoderNetwork(NeuralNetwork):
    def __init__(self, *, input_dim: int, encoding_steps: int, encoded_dim: int):
        self.input_dim = input_dim
        self.encoding_steps = encoding_steps
        self.encoded_dim = encoded_dim

        super().__init__(
            self._initialize_layers(),
            self._initialize_activation_functions(),
        )

    def as_encoder(self) -> NeuralNetwork:
        network_mid_point = len(self.layers) // 2
        encoder_layers = self.layers[:network_mid_point]
        encoder_activation_functions = self.activation_fns[:network_mid_point]

        return NeuralNetwork(
            layers=encoder_layers,
            activation_fns=encoder_activation_functions,
        )

    def _initialize_layers(self) -> list[BaseLayer]:
        layers = []
        last_layer_size = self.input_dim
        layer_dim_step_size = (self.input_dim - self.encoded_dim) // self.encoding_steps

        for layer_number in range(self.encoding_steps):
            layers.append(
                DenseLayer(
                    last_layer_size,
                    last_layer_size - layer_dim_step_size,
                    weights_initializer=GeohotInitializer,
                )
            )
            last_layer_size -= layer_dim_step_size

        for layer_number in range(self.encoding_steps):
            layers.append(
                DenseLayer(
                    last_layer_size,
                    last_layer_size + layer_dim_step_size,
                    weights_initializer=GeohotInitializer,
                )
            )
            last_layer_size += layer_dim_step_size

        return layers

    def _initialize_activation_functions(self) -> list[BaseActivationFunction]:
        fns: list[BaseActivationFunction] = []

        for _ in range(self.encoding_steps * 2):
            fns.append(ReLU())

        fns[-1] = Sigmoid()

        return fns
