from nn_labs.nn.network import NeuralNetwork

import numpy as np


class L1Regularizer:
    def __init__(self, network: NeuralNetwork):
        self.network = network

    def forward(self) -> float:
        loss = 0

        for layer in self.network.layers:
            weights_scaler = max(0.0, layer.l1_weight)
            biases_scaler = max(0.0, layer.l1_bias)

            w_loss = np.sum(np.abs(layer.weights)) * weights_scaler
            b_loss = np.sum(np.abs(layer.biases)) * biases_scaler

            loss += w_loss + b_loss

        return loss


class L2Regularizer:
    def __init__(self, network: NeuralNetwork):
        self.network = network

    def forward(self) -> float:
        loss = 0

        for layer in self.network.layers:
            weights_scaler = max(0.0, layer.l2_weight)
            biases_scaler = max(0.0, layer.l2_bias)

            w_loss = np.sum(layer.weights**2) * weights_scaler
            b_loss = np.sum(layer.biases**2) * biases_scaler

            loss += w_loss + b_loss

        return loss
