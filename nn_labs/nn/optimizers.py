from nn_labs.nn.network import NeuralNetwork
from numpy.typing import NDArray

from nn_labs.nn.perf import Accuracy, CategoricalCrossEntropyLoss
from nn_labs.nn.layers import DenseLayer, Conv2D

from tqdm import trange


class PhilipOptimizer:
    def __init__(self, network: NeuralNetwork, *, learning_rate: float, batch_size: int, epochs: int):
        self.network = network
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.accuracy_fn = Accuracy()
        self.loss_fn = CategoricalCrossEntropyLoss()

        self.losses = []
        self.accs = []

    def run_optimize(self, x: NDArray, y: NDArray):
        for epoch in range(self.epochs):
            self._run_optimize(x, y, epoch)

    def _run_optimize(self, x: NDArray, y: NDArray, epoch: int):
        batch_count = x.shape[0] // self.batch_size

        for batch_no in trange(batch_count):
            start_idx = batch_no * self.batch_size
            end_idx = start_idx + self.batch_size

            x_window = x[start_idx:end_idx]
            y_window = y[start_idx:end_idx]

            predictions = self.network.forward(x_window)

            acc = self.accuracy_fn.calculate(predictions, y_window)
            loss = self.loss_fn.calculate(predictions, y_window)

            self.loss_fn.backward(predictions, y_window)

            self.network.backwards(self.loss_fn.d_inputs)

            self._apply_optimization()

        print(f"{epoch}  --  acc[{acc:.3f}] - loss[{loss:.3f}]")
        self.losses.append(loss)
        self.accs.append(acc)

    def _apply_optimization(self):
        for layer in self.network.layers:
            layer.weights += -self.learning_rate * layer.d_weights
            layer.biases += -self.learning_rate * layer.d_bias


class PhilipOptimizer2:
    def __init__(self, network: NeuralNetwork, *, learning_rate: float):
        self.network = network
        self.learning_rate = learning_rate

    def step(self):
        for layer in self.network.layers:
            if isinstance(layer, DenseLayer):
                layer.weights += -self.learning_rate * layer.d_weights
                layer.biases += -self.learning_rate * layer.d_bias
            elif isinstance(layer, Conv2D):
                layer.kernel += -self.learning_rate * layer.d_weights
                layer.biases += -self.learning_rate * layer.d_bias
