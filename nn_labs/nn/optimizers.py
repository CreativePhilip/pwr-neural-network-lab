from nn_labs.nn.network import NeuralNetwork
from numpy.typing import NDArray

from nn_labs.nn.perf import Accuracy, CategoricalCrossEntropyLoss

from tqdm import trange


class PhilipOptimizer:
    def __init__(self, network: NeuralNetwork, *, learning_rate: float, batch_size: int, epochs: int):
        self.network = network
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.accuracy_fn = Accuracy()
        self.loss_fn = CategoricalCrossEntropyLoss()

    def run_optimize(self, x: NDArray, y: NDArray):
        for epoch in trange(self.epochs):
            self._run_optimize(x, y)

    def _run_optimize(self, x: NDArray, y: NDArray):
        batch_count = x.shape[0] // self.batch_size

        for batch_no in range(batch_count):
            start_idx = batch_no * self.batch_size
            end_idx = start_idx + self.batch_size

            x_window = x[start_idx:end_idx]
            y_window = y[start_idx:end_idx]

            predictions = self.network.forward(x_window)

            acc = self.accuracy_fn.calculate(predictions, y_window)
            loss = self.loss_fn.calculate(predictions, y_window)

            print(f"Batch({batch_no}/{batch_count}) - acc[{acc:.3f}] - loss[{loss:.3f}]")

            self.loss_fn.backward(self.network.activation_fns[-1].output, y_window)

            self.network.backwards(self.loss_fn.d_inputs)

            self._apply_optimization()

    def _apply_optimization(self):
        for layer in self.network.layers:
            layer.weights = -self.learning_rate * layer.d_weights
            layer.biases = -self.learning_rate * layer.d_bias
