from nnfs.datasets import spiral_data

from nn_labs.nn.network import NeuralNetwork
from nn_labs.nn.optimizers import PhilipOptimizer

x, y = spiral_data(samples=100, classes=3)


network = NeuralNetwork(
    input_dim=2,
    hidden_dim=64,
    output_dim=3,
    number_of_hidden_layers=1,
)

optimizer = PhilipOptimizer(
    network,
    learning_rate=0.1,
    batch_size=x.shape[0],
    epochs=50_001,
)

optimizer.run_optimize(x, y)
