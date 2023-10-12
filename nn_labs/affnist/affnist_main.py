from nn_labs.affnist.data_processing import load_affnist
from nn_labs.nn.network import NeuralNetwork
from nn_labs.nn.optimizers import PhilipOptimizer

X, Y = load_affnist(page=2)


start = 2
size = 1000

x = X
y = Y


network = NeuralNetwork(
    input_dim=1600,
    hidden_dim=800,
    output_dim=10,
    number_of_hidden_layers=1,
)

optimizer = PhilipOptimizer(
    network,
    learning_rate=1,
    batch_size=128,
    epochs=1_000,
)

optimizer.run_optimize(x, y)
