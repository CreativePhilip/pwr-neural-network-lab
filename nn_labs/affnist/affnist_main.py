from nn_labs.affnist.data_processing import load_affnist
from nn_labs.nn.network import NeuralNetwork
from nn_labs.nn.optimizers import PhilipOptimizer
import matplotlib.pyplot as plt

X, Y = load_affnist(page=[1, 2, 3, 4])


start = 2
size = 1000

x = X
y = Y


network = NeuralNetwork(
    input_dim=1600,
    hidden_dim=800,
    output_dim=10,
    number_of_hidden_layers=5,
)

optimizer = PhilipOptimizer(
    network,
    learning_rate=0.1,
    batch_size=64,
    epochs=1_000,
)

try:
    optimizer.run_optimize(x, y)
except KeyboardInterrupt:
    pass


x = range(len(optimizer.losses))


plt.ylim((-0.1, 3))
plt.plot(x, optimizer.losses)
plt.plot(x, optimizer.accs)
