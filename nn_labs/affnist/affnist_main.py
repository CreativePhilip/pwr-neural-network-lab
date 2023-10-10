import numpy as np

from nn_labs.affnist.data_processing import load_affnist
from nn_labs.nn.network import NeuralNetwork
from nn_labs.nn.perf import CategoricalCrossEntropyLoss, Accuracy

X, Y = load_affnist(page=2)

print(Y[0:200])


start = 2
size = 10

x = X[start:size]
y = Y[start:size]


network = NeuralNetwork(
    input_dim=1600,
    hidden_dim=3200,
    output_dim=10,
    number_of_hidden_layers=1,
)
loss_fn = CategoricalCrossEntropyLoss()
acc_fn = Accuracy()

print(network)

output = network.forward(x)
loss = loss_fn.calculate(output, y)
acc = acc_fn.calculate(output, y)


print(f"Loss: {loss}")
print(f"Accuracy: {acc}")

print(output)


loss_fn.backward(output, y)
network.activation_fns[-1].backward(loss_fn.d_inputs)
