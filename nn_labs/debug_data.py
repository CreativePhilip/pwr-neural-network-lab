import numpy as np

from nn_labs.nn.network import NeuralNetwork
from nn_labs.nn.perf import CategoricalCrossEntropyLoss, Accuracy

x = np.array(
    [
        [0.2, 0.3, 0.4],
        [0.1, 0.3, 0.2],
    ]
)

y = np.array([0, 2])


network = NeuralNetwork(
    input_dim=3,
    hidden_dim=100,
    output_dim=3,
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
print()


loss_fn.backward(output, y)

print(loss_fn.d_inputs)
print()

network.activation_fns[-1].backward(loss_fn.d_inputs)

print(network.activation_fns[-1].d_inputs)
print()
