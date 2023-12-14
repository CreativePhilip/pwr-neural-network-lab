import numpy as np

from nn_labs.nn.layers import Conv2D, MaxPool2D, Flatten
from nn_labs.nn.activation_functions import ReLU
from tqdm import trange


layer = Conv2D(
    1,
    2,
    kernel_count=16,
    padding=1,
)
pool = MaxPool2D(2, stride=2)

layer2 = Conv2D(
    16,
    2,
    kernel_count=16,
    padding=1,
)
pool2 = MaxPool2D(2, stride=2)

flat = Flatten()


x = np.ones((100, 1, 28, 28))
for _ in trange(int(1e3)):
    out = layer.forward(x)
    out2 = pool.forward(out)
    out3 = layer2.forward(out2)
    out4 = pool2.forward(out3)

    flattened = flat.forward(out4)

    print(out.shape, out2.shape)
    print(out3.shape, out4.shape)
    print(flattened.shape)

    break
