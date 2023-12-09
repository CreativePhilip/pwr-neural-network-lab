import numpy as np

from nn_labs.nn.layers import Conv2D, MaxPool2D
from tqdm import trange


layer = Conv2D(
    28,
    28,
    2,
    kernel_count=10,
)
pool = MaxPool2D(28, 28, 2)
layer_2 = Conv2D(
    28,
    28,
    4,
    kernel_count=2,
)

x = np.ones((2, 28, 28))
for _ in trange(int(1e3)):
    out = layer.forward(x)
    print(out.shape)
    # out2 = pool.forward(out)
    # print(out2.shape)
    #
    # out3 = layer_2.forward(out2)

    break
