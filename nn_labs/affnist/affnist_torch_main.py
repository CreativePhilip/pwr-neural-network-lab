import torch.nn as nn

from nn_labs.affnist.data_processing import load_affnist
import torch

device = "cpu"
X, Y = load_affnist(page=2)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(40**2, 40**2),
            nn.ReLU(),
            nn.Linear(40**2, 800),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
