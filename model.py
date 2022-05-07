import torch
import numpy as np
from torch import nn


class MLPModel(nn.Module):
    def __init__(self, num_classes: int,
                 embed_dim: int, activation_func=nn.ReLU):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.lin1 = nn.Linear(28 * 28, embed_dim)
        self.act = activation_func()
        self.lin2 = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.lin1.weight,
                                self.calculate_gain(activation_func))
        nn.init.constant_(self.lin1.bias, 0.0)
        nn.init.xavier_uniform_(self.lin2.weight,
                                self.calculate_gain(activation_func))
        nn.init.constant_(self.lin2.bias, 0.0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        representation = self.lin1(x)
        act_representation = self.act(representation)
        x = self.lin2(act_representation)
        return x, (representation, act_representation)

    @staticmethod
    def calculate_gain(activation_function):
        if activation_function == nn.Tanh:
            return 5/3
        elif activation_function == nn.Sigmoid:
            return 1
        elif activation_function == nn.ReLU:
            return np.sqrt(2)
        else:
            raise Warning(f'Activation function: {activation_function} not supported')
