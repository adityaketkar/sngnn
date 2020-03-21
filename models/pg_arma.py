import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv

class PARMA(torch.nn.Module):
    def __init__(self, num_features, n_classes, num_hidden, num_hidden_layers,activation, dropout=0, num_stacks=1, num_layers=1, shared_weights=False, bias=True):
        super(PARMA, self).__init__()
        # dropout
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Dropout(p=0.)
        #activation
        self.activation = activation
        # input layer
        self.conv_input = ARMAConv(num_features, num_hidden, num_stacks=num_stacks, num_layers=num_layers, shared_weights=shared_weights, bias=bias)
        # Hidden layers
        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(ARMAConv(num_hidden, num_hidden, num_stacks=num_stacks, num_layers=num_layers, shared_weights=shared_weights, bias=bias))
        # output layer
        self.conv_output = ARMAConv(num_hidden, n_classes, num_stacks=num_stacks, num_layers=num_layers, shared_weights=shared_weights, bias=bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.dropout(x)
        x = self.conv_input(x, edge_index)
        x = self.activation(x)
        for layer in self.layers:
             x = layer(x, edge_index)
             x = self.activation(x)
             x = self.dropout(x)
        x = self.conv_output(x, edge_index)
        return torch.tanh(x)
