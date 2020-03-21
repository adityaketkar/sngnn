import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class PCheby(torch.nn.Module):
    def __init__(self, num_features, n_classes, num_hidden, num_hidden_layers, dropout, activation, K=2,improved=True, bias=True):
        super(PCheby, self).__init__()
        # dropout
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Dropout(p=0.)
        #activation
        self.activation = activation
        # input layer
        self.conv_input = ChebConv(num_features, num_hidden, K=K, bias=bias)
        # Hidden layers
        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(ChebConv(num_hidden, num_hidden, K=K, bias=bias))
        # output layer
        self.conv_output = ChebConv(num_hidden, n_classes, K=K, bias=bias)

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
