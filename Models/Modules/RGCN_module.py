import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Layers.RGCN_layer import RelationalGraphConvLayer

class RelationalGraphConvModel(nn.Module):

    
    def __init__(self, input_size, hidden_size, output_size, num_bases, num_rel, num_layer, dropout):

        super().__init__()
        self.num_layer = num_layer
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()

        for i in range(self.num_layer):
            if i == 0:
                self.layers.append(RelationalGraphConvLayer(input_size, hidden_size, num_bases, num_rel, bias=True))
            else:
                if i == self.num_layer - 1:
                    self.layers.append(RelationalGraphConvLayer(hidden_size,output_size, num_bases, num_rel, bias=True))
                else:
                    self.layers.append(RelationalGraphConvLayer(hidden_size, hidden_size, num_bases, num_rel, bias=True))

    def forward(self, A, x):

        for i, layer in enumerate(self.layers):
            x = layer(A, x)
            if i != self.num_layer - 1:
                x = F.dropout(self.relu(x), self.dropout, training=self.training)
            else:
                x = F.dropout(x, self.dropout, training=self.training)
        return x