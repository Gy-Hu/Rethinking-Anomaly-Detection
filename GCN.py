import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GraphConv
import torch.nn.functional as F

class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers, dropout=0):
        super(GCNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, hidden_feats))
        for _ in range(num_layers - 2):
            self.layers.append(GraphConv(hidden_feats, hidden_feats))
        self.layers.append(GraphConv(hidden_feats, out_feats))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        x = features
        for i, layer in enumerate(self.layers):
            x = layer(g, x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x
