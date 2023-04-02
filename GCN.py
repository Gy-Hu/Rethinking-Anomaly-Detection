import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GraphConv
import torch.nn.functional as F
from dgl.nn import ChebConv, GATConv, SAGEConv

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

'''
-------------------------Some other models------------------------
'''
# WIP: may consider three different types of graph convolution layers: ChebConv, GATConv, and SAGEConv
class ChebConvModel(nn.Module):
    def __init__(self, in_feats, out_feats, k=2):
        super(ChebConvModel, self).__init__()
        self.conv = ChebConv(in_feats, out_feats, k)

    def forward(self, graph, inputs):
        return self.conv(graph, inputs)

class GATConvModel(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads=1):
        super(GATConvModel, self).__init__()
        self.conv = GATConv(in_feats, out_feats, num_heads)

    def forward(self, graph, inputs):
        return self.conv(graph, inputs).flatten(1)

class SAGEConvModel(nn.Module):
    def __init__(self, in_feats, out_feats, aggregator_type='mean'):
        super(SAGEConvModel, self).__init__()
        self.conv = SAGEConv(in_feats, out_feats, aggregator_type)

    def forward(self, graph, inputs):
        return self.conv(graph, inputs)
