import torch
import numpy as np
from sklearn.base import BaseEstimator
from GCN import GCNModel
from train_utils import train, evaluate

class GCNEstimator(BaseEstimator):
    def __init__(self, hid_dim=64, num_layers=2, **kwargs):
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.kwargs = kwargs
        self.model = None
        self.graph = None
        
    def fit(self, graph, args):
        self.graph = graph
        in_feats = graph.ndata['feature'].shape[1]
        num_classes = 2

        self.model = GCNModel(in_feats, self.hid_dim, num_classes, self.num_layers)
        _,_,self.model, self.graph = train(self.model, self.graph, args)  # Pass the args variable to the train function
        
    def score(self, args):
        print("Model during scoring:", self.model)
        print("Graph during scoring:", self.graph)
        return evaluate(self.model, self.graph, args)

