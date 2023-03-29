import dgl
import torch
import torch.nn.functional as F
import numpy as np
import itertools
import argparse
import time
from dataset import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from BWGNN import *
from sklearn.model_selection import train_test_split
from GCN import GCNModel
from torch.optim.lr_scheduler import StepLR
from gcn_estimator import GCNEstimator
from sklearn.model_selection import GridSearchCV
from train_utils import train

def hyperparameter_tuning(param_grid, graph, args, cv=3):
    hid_dim_values = param_grid['hid_dim']
    num_layers_values = param_grid['num_layers']

    best_auc = -np.inf
    best_params = None

    for hid_dim, num_layers in itertools.product(hid_dim_values, num_layers_values):
        print(f"Trying hid_dim: {hid_dim}, num_layers: {num_layers}")

        auc_scores = []
        for _ in range(cv):
            estimator = GCNEstimator(hid_dim=hid_dim, num_layers=num_layers)
            estimator.fit(graph, args)
            auc = estimator.score(graph, args)
            auc_scores.append(auc)
        

        avg_auc = np.mean(auc_scores)
        print(f"Average AUC: {avg_auc} in {cv} runs, with hid_dim: {hid_dim}, num_layers: {num_layers}")

        if avg_auc > best_auc:
            best_auc = avg_auc
            best_params = {'hid_dim': hid_dim, 'num_layers': num_layers}

    return best_params, best_auc


def weights_init(m):
    if isinstance(m, (nn.Linear, GraphConv)):
        torch.nn.init.xavier_uniform_(m.weight)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="tfinance",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")
    parser.add_argument("--knn-reconstruct-graph", action='store_true',help="Reconstruct graph using KNN algorithm")
    parser.add_argument("--knn-reconstruct-graph-approximate", action='store_true', help="Reconstruct graph using approximate KNN algorithm (Fast verision)")
    parser.add_argument("--alpha",type=float, default=1.0, help="alpha of thershold in KNN algorithm")

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    h_feats = args.hid_dim
    graph = Dataset(name=dataset_name,knn_reconstruct=args.knn_reconstruct_graph).graph\
        if args.knn_reconstruct_graph \
            else Dataset(name=dataset_name,knn_reconstruct_approximate=args.knn_reconstruct_graph_approximate).graph \
            if args.knn_reconstruct_graph_approximate \
                else Dataset(name=dataset_name).graph
    in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2

    if args.run == 1:
        #model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
        model = GCNModel(in_feats, h_feats, num_classes, args.num_layers)
        model.apply(weights_init)
        train(model, graph, args)
    elif args.run == 2: # hyperparameter tuning
        param_grid = {
            'hid_dim': [32, 64, 128],
            'num_layers': [3, 4, 5],
        }

        best_params, best_auc = hyperparameter_tuning(param_grid, graph, args)

        print("Best parameters found: ", best_params)
        print("Highest AUC found: ", best_auc)
    else:
        final_mf1s, final_aucs = [], []
        for _ in range(args.run):
            #model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
            model = GCNModel(in_feats, h_feats, num_classes, args.num_layers)
            model.apply(weights_init)
            mf1, auc = train(model, graph, args)
            final_mf1s.append(mf1)
            final_aucs.append(auc)
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        print('MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_mf1s),
                                                                                            100 * np.std(final_mf1s),
                                                               100 * np.mean(final_aucs), 100 * np.std(final_aucs)))

    