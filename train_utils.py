import dgl
import torch
import torch.nn.functional as F
import numpy
import argparse
import time
from dataset import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from BWGNN import *
from sklearn.model_selection import train_test_split
from GCN import GCNModel, ChebConvModel, GATConvModel, SAGEConvModel
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import GridSearchCV
import copy

# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro',zero_division=1)
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def evaluate(model, graph, args):
    model.to(args.device)
    graph = graph.to(args.device)
    model.eval()
    with torch.no_grad():
        logits = model(graph, graph.ndata['feature'])
        print("Evaluate function: test_mask", graph.ndata['test_mask'])
        valid_logits = logits[graph.ndata['test_mask'].cpu()]
        valid_labels = graph.ndata['label'][graph.ndata['test_mask'].cpu()]
        _, indices = torch.max(valid_logits, dim=1)
        correct = torch.sum(indices == valid_labels)
        accuracy = correct.item() / len(valid_labels)
        #return roc_auc_score(valid_labels.cpu().numpy(), valid_logits[:, 1].cpu().numpy())
        return roc_auc_score(
            valid_labels.cpu().numpy(), valid_logits[:, 1].cpu().numpy()
        ), recall_score(
            valid_labels.cpu().numpy(), indices.cpu().numpy(), average='macro',zero_division=1
        ), precision_score(
            valid_labels.cpu().numpy(), indices.cpu().numpy()
        ), f1_score(
            valid_labels.cpu().numpy(), indices.cpu().numpy(), average='binary', zero_division=0
        )

def train(model, g, args):
    #print("Graph object:", g) 
    # print class names of model
    print(model.__class__.__name__)
    # train in gpu if available
    model.to(args.device)
    g = g.to(args.device)
    
    features = g.ndata['feature'].to(args.device)
    labels = g.ndata['label']
    index = list(range(len(labels)))

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index].cpu(),
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest.cpu(),
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool().to(args.device)
    val_mask = torch.zeros([len(labels)]).bool().to(args.device)
    test_mask = torch.zeros([len(labels)]).bool().to(args.device)




    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())

    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    
    print("Train function: val_mask", g.ndata['val_mask'])


    optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # Start with a smaller learning rate
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    #scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  # Adjust the step_size and gamma based on your dataset
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)

    time_start = time.time()
    for e in range(args.epoch):
        model.train()
        #logits = model(features)
        #logits = model(g, features)
        if args.choose_model == 'GCN':
            #edge_weight = torch.ones(g.number_of_edges()).to(features.device)  # Replace with your desired edge weights
            #logits = model(g, features, edge_weight)
            logits = model(g, features)
        elif args.choose_model == 'BWGNN':
            logits = model(features)
        else:
            assert False, "Choose a model from GCN or BWGNN"

        #logits = model(g, features) if args.choose_model == 'GCN' else model(features) if args.choose_model == 'BWGNN' else None
        #assert logits is not None, "Choose a model from GCN or BWGNN"
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]).to(args.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        model.eval()
        probs = logits.softmax(1)
        f1, thres = get_best_f1(labels[val_mask].cpu(), probs[val_mask].cpu())
        preds = numpy.zeros_like(labels.cpu())
        preds[probs[:, 1].cpu() > thres] = 1
        trec = recall_score(labels[test_mask.cpu()].cpu(), preds[test_mask.cpu()])
        tpre = precision_score(labels[test_mask.cpu()].cpu(), preds[test_mask.cpu()],average='binary', zero_division=0)
        tmf1 = f1_score(labels[test_mask.cpu()].cpu(), preds[test_mask.cpu()], average='macro',zero_division=1)
        tauc = roc_auc_score(labels[test_mask.cpu()].cpu(), probs[test_mask.cpu()][:, 1].cpu().detach().numpy())

        if best_f1 < f1:
            best_f1 = f1
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
        print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, loss, f1, best_f1))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100,
                                                                     final_tpre*100, final_tmf1*100, final_tauc*100))
    if args.run == 2:
        print("Model after training:", model)
        print("Graph after training:", g)
    # Save model with time stamp
    if args.save_model:
        torch.save(model.state_dict(), f'{args.model_path}/model_{int(time.time())}.pt')
    return final_tmf1, final_tauc, copy.deepcopy(model), copy.deepcopy(g)