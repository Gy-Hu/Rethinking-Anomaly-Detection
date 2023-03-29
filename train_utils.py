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
from GCN import GCNModel
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import GridSearchCV

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
    model.eval()
    with torch.no_grad():
        logits = model(graph, graph.ndata['feature'])
        valid_logits = logits[graph.ndata['val_mask']]
        valid_labels = graph.ndata['label'][graph.ndata['val_mask']]
        _, indices = torch.max(valid_logits, dim=1)
        correct = torch.sum(indices == valid_labels)
        accuracy = correct.item() / len(valid_labels)
        return roc_auc_score(
            valid_labels.cpu().numpy(), valid_logits[:, 1].cpu().numpy()
        )

def train(model, g, args):
    features = g.ndata['feature']
    labels = g.ndata['label']
    index = list(range(len(labels)))
    if args.dataset == 'amazon':
        index = list(range(3305, len(labels)))

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()




    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())

    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask

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
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        model.eval()
        probs = logits.softmax(1)
        f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
        preds = numpy.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        trec = recall_score(labels[test_mask], preds[test_mask])
        tpre = precision_score(labels[test_mask], preds[test_mask],average='binary', zero_division=0)
        tmf1 = f1_score(labels[test_mask], preds[test_mask], average='macro',zero_division=1)
        tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())

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
    # Save model with time stamp
    if args.save_model:
        torch.save(model.state_dict(), f'{args.model_path}/model_{int(time.time())}.pt')
    return final_tmf1, final_tauc