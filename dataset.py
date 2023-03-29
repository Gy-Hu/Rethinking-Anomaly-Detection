import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
import dgl
import numpy as np
import torch
from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
from annoy import AnnoyIndex

class Dataset:
    def __init__(self, name='tfinance', homo=True, anomaly_alpha=None, anomaly_std=None, knn_reconstruct=False, knn_reconstruct_approximate=False, k=3, threshold=None, alpha=0.8):
        self.name = name
        graph = None
        if name == 'tfinance':
            graph, label_dict = load_graphs('dataset/tfinance')
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)
            
            graph = self.reconstruct_graph_knn(graph, k, threshold, alpha) \
                if knn_reconstruct \
                    else  self.reconstruct_graph_approximate_knn(graph, k, threshold, alpha)\
                        if knn_reconstruct_approximate else graph

            assert graph is not None, "re-build using knn not worked"
            if anomaly_std:
                graph, label_dict = load_graphs('dataset/tfinance')
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = graph.ndata['label'][:,1].nonzero().squeeze(1)
                feat = (feat-np.average(feat,0)) / np.std(feat,0)
                feat[anomaly_id] = anomaly_std * feat[anomaly_id]
                graph.ndata['feature'] = torch.tensor(feat)
                graph.ndata['label'] = graph.ndata['label'].argmax(1)

            if anomaly_alpha:
                graph, label_dict = load_graphs('dataset/tfinance')
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = list(graph.ndata['label'][:, 1].nonzero().squeeze(1))
                normal_id = list(graph.ndata['label'][:, 0].nonzero().squeeze(1))
                label = graph.ndata['label'].argmax(1)
                diff = anomaly_alpha * len(label) - len(anomaly_id)
                import random
                new_id = random.sample(normal_id, int(diff))
                # new_id = random.sample(anomaly_id, int(diff))
                for idx in new_id:
                    aid = random.choice(anomaly_id)
                    # aid = random.choice(normal_id)
                    feat[idx] = feat[aid]
                    label[idx] = 1  # 0

        else:
            print('no such dataset')
            exit(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph

    def reconstruct_graph_knn(self, graph, k, threshold, alpha):
        # Convert DGL graph to NetworkX graph
        G_nx = dgl.to_networkx(graph,node_attrs=['label','feature'])

        # Extract embeddings (features)
        embeddings = graph.ndata['feature'].numpy()

        # Calculate Euclidean distances between all pairs of nodes
        distance_matrix = euclidean_distances(embeddings)
        
        if threshold is None:
            mean_distance = np.mean(distance_matrix)
            alpha = 1.0  # This is a tunable hyperparameter
            threshold = alpha * mean_distance

        # Find the k nearest neighbors and filter by threshold
        sorted_indices = np.argsort(distance_matrix, axis=1)
        nearest_indices = sorted_indices[:, 1:k+1]  # Exclude the first column because it contains the distance to the node itself (0)

        # Reconstruct the graph based on distances
        G_reconstructed = nx.Graph()
        for i, node in enumerate(G_nx.nodes):
            for neighbor_idx in nearest_indices[i]:
                if distance_matrix[i][neighbor_idx] <= threshold:
                    G_reconstructed.add_edge(node, neighbor_idx)

        # Copy the 'label' and 'feature' attributes from G_nx to G_reconstructed
        for node in G_reconstructed.nodes:
            G_reconstructed.nodes[node]['label'] = G_nx.nodes[node]['label']
            G_reconstructed.nodes[node]['feature'] = G_nx.nodes[node]['feature']

        # Check for isolated nodes and add them back to the graph with attributes
        isolated_nodes = set(G_nx.nodes) - set(G_reconstructed.nodes)
        for isolated_node in isolated_nodes:
            G_reconstructed.add_node(isolated_node, label=G_nx.nodes[isolated_node]['label'], feature=G_nx.nodes[isolated_node]['feature'])

        # Convert the reconstructed NetworkX graph back to DGL graph
        graph = dgl.from_networkx(G_reconstructed, node_attrs=['label', 'feature'])

        # Add self-loops to the graph
        graph = dgl.add_self_loop(graph)
        return graph

    def reconstruct_graph_approximate_knn(self, graph, k, threshold, alpha):
        # Convert DGL graph to NetworkX graph
        G_nx = dgl.to_networkx(graph, node_attrs=['label', 'feature'])

        # Extract embeddings (features)
        embeddings = graph.ndata['feature'].numpy()

        # Build Annoy index with cosine similarity
        num_trees = 10  # This is a tunable hyperparameter
        annoy_index = AnnoyIndex(embeddings.shape[1], 'angular')  # Use angular similarity for normalized embeddings
        for i, embedding in enumerate(embeddings):
            annoy_index.add_item(i, embedding)
        annoy_index.build(num_trees)

        if threshold is None:
            mean_distance = np.mean(annoy_index.get_nns_by_item(0, k+1)[1:])  # Exclude the first neighbor, which is the node itself (distance = 0)
            alpha = 1.0  # This is a tunable hyperparameter
            threshold = alpha * mean_distance

        # Find the k nearest neighbors and filter by threshold
        nearest_indices = [annoy_index.get_nns_by_item(i, k+1)[1:] for i in range(embeddings.shape[0])]  # Exclude the first neighbor, which is the node itself (distance = 0)

        # Reconstruct the graph based on distances
        G_reconstructed = nx.Graph()
        for i, node in enumerate(G_nx.nodes):
            for neighbor_idx in nearest_indices[i]:
                neighbor_distance = 1 - annoy_index.get_distance(i, neighbor_idx)  # Convert cosine similarity to distance
                if neighbor_distance <= threshold:
                    G_reconstructed.add_edge(node, neighbor_idx)

        # Copy the 'label' and 'feature' attributes from G_nx to G_reconstructed
        for node in G_reconstructed.nodes:
            G_reconstructed.nodes[node]['label'] = G_nx.nodes[node]['label']
            G_reconstructed.nodes[node]['feature'] = G_nx.nodes[node]['feature']

        # Check for isolated nodes and add them back to the graph with attributes
        isolated_nodes = set(G_nx.nodes) - set(G_reconstructed.nodes)
        for isolated_node in isolated_nodes:
            G_reconstructed.add_node(isolated_node, label=G_nx.nodes[isolated_node]['label'], feature=G_nx.nodes[isolated_node]['feature'])

        # Convert the reconstructed NetworkX graph back to DGL graph
        graph = dgl.from_networkx(G_reconstructed, node_attrs=['label', 'feature'])

        # Add self-loops to the graph
        graph = dgl.add_self_loop(graph)
        return graph
