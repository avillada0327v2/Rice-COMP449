from utils import *
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.nn import BCEWithLogitsLoss

def get_node_id_index(df):
    """
    Create a mapping from node IDs to a unique index for all nodes appearing in the DataFrame, and return the list of all unique node IDs.
    
    This function aggregates all unique source and target node IDs from the DataFrame,
    assigns a unique index to each, and returns this mapping as a dictionary along with
    the list of all unique node IDs. This is useful for converting node IDs into indices
    for array or tensor operations, especially in graph-based models where nodes are
    identified by arbitrary IDs rather than sequential indices.
    
    Parameters:
    - df: DataFrame containing at least two columns: 'source_id' and 'target_id'.
    
    Returns:
    - all_node_ids: An array of all unique node IDs.
    - node_to_index: A dictionary mapping each unique node ID to a unique index.
    """
    # Concatenate 'source_id' and 'target_id' from the DataFrame and find unique values
    all_node_ids = pd.concat([df['source_id'], df['target_id']]).unique()
    # Create a dictionary mapping each node ID to its index in the unique array
    node_to_index = {node_id: i for i, node_id in enumerate(all_node_ids)}
    return all_node_ids, node_to_index


def generate_negative_samples(edge_index, num_nodes, num_negative_samples):
    """
    Generate negative samples for graph edges.

    Parameters:
    - edge_index: Tensor representing positive edges in the graph.
    - num_nodes: Total number of nodes in the graph.
    - num_negative_samples: Number of negative samples to generate.

    Returns:
    - A tensor of negative edge samples.
    """
    # Generate a set of all possible pairs of nodes
    all_possible_pairs = set((i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j)
    
    # Remove the positive edges from the set of all possible pairs to get potential negative pairs
    positive_edges = set((i.item(), j.item()) for i, j in zip(*edge_index))
    negative_candidates = list(all_possible_pairs - positive_edges)
    
    # Randomly select negative samples from the candidates
    negative_samples = random.sample(negative_candidates, num_negative_samples)
    
    return torch.tensor(negative_samples, dtype=torch.long).t()

class GNN(torch.nn.Module):
    """
    Graph Neural Network (GNN) using Graph Convolutional Network (GCN) layers.

    Parameters:
    - num_features: Number of features per node.
    - hidden_dim: Dimension of hidden layer.
    - output_dim: Dimension of output layer.
    """
    def __init__(self, num_features, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def gnn_train(model, data, optimizer, criterion, device):
    """
    Train a GNN model for one epoch.

    Parameters:
    - model: The GNN model to train.
    - data: Graph data containing nodes, edges, and labels.
    - optimizer: Optimizer to use for training.
    - criterion: Loss function.
    - device: The device (CPU or GPU) for training.

    Returns:
    - The loss value as a float.
    """
    model.train()
    optimizer.zero_grad()
    # Ensure data and model are on the correct device
    out = model(data)
    pred = out[data.edge_label_index[0]] * out[data.edge_label_index[1]]  # Example prediction logic
    pred = pred.sum(dim=-1)  # Sum over features for a simple score
    loss = criterion(pred, data.edge_labels)  # Ensure labels are on the correct device
    loss.backward()
    optimizer.step()
    return loss.item()
