from utils import *
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.nn import BCEWithLogitsLoss
from torch_geometric.transforms import RandomLinkSplit

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

def generate_data_object(df, node_to_index, node_features_tensor):
    """
    Create a PyTorch Geometric Data object from DataFrame and node features.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the edges of the graph. Must have columns 'target_id' and 'source_id'.
    - node_to_index (dict): A dictionary mapping node IDs to their corresponding index in the tensor.
    - node_features_tensor (torch.Tensor): A tensor containing the features of each node in the graph.
    
    Returns:
    - data (torch_geometric.data.Data): A PyTorch Geometric Data object ready for use with GNN models.
    """
    edge_index_list = [(node_to_index[tid], node_to_index[sid]) for tid, sid in zip(df['target_id'], df['source_id'])]
    edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    data = Data(x=node_features_tensor, edge_index=edge_index_tensor, num_nodes=len(node_to_index))
    return data

def split_edges(data, test_ratio=0.1):
    """
    Split edges into training and testing sets using the RandomLinkSplit transform.
    
    Parameters:
    - data (torch_geometric.data.Data): The complete graph data.
    - test_ratio (float): The proportion of edges to use for the test set.
    
    Returns:
    - train_data (torch_geometric.data.Data): Data object containing the training set.
    - test_data (torch_geometric.data.Data): Data object containing the test set.
    """
    transform = RandomLinkSplit(is_undirected=False, num_val=0, num_test=test_ratio, neg_sampling_ratio=1.0)
    train_data, _, test_data = transform(data)
    return train_data, test_data


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
    loss = criterion(pred, data.edge_label)  # Ensure labels are on the correct device
    loss.backward()
    optimizer.step()
    return loss.item()
