from utils import *
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
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

 
class Net(torch.nn.Module):
    """
    A Graph Neural Network (GNN) model using Graph Convolutional Network (GCN) layers for link prediction.
    
    Parameters:
    - in_channels (int): Number of features per input node.
    - hidden_channels (int): Number of features per node in hidden layers.
    - out_channels (int): Number of features per output node.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        """Encodes graph data into node embeddings."""
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.conv3(x, edge_index)

    def decode(self, z, edge_label_index):
        """Decodes node embeddings to predict link existence."""
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        """Decodes node embeddings to predict all possible links."""
        return (z @ z.t() > 0).nonzero(as_tuple=False).t()

def train_link_predictor(model, train_data, val_data, optimizer, criterion, n_epochs=1000, model_save_path='model.pth'):
    """
    Trains the link prediction model and saves it to disk.
    
    Parameters:
    - model: Instance of the Net model.
    - train_data: Training dataset.
    - val_data: Validation dataset.
    - optimizer: Optimizer for the model.
    - criterion: Loss function.
    - n_epochs (int): Number of training epochs.
    - model_save_path (str): File path to save the trained model.
    """
    set_seeds(42)
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        out = model.decode(z, train_data.edge_label_index).view(-1)
        loss = criterion(out, train_data.edge_label)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            val_auc = eval_link_predictor(model, val_data)
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")
    
    # Save the model to disk after training
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
    
    return model


@torch.no_grad()
def eval_link_predictor(model, data):
    """
    Evaluates the link prediction model on a dataset.
    
    Parameters:
    - model: The trained model.
    - data: Dataset for evaluation.
    
    Returns:
    - auc_score (float): The AUC score of the model on the given dataset.
    """
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

def get_test_report(model, data, df, all_node_ids, node_to_index):
    """
    Evaluates the GNN model on test data and prints the performance metrics including
    Mean Reciprocal Rank (MRR), Mean Average Precision at various levels of K (MAP@K),
    and Recall at K for different values of K.
    
    Parameters:
    - model: The trained GNN model for link prediction.
    - data: The PyTorch Geometric data object used for model evaluation.
    - df: DataFrame containing the ground truth links for evaluation.
    - all_node_ids: A list of all node IDs in the graph.
    - node_to_index: A dictionary mapping node IDs to their index in the adjacency matrix.
    
    Note:
    - It is assumed that 'get_relevant_items', 'get_top_k_links_per_node', 'mean_reciprocal_rank',
      'mean_average_precision_at_k', and 'recall_at_k' are predefined functions available in the scope.
    """
    
    relevant_items = get_relevant_items(df)
    top_k_links = get_top_k_links_per_node(model, data, df, all_node_ids, node_to_index)
    
    print('Test data recommendations performance:')
    print(f'Mean Reciprocal Rank: {mean_reciprocal_rank(top_k_links, relevant_items)}')
    for k in [5, 10, 30, 50, 80]:
        print(f'Mean Average Precision@{k}: {mean_average_precision_at_k(top_k_links, relevant_items, k)}')          
    for k in [5, 10, 30, 50, 80]:
        print(f'Recall@{k}: {recall_at_k(top_k_links, relevant_items, k)}')


@torch.no_grad()
def get_top_k_links_per_node(model, data, df, all_node_ids, node_to_index):
    """
    For each node, get the top-K highest scoring potential links.
    
    Parameters:
    - model: The trained Net model for link prediction.
    - data: The graph data as a PyTorch Geometric Data object.
    - df: DataFrame containing 'target_id' and 'source_id' for calculating recommendations.
    - all_node_ids: A list of all node IDs in the order they appear in the embeddings.
    - node_to_index: A dictionary mapping node IDs to their index in the embedding matrix.
    
    Returns:
    - top_k_links: A dictionary where each key is a target node, and the value is a list of recommended item IDs.
    """
    model.eval()
    z = model.encode(data.x, data.edge_index)
    
    # Compute similarity score for all pairs
    similarity_matrix = torch.matmul(z, z.T)
    
    top_k_links = {}
    for target in df['target_id'].unique():
        node_idx = node_to_index[target]
        # Exclude self-link
        similarity_matrix[node_idx, node_idx] = float('-inf')
        
        # Get top-K scores and their indices
        scores, indices = torch.sort(similarity_matrix[node_idx], descending=True)
       
        # Exclude self-link from recommendations
        recommended_indices = indices[1:] if indices[0] == node_idx else indices
        top_k_ids = [all_node_ids[idx] for idx in recommended_indices if idx != node_idx]
        recommended_scores = scores[1:] if scores[0] == node_idx else scores
        
        
        top_k_links[target] = top_k_ids
    
    return top_k_links
