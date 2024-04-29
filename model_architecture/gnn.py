"""
Builds, runs, and evaluates a GNN model using GCN layers for link prediction.
"""
import torch
import torch.nn.functional as F
from torch.nn import Dropout
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch.nn import BCEWithLogitsLoss
from torch_geometric.transforms import RandomLinkSplit
from model_architecture.utils import *


class GCN(torch.nn.Module):
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
    

class GAT(torch.nn.Module):
    """
    A Graph Neural Network (GNN) model using Graph Attention Network (GAT) layers for link prediction.
    
    Parameters:
    - in_channels (int): Number of features per input node.
    - hidden_channels (int): Number of features per node in hidden layers.
    - out_channels (int): Number of features per output node.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        """Encodes graph data into node embeddings."""
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return self.conv4(x, edge_index)

    def decode(self, z, edge_label_index):
        """Decodes node embeddings to predict link existence."""
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        """Decodes node embeddings to predict all possible links."""
        return (z @ z.t() > 0).nonzero(as_tuple=False).t()
    
    
def train_gnn_link_predictor(model, train_data, val_data, optimizer, criterion, n_epochs=1000, model_save_path='gnn_model.pth'):
    """
    Trains the gnn link prediction model and saves it to disk.
    
    Parameters:
    - model (torch.nn.Module): Instance of the Net model.
    - train_data (Data): Training dataset.
    - val_data (Data): Validation dataset.
    - optimizer (torch.optim.Optimizer): Optimizer for the model.
    - criterion (torch.nn.modules.loss._Loss): Loss function.
    - n_epochs (int): Number of training epochs.
    - model_save_path (str): File path to save the trained model.
    """
    set_seeds(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_data.edge_label_index = torch.cat([train_data.pos_edge_label_index, train_data.neg_edge_label_index], dim=-1)
    train_data.edge_label = torch.cat([train_data.pos_edge_label , train_data.neg_edge_label], dim=-1)
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        out = model.decode(z, train_data.edge_label_index).view(-1)
        loss = criterion(out, train_data.edge_label)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            val_auc = eval_gnn_link_predictor(model, val_data)
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")
    
    # Save the model to disk after training
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
    
    return model


@torch.no_grad()
def eval_gnn_link_predictor(model, data):
    """
    Evaluates the gnn link prediction model on a dataset.
    
    Parameters:
    - model: The trained model.
    - data: Dataset for evaluation.
    
    Returns:
    - auc_score (float): The AUC score of the model on the given dataset.
    """
    model.eval()
    data.edge_label_index = torch.cat([data.pos_edge_label_index, data.neg_edge_label_index], dim=-1)
    data.edge_label = torch.cat([data.pos_edge_label, data.neg_edge_label], dim=-1)
   
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


@torch.no_grad()
def get_gnn_top_k_links(model, data, df):
    """
    For each node, get the top-K highest scoring potential links.
    
    Parameters:
    - model: The trained gnn_ model for link prediction.
    - data: The graph data as a PyTorch Geometric Data object.
    - df: DataFrame containing 'target_id' and 'source_id' for calculating recommendations.
    
    Returns:
    - top_k_links: A dictionary where each key is a target node, and the value is a list of recommended item IDs.
    """
    all_node_ids, node_to_index = get_node_id_index(df)
    
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


def get_gnn_test_report(model, model_path, data, df):
    """
    Evaluates the GNN model on test data and prints performance metrics including Mean Reciprocal Rank (MRR),
    Mean Average Precision at various levels of K (MAP@K), and Recall at K for different values of K.

    Parameters:
    - model (torch.nn.Module): The trained GNN model ready for evaluation.
    - model_path (str): Path to the saved model; used if model needs to be loaded.
    - data (Data): PyTorch Geometric Data object used for model evaluation.
    - df (pd.DataFrame): DataFrame containing the ground truth links for evaluation
    
    Returns:
    - report (dict): A dictionary containing the evaluation metrics.
    
    Note:
    - It is assumed that 'get_relevant_items', 'get_gnn_top_k_links', 'mean_reciprocal_rank',
      'mean_average_precision_at_k', and 'recall_at_k' are predefined functions available in the scope.
    """
    # Check if the model file exists
    if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path}. Please run in 'train' mode first.")
            
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    all_node_ids, node_to_index = get_node_id_index(df)
    relevant_items = get_relevant_items(df)
    top_k_links = get_gnn_top_k_links(model, data, df)

    # Dictionary to hold metrics
    report = {
        'MRR': mean_reciprocal_rank(top_k_links, relevant_items)
    }
    
    report.update({f'MAP@{k}': mean_average_precision_at_k(top_k_links, relevant_items, k) for k in [5, 10, 30, 50, 80]})
    report.update({f'Recall@{k}': recall_at_k(top_k_links, relevant_items, k) for k in [5, 10, 30, 50, 80]})
    return report

        
def run_gnn(df, mode, model, model_path):
    """
    Orchestrates the workflow for training and evaluating a GNN link prediction model based on provided data.
    
    Parameters:
    - df: DataFrame containing citation data.
    - mode (str): Specifies the operation mode. Supported modes: 'train', 'evaluate'.
    - model (str): Specifies which GNN model we gonna use. Supported model: 'GCN', 'GAT'.
    - model_path (str): Path where the trained model is saved or to be saved.
    """
    if mode not in ['train', 'evaluate']:
        raise ValueError("Mode must be 'train' or 'evaluate'")
        
    set_seeds(42)
    
    # get the grpah data
    graph = prepare_graph_data(df)
    
    # Data splitting
    split = RandomLinkSplit(num_val=0.1, 
                            num_test=0.1, 
                            is_undirected=False, 
                            split_labels=True,
                            add_negative_train_samples=True, 
                            neg_sampling_ratio=1.0)
    
    train_data, val_data, test_data = split(graph)
    
    parameters = {'GCN':{'hidden_channels': 64, 'output_channels': 64, 'epochs': 2200},
                  'GAT':{'hidden_channels': 70, 'output_channels': 64, 'epochs': 600}
                 }
    
    # Model training
    if model == 'GCN':
        gnn_model = GCN(graph.num_features, 
                        parameters['GCN']['hidden_channels'], 
                        parameters['GCN']['output_channels'])
        epochs = parameters['GCN']['epochs']
        
    elif model == 'GAT':
        gnn_model = GAT(graph.num_features, 
                        parameters['GAT']['hidden_channels'], 
                        parameters['GAT']['output_channels'])
        epochs = parameters['GAT']['epochs']
        
    optimizer = torch.optim.Adam(params=gnn_model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    if mode == 'train':
        gnn_model = train_gnn_link_predictor(gnn_model, train_data, val_data, optimizer, criterion, epochs, model_path)
    
    # Performance evaluation
    if mode == 'evaluate':
        report = get_gnn_test_report(gnn_model, model_path, test_data, df)
        return report
