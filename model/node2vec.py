import torch
import torch.optim as optim
from torch_geometric.nn import Node2Vec
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
from model.utils import *

def train_node2vec_link_predictor(model, train_data, val_data, optimizer, criterion, n_epochs=1000, model_save_path='node2vec_model.pth'):
    """
    Trains the Node2Vec model for the task of link prediction and saves the trained model.

    Parameters:
    - model (torch.nn.Module): The Node2Vec model instance to be trained.
    - train_data (Data): The training data containing the edge indices and edge labels.
    - optimizer (torch.optim.Optimizer): The optimizer for training.
    - criterion (torch.nn.modules.loss._Loss): The loss function used for training.
    - n_epochs (int): The number of epochs to train the model for.
    - model_save_path (str): The path where the trained model will be saved.

    Returns:
    - model (torch.nn.Module): The trained Node2Vec model.
    """
    set_seeds(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_data.edge_index = train_data.pos_edge_label_index.to(device)
    train_data.neg_edge_index = train_data.neg_edge_label_index.to(device)
    train_data.edge_label = torch.cat([train_data.pos_edge_label, train_data.neg_edge_label], dim=-1)
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Generate embeddings
        z = model().to(device)  # Adjusted for device

        # Use the provided negative samples without generating new ones
        # Combine pos and neg edges
        total_edge_index = torch.cat([train_data.edge_index, train_data.neg_edge_index], dim=-1)
       
        # Predictions
        out = (z[total_edge_index[0]] * z[total_edge_index[1]]).sum(dim=1)  # Dot product for edge prediction
        loss = criterion(out, train_data.edge_label)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            val_auc = eval_node2vec_link_predictor(model, val_data)
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")
            
    # Save the model to disk
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
    
    return model


@torch.no_grad()
def eval_node2vec_link_predictor(model, data):
    """
    Evaluates the node2vec link prediction model on a dataset.
    
    Parameters:
    - model (torch.nn.Module): The trained model.
    - data (Data): Dataset for evaluation.
    
    Returns:
    - auc_score (float): The AUC score of the model on the given dataset.
    """
    model.eval()
    
    # Get node embeddings
    z = model()
    
    pos_edge_index = data.pos_edge_label_index.to('cpu')
    neg_edge_index = data.neg_edge_label_index.to('cpu')  

    # Predictions for positive samples
    pos_out = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1).sigmoid()

    # Predictions for negative samples
    neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1).sigmoid()

    # Concatenate predictions and true labels
    preds = torch.cat([pos_out, neg_out], dim=0).cpu().numpy()
    labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0).cpu().numpy()

    # Compute AUC score
    auc_score = roc_auc_score(labels, preds)

    return auc_score


@torch.no_grad()
def get_node2vec_topk_links(model, data, df):
    """
    For each node, get the top-K highest scoring potential links.
    
    Parameters:
    - model (torch.nn.Module): The trained node2vec model for link prediction.
    - data (Data): The graph data as a PyTorch Geometric Data object.
    - df (pd.DataFrame): DataFrame containing 'target_id' and 'source_id' for calculating recommendations.
    
    Returns:
    - top_k_links (dict): A dictionary where each key is a target node, and the value is a list of recommended item IDs.
    """
    all_node_ids, node_to_index = get_node_id_index(df)
    model.eval()
    
    # Get node embeddings
    z = model()
    device = z.device
    
    # Compute similarity scores between all pairs of nodes
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


def get_node2vec_test_report(model, model_path, data, df):
    """
    Evaluates the node2vec model on test data and prints performance metrics including Mean Reciprocal Rank (MRR),
    Mean Average Precision at various levels of K (MAP@K), and Recall at K for different values of K.

    Parameters:
    - model (torch.nn.Module): The trained node2vec model ready for evaluation.
    - model_path (str): Path to the saved model; used if model needs to be loaded.
    - data (Data): PyTorch Geometric Data object used for model evaluation.
    - df (pd.DataFrame): DataFrame containing the ground truth links for evaluation
    
    Returns:
    - report (dict): A dictionary containing the evaluation metrics.
    
    Note:
    - It is assumed that 'get_relevant_items', 'get_node2vec_topk_links', 'mean_reciprocal_rank',
      'mean_average_precision_at_k', and 'recall_at_k' are predefined functions available in the scope.
    """
    # Check if the model file exists
    if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path}. Please run in 'train' mode first.")
            
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    all_node_ids, node_to_index = get_node_id_index(df)
    relevant_items = get_relevant_items(df)
    top_k_links = get_node2vec_topk_links(model, data, df)

    # Dictionary to hold metrics
    report = {
        'MRR': mean_reciprocal_rank(top_k_links, relevant_items)
    }

    report.update({f'MAP@{k}': mean_average_precision_at_k(top_k_links, relevant_items, k) for k in [5, 10, 30, 50, 80]})
    report.update({f'Recall@{k}': recall_at_k(top_k_links, relevant_items, k) for k in [5, 10, 30, 50, 80]})

    return report
        
        
def run_node2vec(df, mode, model_path):
    """
    Orchestrates the workflow for training and evaluating a Node2Vec link prediction model.
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing the graph data.
    - mode (str): The operation mode ('train' or 'evaluate').
    - model_path (str): The file path to save or load the model.
    
    Returns:
    - report (dict): A dictionary containing the evaluation metrics if in 'evaluate' mode.
    - None: If in 'train' mode, as the model is saved to disk without returning an object.
    
    Raises:
    - ValueError: If the provided mode is not 'train' or 'evaluate'.
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
    
    # Model training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node2vec_model = Node2Vec(edge_index=train_data.edge_index, 
                     embedding_dim=128, 
                     walk_length=30, 
                     context_size=10, 
                     walks_per_node=10, 
                     num_negative_samples=1, 
                     sparse=True).to(device)
    optimizer = optim.Adam(node2vec_model.parameters(), lr=0.05)
    criterion = torch.nn.BCEWithLogitsLoss()
    if mode == 'train':
        model = train_node2vec_link_predictor(node2vec_model, train_data, val_data, optimizer, criterion, 1000, model_path)
        
    # Performance evaluation
    if mode == 'evaluate':
        report = get_node2vec_test_report(node2vec_model, model_path, test_data, df)
        return report
