import gc
import numpy as np
import pandas as pd
import pickle
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_recommended_items(model, data, df, all_node_ids, node_to_index):
    """
    Generate item recommendations for each target node based on model embeddings.
    
    Parameters:
    - model: The trained model that generates embeddings.
    - data: The input data for the model to generate embeddings.
    - df: DataFrame containing 'target_id' and 'source_id' for calculating recommendations.
    - all_node_ids: A list of all node IDs in the order they appear in the embeddings.
    - node_to_index: A dictionary mapping node IDs to their index in the embedding matrix.
    
    Returns:
    - recommended_items: A dictionary where each key is a target node, and the value is a list of recommended item IDs.
    - recommended_scores: A dictionary where each key is a target node, and the value is a list of scores corresponding to the recommended items.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(data)
    recommended_items = {}
    recommended_scores = {}
    
    for target in df['target_id'].unique():
        target_index = node_to_index[target]  # Get the index of the target node
        target_embedding = embeddings[target_index].unsqueeze(0)  # Add batch dimension for broadcasting
        
        # Compute similarity scores via dot product between the target and all embeddings
        scores = torch.matmul(target_embedding, embeddings.T).squeeze(0)  # Remove batch dimension
        
        # Sort scores in descending order and get their indices
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    
        
        # Exclude the target node itself from the recommendations
        exclude_target = sorted_indices != target_index
        
        # Apply the mask to exclude the target and get the top-k indices and their scores
        top_k_indices = sorted_indices[exclude_target]
        top_k_scores = sorted_scores[exclude_target]
        
        # Convert indices back to node IDs
        top_k_ids = [all_node_ids[idx] for idx in top_k_indices.cpu().numpy()]
        recommended_items[target] = top_k_ids
        recommended_scores[target] = top_k_scores
        
    return recommended_items, recommended_scores

def get_relevant_items(df):
    """
    Retrieves the relevant items for each target node based on the DataFrame provided.
    
    Parameters:
    - df: DataFrame with 'target_id' and 'source_id' columns
    
    Returns:
    - A dictionary where keys are target node IDs and values are lists of relevant source node IDs.
    """
    relevant_items = {}
    for target in df['target_id'].unique():
        relevant_sources = df[df['target_id'] == target]['source_id'].tolist()
        relevant_items[target] = relevant_sources
    return relevant_items

def recall_at_k(recommended_items, relevant_items, k=5):
    """
    Calculate the recall at k for the recommendations.
    
    Parameters:
    - recommended_items: A dictionary of recommended item IDs for each target node.
    - relevant_items: A dictionary of relevant item IDs for each target node.
    - k: The number of top recommendations to consider.
    
    Returns:
    - The average recall at k across all target nodes.
    """
    recalls = []
    for target, recommendations in recommended_items.items():
        true_positives_at_k = set(recommendations[:k]).intersection(relevant_items.get(target, []))
        if relevant_items.get(target):
            recalls.append(len(true_positives_at_k) / len(relevant_items[target]))
    return sum(recalls) / len(recalls) if recalls else 0


def mean_reciprocal_rank(recommended_items, relevant_items):
    """
    Calculate the Mean Reciprocal Rank (MRR) for a set of recommendations.
    
    MRR is a statistical measure used for evaluating the effectiveness of a recommendation system. 
    It calculates the average of the reciprocal ranks of the first relevant recommendation for each query.
    
    Parameters:
    - recommended_items: A dictionary where keys are query identifiers and values are lists of recommended item IDs, ordered by relevance.
    - relevant_items: A dictionary where keys are the same query identifiers as in recommended_items and values are lists of relevant item IDs.
    
    Returns:
    - The mean reciprocal rank of the recommendations across all queries. Returns 0 if there are no relevant items found in the recommendations.
    """
    rr = []  # List to store reciprocal ranks
    for target, recommendations in recommended_items.items():
        # Check each recommended item for its relevance
        for idx, item in enumerate(recommendations, start=1):  # Start indexing at 1 for human-readable ranks
            if item in relevant_items.get(target, []):
                rr.append(1.0 / idx)  # Compute reciprocal rank
                break  # Only consider the first relevant item
    
    # Calculate the mean of the reciprocal ranks; return 0 if rr is empty
    return sum(rr) / len(rr) if rr else 0


def mean_average_precision_at_k(recommended_items, relevant_items, k):
    """
    Calculate the Mean Average Precision at k (MAP@k) for a set of recommendations.
    
    MAP@k is a measure used to evaluate the quality of a recommendation system, considering both the order of recommendations and their relevance. It computes the average precision for each query and then the mean of these average precisions across all queries, considering only the top k recommendations.
    
    Parameters:
    - recommended_items: A dictionary with query identifiers as keys and ordered lists of recommended item IDs as values.
    - relevant_items: A dictionary with query identifiers as keys and lists of relevant item IDs as values.
    - k: The number of top recommendations to consider for calculating precision.
    
    Returns:
    - The mean average precision at k across all queries. Returns 0 if there are no queries.
    """
    ap = []  # List to store the average precision for each query
    for target, recommendations in recommended_items.items():
        score = 0.0
        num_hits = 0.0
        for i, item in enumerate(recommendations[:k]):
            if item in relevant_items.get(target, []): 
                num_hits += 1.0
                score += num_hits / (i + 1.0)  # Calculate precision at i
        if relevant_items.get(target):  # Avoid division by zero
            # Normalize by the smaller of the number of relevant items or k
            ap.append(score / min(len(relevant_items[target]), k))  
        else:
            ap.append(0)  # Append 0 for targets with no relevant items to ensure fairness

    # Calculate the mean of the average precisions; return 0 if ap is empty
    return sum(ap) / len(ap) if ap else 0