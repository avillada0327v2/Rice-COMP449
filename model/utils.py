import os
import gc
import numpy as np
import pandas as pd
import pickle
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
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
    
def get_citation_text_id(df):
    """
    Generates a unique ID for each unique cited text combination in the DataFrame.

    This function adds a new column to the DataFrame, 'cited_text', which is a combination
    of 'left_cited_text' and 'right_cited_text'. It then creates a mapping from each unique
    cited text to a unique ID and assigns these IDs to a new column, 'cited_text_id'.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the citation texts.

    Returns:
    - pd.DataFrame: The original DataFrame augmented with 'cited_text' and 'cited_text_id' columns.
    """
    
    # Combine the left and right cited text with a space in between to form a complete citation
    df['citated_text'] = df['left_citated_text'] + " " + df['right_citated_text']
    
    # Create a dictionary mapping each unique cited text to a unique ID
    # Using enumerate ensures each cited text gets a unique, sequential ID
    cited_voca = {text: id_ for id_, text in enumerate(df['citated_text'].unique())}
    
    # Map the cited texts in the DataFrame to their respective IDs
    # The mapping uses the dictionary created above
    df['citated_text_id'] = df['citated_text'].map(cited_voca)
    
    return df

def data_preprocess(data):
    """
    Loads a dataset from a CSV file and preprocesses it by assigning unique IDs to citation texts.
    
    Parameters:
    - data (str or pd.DataFrame): The path to the CSV file containing the dataset or a DataFrame.
    
    Returns:
    - pd.DataFrame: A DataFrame with the preprocessing applied.
    - None: If the file is not found or an error occurs during loading.
    """
    # Validate input
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, str):
        try:
            df = pd.read_csv(data)
        except FileNotFoundError:
            print(f"CSV file '{data}' not found.")
            return None
        except Exception as e:
            print(f"An error occurred while loading the CSV file: {e}")
            return None
    else:
        print("Input type for 'data' is not recognized. Please provide a file path or a DataFrame.")
        return None

    # Assuming get_citation_text_id is a function that you've defined elsewhere
    try:
        df = get_citation_text_id(df)
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return None

    return df

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