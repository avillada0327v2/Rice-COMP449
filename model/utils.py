import os  # Operating system interfaces
import gc  # Garbage Collector interface
import numpy as np  # Support for large, multi-dimensional arrays and matrices
import pandas as pd  # Data manipulation and analysis
import pickle  # Object serialization
import random  # Generating random numbers
import torch  # PyTorch deep learning framework
from sklearn.model_selection import train_test_split  # Split arrays or matrices into random train and test subsets
from sklearn.preprocessing import MultiLabelBinarizer  # Transform between iterable of iterables and a multilabel format
from sklearn.metrics import roc_auc_score  # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
from torch_geometric.data import Data  # Data handling of graphs in PyTorch Geometric
from torch.utils.data import DataLoader, TensorDataset  # DataLoader and Dataset wrapping tensors for PyTorch
from transformers import BertTokenizer, BertModel  # BERT model and tokenizer from Hugging Face's Transformers library
from model.bert_embeddings import *
import matplotlib.pyplot as plt


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
    - df (pd.DataFrame): DataFrame with 'target_id' and 'source_id' columns
    
    Returns:
    - A dictionary where keys are target node IDs and values are lists of relevant source node IDs.
    """
    relevant_items = {}
    for target in df['target_id'].unique():
        relevant_sources = df[df['target_id'] == target]['source_id'].tolist()
        relevant_items[target] = relevant_sources
    return relevant_items


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


def prepare_graph_data(df):
    """
    Prepares graph data for GNN training or evaluation by generating node feature tensors
    from embeddings and creating a PyTorch Geometric Data object.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the dataset with node identifiers and any additional information required for generating embeddings.

    Returns:
    - Data: A PyTorch Geometric Data object ready for GNN processing.
    
    Assumes existence and documentation of:
    - get_target_embeddings(df): Function to get embeddings for target nodes.
    - get_source_abstract_embeddings(): Function to get embeddings for source abstracts.
    - get_missing_embeddings(): Function to handle missing embeddings.
    - get_node_id_index(df): Function to map node identifiers to their index in the tensor.
    - generate_data_object(df, node_to_index, node_features_tensor): Function to create a PyTorch Geometric Data object.
    """
    # Retrieve embeddings
    target_embeddings = get_target_embeddings(df)
    source_abstract_embeddings = get_source_abstract_embeddings()
    missing_cls_embeddings = get_missing_embeddings()
    
    # Generate node features and data object for GNN
    all_node_ids, node_to_index = get_node_id_index(df)

    # Retrieve embeddings and store them in a list first
    node_embeddings_list = [target_embeddings.get(node_id, source_abstract_embeddings.get(node_id, missing_cls_embeddings)) for node_id in all_node_ids]

    # Convert the list of embeddings to a single NumPy array
    node_embeddings_array = np.array(node_embeddings_list)
    node_features_tensor = torch.tensor(node_embeddings_array, dtype=torch.float)

    # create the grpah
    graph = generate_data_object(df, node_to_index, node_features_tensor)
    return graph


def recall_at_k(recommended_items, relevant_items, k=5):
    """
    Calculate the recall at k for the recommendations.
    
    Parameters:
    - recommended_items (dict): A dictionary of recommended item IDs for each target node.
    - relevant_items (dict): A dictionary of relevant item IDs for each target node.
    - k (int): The number of top recommendations to consider.
    
    Returns:
    - The average recall at k across all target nodes.
    """
    recalls = []
    for target, recommendations in recommended_items.items():
        true_positives_at_k = set(recommendations[:k]).intersection(relevant_items.get(target, []))
        if relevant_items.get(target):
            recalls.append(len(true_positives_at_k) / len(relevant_items[target]))
    return round(sum(recalls) / len(recalls), 3) if recalls else 0


def mean_reciprocal_rank(recommended_items, relevant_items):
    """
    Calculate the Mean Reciprocal Rank (MRR) for a set of recommendations.
    
    MRR is a statistical measure used for evaluating the effectiveness of a recommendation system. 
    It calculates the average of the reciprocal ranks of the first relevant recommendation for each query.
    
    Parameters:
    - recommended_items (dict): A dictionary where keys are query identifiers and values are lists of recommended item IDs, ordered by relevance.
    - relevant_items (dict): A dictionary where keys are the same query identifiers as in recommended_items and values are lists of relevant item IDs.
    
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
    return round(sum(rr) / len(rr), 3) if rr else 0


def mean_average_precision_at_k(recommended_items, relevant_items, k):
    """
    Calculate the Mean Average Precision at k (MAP@k) for a set of recommendations.
    
    MAP@k is a measure used to evaluate the quality of a recommendation system, considering both the order of recommendations and their relevance. It computes the average precision for each query and then the mean of these average precisions across all queries, considering only the top k recommendations.
    
    Parameters:
    - recommended_items (dict): A dictionary with query identifiers as keys and ordered lists of recommended item IDs as values.
    - relevant_items (dict): A dictionary with query identifiers as keys and lists of relevant item IDs as values.
    - k (int): The number of top recommendations to consider for calculating precision.
    
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
    return round(sum(ap) / len(ap), 3) if ap else 0


def metric_evaluation_table(models):
    """
    Creates a DataFrame that compares the performance metrics of various models.

    Parameters:
    - models (dict): A dictionary where each key is a model's name and each value is its evaluation report.

    Returns:
    - pd.DataFrame: A DataFrame where each row represents a model and each column represents a metric.
    """
    # List to store each model's report for DataFrame creation
    model_reports = []

    # Loop through the models dictionary to process each model's report
    for model_name, model_report in models.items():
        # Add the model's name to its report
        model_report['Model'] = model_name
        # Append the modified report to the list
        model_reports.append(model_report)

    # Create DataFrame for comparison and set the 'Model' column as the index
    df_comparison = pd.DataFrame(model_reports)
    df_comparison.set_index('Model', inplace=True)

    # Return the comparison DataFrame
    return df_comparison

def plot_result(evaluation_table):
    """
    Plot the MRR performance and MAP, Recall scores for different models.

    The function takes an evaluation table as input and produces three plots:
    1. A bar plot for MRR performance per model.
    2. A line plot for MAP scores at different cut-offs for each model.
    3. A line plot for Recall scores at different cut-offs for each model.

    Parameters:
    - evaluation_table (DataFrame): A pandas DataFrame containing the MRR, MAP, and Recall metrics for different models.
    """
    # Bar plot for MRR Performance per Model
    plt.figure(figsize=(10, 5))
    evaluation_table['MRR'].plot(kind='bar', color=['powderblue', 'lightskyblue', 'steelblue'])
    plt.title('MRR Performance per Model')
    plt.ylabel('MRR')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Line plot for MAP Scores for Different Models
    plt.figure(figsize=(10, 5))
    for model, color in zip(list(evaluation_table.index), ['powderblue', 'lightskyblue', 'steelblue']):
        plt.plot(['MAP@5', 'MAP@10', 'MAP@30', 'MAP@50', 'MAP@80'], 
                 [evaluation_table.loc[model]['MAP@5'], 
                  evaluation_table.loc[model]['MAP@10'], 
                  evaluation_table.loc[model]['MAP@30'], 
                  evaluation_table.loc[model]['MAP@50'], 
                  evaluation_table.loc[model]['MAP@80']], 
                 marker='o', label=model, color=color)
    plt.title('MAP Scores for Different Models')
    plt.ylabel('MAP Score')
    plt.legend(title='Model', loc='center left', bbox_to_anchor=(1, 0.5))
    # Adjust the layout so the legend is fully visible
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.grid(True)
    plt.show()

    # Line plot for Recall Scores for Different Models
    plt.figure(figsize=(10, 5))
    for model, color in zip(list(evaluation_table.index), ['powderblue', 'lightskyblue', 'steelblue']):
        plt.plot(['Recall@5' ,'Recall@10', 'Recall@30', 'Recall@50', 'Recall@80'], 
                 [evaluation_table.loc[model]['Recall@5'], 
                  evaluation_table.loc[model]['Recall@10'], 
                  evaluation_table.loc[model]['Recall@30'], 
                  evaluation_table.loc[model]['Recall@50'], 
                  evaluation_table.loc[model]['Recall@80']], 
                 marker='o', label=model, color=color)
    plt.title('Recall Scores for Different Models')
    plt.ylabel('Recall Score')
    plt.legend(title='Model', loc='center left', bbox_to_anchor=(1, 0.5))
    # Adjust the layout so the legend is fully visible
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.grid(True)                                        
    plt.show()                                       
                                        
