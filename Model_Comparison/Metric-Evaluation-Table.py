#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.metrics import precision_score, average_precision_score
import joblib
import torch
from torcheval.metrics import HitRate


# In[2]:


# Precision@K evaluation
def average_precision_at_k(recommended_items, relevant_items, k):
    """Calculate the average precision at k for a single query.

    Args:
        recommended_items (list): A list of recommended items.
        relevant_items (set): A set of relevant items.
        k (int): The cutoff for top-k items.

    Returns:
        float: The average precision at k.
    """
    if not recommended_items:
        return 0

    score = 0.0
    num_hits = 0.0

    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / len(relevant_items) if relevant_items else 0

def mean_average_precision_at_k(query_results, relevant_items_per_query, k):
    """Calculate the mean average precision at k for a set of queries.

    Args:
        query_results (dict): A dictionary where the key is the query id and the value is a list of recommended items.
        relevant_items_per_query (dict): A dictionary where the key is the query id and the value is a set of relevant items.
        k (int): The cutoff for top-k items.

    Returns:
        float: The mean average precision at k.
    """
    average_precisions = []

    for query_id, recommended_items in query_results.items():
        if query_id in relevant_items_per_query:
            relevant_items = relevant_items_per_query[query_id]
            ap = average_precision_at_k(recommended_items, relevant_items, k)
            average_precisions.append(ap)

    return sum(average_precisions) / len(average_precisions) if average_precisions else 0


# In[3]:


# Hits@K evaluation
def hits_at_k(recommended_items, relevant_items, k):
    """Calculate the hits at k for a single query.

    Args:
        recommended_items (list): A list of recommended items.
        relevant_items (set): A set of relevant items.
        k (int): The cutoff for top-k items.

    Returns:
        float: The hits at k.
    """
    if not recommended_items:
        return 0

    num_hits = 0.0

    for item in recommended_items[:k]:
        if item in relevant_items:
            num_hits += 1.0

    return num_hits / min(k, len(relevant_items)) if relevant_items else 0

def mean_hits_at_k(query_results, relevant_items_per_query, k):
    """Calculate the mean hits at k for a set of queries.

    Args:
        query_results (dict): A dictionary where the key is the query id and the value is a list of recommended items.
        relevant_items_per_query (dict): A dictionary where the key is the query id and the value is a set of relevant items.
        k (int): The cutoff for top-k items.

    Returns:
        float: The mean hits at k.
    """
    hits_at_ks = []

    for query_id, recommended_items in query_results.items():
        if query_id in relevant_items_per_query:
            relevant_items = relevant_items_per_query[query_id]
            hits_at_k_value = hits_at_k(recommended_items, relevant_items, k)
            hits_at_ks.append(hits_at_k_value)

    return sum(hits_at_ks) / len(hits_at_ks) if hits_at_ks else 0


# In[4]:


# Recall@K evaluation
def recall_at_k(recommended_items, relevant_items, k):
    """
    Calculate the recall at k.

    Args:
        recommended_items (list): A list of recommended items, ordered by relevance.
        relevant_items (set): A set of relevant items.
        k (int): The cutoff rank for calculating recall.

    Returns:
        float: The recall at k.
    """
    # Identify the set of true positives among the top k recommendations
    true_positives_at_k = set(recommended_items[:k]).intersection(relevant_items)

    # The total number of relevant items
    total_relevant = len(relevant_items)

    # Recall is the fraction of relevant items that are retrieved among the top k
    if total_relevant > 0:
        return len(true_positives_at_k) / total_relevant
    else:
        return 0

def mean_recall_at_k(query_results, relevant_items_per_query, k):
    """Calculate the mean recall at k for a set of queries.

    Args:
        query_results (dict): A dictionary where the key is the query id and the value is a list of recommended items.
        relevant_items_per_query (dict): A dictionary where the key is the query id and the value is a set of relevant items.
        k (int): The cutoff for top-k items.

    Returns:
        float: The mean recall at k.
    """
    recalls = []

    for query_id, recommended_items in query_results.items():
        if query_id in relevant_items_per_query:
            relevant_items = relevant_items_per_query[query_id]
            recall = recall_at_k(recommended_items, relevant_items, k)
            recalls.append(recall)

    return sum(recalls) / len(recalls) if recalls else 0


# In[5]:


# Load saved models
# Assuming using joblib.load:
#bert_model = joblib.load('path/to/bert_model.pkl')
#node2vec_model = joblib.load('path/to/node2vec_model.pkl')
#custom_model = joblib.load('path/to/custom_model.pkl')

# If loading using pytorch:
#bert_model.load_state_dict(torch.load('path/to/your_pytorch_model.pth'))
#bert_model.eval()
#node2vec_model.load_state_dict(torch.load('path/to/your_pytorch_model.pth'))
#node2vec_model.eval()
#custom_model.load_state_dict(torch.load('path/to/your_pytorch_model.pth'))
#custom_model.eval()

# Load data
test_data = pd.read_csv('full_context_PeerRead.csv')


# In[ ]:


# Evaluate our models
# BERT model
def get_relevant(G):
    # Generate recommendatiodns for each node
    relevant_items_per_query = {}
    for node in G.nodes():
        relevant_items = set(G.neighbors(node))
        relevant_items_per_query[node] = relevant_items
    return relevant_items_per_query

# Node2Vec
def get_query_and_relevant(G, model, k):
    # Generate recommendations for each node
    query_results = {}
    relevant_items_per_query = {}
    for node in G.nodes():
        similar_nodes = model.wv.most_similar(node, topn=k)
        similar_node_names = [node[0] for node in similar_nodes]
        recommended_items = similar_node_names
        relevant_items = set(G.neighbors(node))
        query_results[node] = recommended_items
        relevant_items_per_query[node] = relevant_items
    return query_results, relevant_items_per_query

query_results1, relevant_items_per_query1 = get_query_and_relevant(G, model, k=20)

# Custom model
# Replace with appropriate evaluation/recommendation code snippet
custom_model_predictions = custom_model.predict(test_data)


# In[ ]:


# Calculate evaluation metrics
bert_map = mean_average_precision_at_k(top_k_similar_nodes, get_relevant(G), k=20)
node2vec_map = mean_average_precision_at_k(query_results1, relevant_items_per_query1, k=20)
# Replace custom_model metrics with appropriate parameters
custom_model_map = mean_average_precision_at_k(query_results, relevant_items_per_query, k=20)

bert_hits = mean_hits_at_k(top_k_similar_nodes, get_relevant(G), k=20)
node2vec_hits = mean_hits_at_k(query_results1, relevant_items_per_query1, k=20)
custom_model_hits = mean_hits_at_k(query_results, relevant_items_per_query, k=20)

bert_recall = mean_recall_at_k(top_k_similar_nodes, get_relevant(G), k=20)
node2vec_recall = mean_recall_at_k(query_results1, relevant_items_per_query1, k=20)
custom_model_recall = mean_recall_at_k(query_results, relevant_items_per_query, k=20)


# In[ ]:


# Organize results into a table
results = pd.DataFrame({
    'Model': ['BERT', 'Node2Vec', 'Custom Model'],
    'MAP@K': [bert_map, node2vec_map, custom_model_map],
    'Hits@K': [bert_hits, node2vec_hits, custom_model_hits],
    'Recall@K': [bert_recall, node2vec_recall, custom_model_recall]
})

# Display results
print(results)

