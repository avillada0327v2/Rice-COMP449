from utils import *
from gnn import *
from bert_embeddings import *
import argparse

def run(method):
    """
    Run the training or evaluation process for a GNN model based on the specified method.
    
    Parameters:
    - method: A string indicating whether to 'train' the model or 'eval'uate it.
    """
    # Fixing random seeds for reproducibility
    set_seeds(42)
    
    # Load DataFrame
    df = pd.read_csv('full_context_PeerRead.csv')

    target_embeddings = get_target_embeddings(df)
    missing_cls_embeddings = get_missing_embeddings()

    # Generate node ID to index mapping and node features
    all_node_ids, node_to_index = get_node_id_index(df)

    # Assuming target_embeddings is a dictionary mapping node IDs to embeddings and missing_cls_embeddings is the embedding for missing nodes
    node_embeddings = {node_id: target_embeddings.get(node_id, missing_cls_embeddings) for node_id in all_node_ids}

    # Prepare node features tensor
    node_features = [node_embeddings[node_id] for node_id in all_node_ids]
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)

    # Prepare edge index tensor and labels
    edge_index_list = [(node_to_index[tid], node_to_index[sid]) for tid, sid in zip(df['target_id'], df['source_id'])]
    edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    # Generate negative samples to have negative edges for training
    num_negative_samples = edge_index_tensor.size(1)  # Matching number of negative samples to positive samples
    negative_edge_index = generate_negative_samples(edge_index_tensor, len(all_node_ids), num_negative_samples)

    # Combine positive and negative edge indices
    edge_label_index = torch.cat([edge_index_tensor, negative_edge_index], dim=1)

    # Create labels for edges: 1 for positive (existing edges), 0 for negative (non-existing edges)
    edge_labels = torch.cat([torch.ones(edge_index_tensor.size(1), dtype=torch.float),
                             torch.zeros(negative_edge_index.size(1), dtype=torch.float)], dim=0)

    # Create the graph data object
    data = Data(x=node_features_tensor, edge_index=edge_index_tensor)
    data.edge_label_index = edge_label_index
    data.edge_labels = edge_labels

    # Now 'data' is ready to be used for training the GNN model
    # Initialize model, optimizer, and loss function

    num_features = data.num_features
    hidden_dim = 64
    output_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN(num_features, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = BCEWithLogitsLoss()
    
    # Train the model
    for epoch in range(2000):
        loss = gnn_train(model, data, optimizer, criterion, device)
        if (epoch + 1) % 50 == 0 and method == 'train':
            print(f"Epoch {epoch + 1}: Loss {loss}") 
            
    # Evaluate
    if method == 'eval':
        relevant_items = get_relevant_items(df)
        recommended_items, recommended_scores = get_recommended_items(model, data, df, all_node_ids, node_to_index)

        print(f'Mean Reciprocal Rank: {mean_reciprocal_rank(recommended_items, relevant_items)}')
        for k in [5, 10, 30, 50, 80]:
            print(f'Mean Average Precision@{k}: {mean_average_precision_at_k(recommended_items, relevant_items, k)}')          
        for k in [5, 10, 30, 50, 80]:
            print(f'Recall@{k}: {recall_at_k(recommended_items, relevant_items, k)}')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', '-m', type=str, default='train')
    args = parser.parse_args()
    
    run(args.method)
