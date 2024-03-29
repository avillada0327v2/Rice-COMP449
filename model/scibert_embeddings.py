from transformers import *
import pandas as pd
import torch
from model.bert_embeddings import *
from model.gnn import *

warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

# Load your fine-tuned model weights
checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))

# Apply the weights to your model
model.load_state_dict(checkpoint)


def generate_embeddings(df, model, tokenizer):
    """
    Generates embeddings for various text fields within the DataFrame using a pre-trained BERT model.

    This function preprocesses the DataFrame to assign unique IDs to citation texts,
    generates embeddings for source abstracts and cited contexts, and handles missing embeddings.

    Parameters:
    - df (pd.DataFrame): DataFrame containing citation data with 'source_abstract' and 'cited_context' fields.

    The function saves embeddings to disk and does not return any value.
    """

    if 'source_abstract' not in df.columns or 'citated_text' not in df.columns or 'citated_text_id' not in df.columns:
        raise ValueError("DataFrame must contain 'source_abstract', 'cited_text' and 'citated_text_id' columns.")

    set_seeds(42)

    # Assuming each function below properly handles batch processing and saves embeddings to disk.
    generate_source_abstract_embeddings_dict(tokenizer, model, df)  # Embeddings for source abstract
    generate_citated_text_embeddings_dict(tokenizer, model, df)  # Embeddings for cited context
    generate_missing_embeddings(tokenizer, model)  # Embeddings for missing abstracts or contexts


df = data_preprocess('/Users/haojiang/Desktop/COMP449/Rice-COMP449/data/full_context_PeerRead.csv')
generate_embeddings(df, model, tokenizer)
