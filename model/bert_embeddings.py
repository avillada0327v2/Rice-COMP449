import torch
import pandas as pd
import pickle
import gc
from transformers import BertTokenizer, BertModel
from utils import *

def process_text(tokenizer, model, context):
    """
    Process a given text context through a BERT model and return the [CLS] token embedding.
    
    This function tokenizes the input context, truncates it to fit the model's maximum input size if necessary,
    and processes it through the BERT model to obtain the [CLS] token embedding.
    
    Parameters:
    - tokenizer: The tokenizer corresponding to the BERT model.
    - model: The BERT model to process the text.
    - context: The text context to be processed.
    
    Returns:
    - A tensor representing the [CLS] token embedding.
    """
    tokens = tokenizer.tokenize(context)
    chunk_size = 512 - 2  # Account for [CLS] and [SEP]
    chunk_tokens = tokens[:chunk_size]
    chunk_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + chunk_tokens + ['[SEP]'])
    input_ids = torch.tensor([chunk_ids]).to(model.device)
    attention_mask = torch.tensor([[1] * len(chunk_ids)]).to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    
    return cls_embedding


def generate_embeddings_dict(tokenizer, model, df):
    """
    Generate embeddings for each row in a DataFrame and save them to a file.
    
    Processes each text in the DataFrame through a BERT model to obtain [CLS] token embeddings
    and saves these embeddings in a pickle file.
    
    Parameters:
    - tokenizer: The tokenizer corresponding to the BERT model.
    - model: The BERT model used for generating embeddings.
    - df: DataFrame containing the texts to be processed in a column named 'cited'.
    """
    abstract_dict = {idx: row['cited'] for idx, row in df.iterrows()}
    abstract_embeddings = {}
    for key, abstract in abstract_dict.items():
        abstract_embeddings[key] = process_text(tokenizer, model, abstract).cpu().numpy()
        gc.collect()
    
    filename = 'full_cls_abstract_embeddings.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(abstract_embeddings, file)

def generate_missing_embeddings(tokenizer, model):
    """
    Generate a [CLS] token embedding for an empty context and save it to a file.
    
    This function is used to generate an embedding for missing or null contexts by processing an empty string
    through the BERT model.
    
    Parameters:
    - tokenizer: The tokenizer for the BERT model.
    - model: The BERT model.
    """
    inputs = tokenizer("", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    missing_cls_embeddings = outputs.last_hidden_state[:, 0, :]
    missing_embeddings = missing_cls_embeddings.squeeze(0).cpu().numpy()
    
    filename = 'missing_cls_abstract_embeddings.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(missing_embeddings, file)

        
def get_target_embeddings(df):
    """
    Load embeddings from a pickle file and map them to target IDs in the DataFrame.
    
    This function reads abstract embeddings from a file, processes them to adjust dimensions,
    and then aggregates embeddings for each target ID present in the DataFrame.
    
    Parameters:
    - df: DataFrame containing 'target_id' columns.
    
    Returns:
    - A dictionary mapping each 'target_id' to its aggregated embedding.
    """
    
    # Load the abstract embeddings from a file
    with open('full_cls_abstract_embeddings.pkl', 'rb') as file:
        abstract_embeddings = pickle.load(file)

    # Adjust the dimensions of the embeddings
    abstract_embeddings = {key: val.squeeze(0) for key, val in abstract_embeddings.items()}
        
    df['embeddings'] = df.index.map(abstract_embeddings.get)
    
    # # Aggregate embeddings for each 'target_id'
    target_embeddings = {}
    for i in range(len(df)):
        target_id = df.iloc[i]['target_id']
        if target_id not in target_embeddings:
            target_embeddings[target_id] = df.iloc[i]['embeddings']
        else:
            # Assuming aggregation means summing embeddings
            target_embeddings[target_id] += df.iloc[i]['embeddings']
    return target_embeddings

def get_missing_embeddings():
    """
    Load the missing embeddings from a pickle file.
    
    This function reads the embeddings for missing or null contexts,
    which can be used as a fallback for missing data.
    
    Returns:
    - The missing embeddings as a numpy array.
    """
    with open('missing_cls_abstract_embeddings.pkl', 'rb') as file:
        missing_cls_embeddings = pickle.load(file)
    return missing_cls_embeddings.squeeze()


if __name__ == "__main__":
    set_seeds(42)
    
    # Load the data
    df = pd.read_csv('full_context_PeerRead.csv')

    # Preprocess and combine text
    df['cited'] = df['left_citated_text'] + df['right_citated_text']
    
    # Load pre-trained model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    model.eval()
    generate_embeddings_dict(df) # get embeddings for cited context
    generate_missing_embeddings(tokenizer, model) # get missing embeddings 
    