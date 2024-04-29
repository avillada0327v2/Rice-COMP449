import pandas as pd
import pickle
import gc  # Garbage Collector interface
import os
import torch
import random
import numpy as np
import warnings
from transformers import BertTokenizer, BertModel

warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(tokenizer, model, context):
    """
    Process a given text context through a BERT model and return the [CLS] token embedding.
    
    This function tokenizes the input context, truncates it to fit the model's maximum input size if necessary,
    and processes it through the BERT model to obtain the [CLS] token embedding.
    
    Parameters:
    - tokenizer (BertTokenizer): The tokenizer corresponding to the BERT model.
    - model (BertModel): The BERT model to process the text.
    - context (str): The text context to be processed.
    
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


def generate_source_abstract_embeddings_dict(tokenizer, model, df):
    """
    Generate embeddings for each source_id in a DataFrame and save them to a file.
    
    Processes each source abstract in the DataFrame through a BERT model to obtain [CLS] token embeddings
    and saves these embeddings in a pickle file.
    
    Parameters:
    - tokenizer (BertTokenizer): The tokenizer corresponding to the BERT model.
    - model (BertModel): The BERT model used for generating embeddings.
    - df (pd.DataFrame): DataFrame containing the texts to be processed in a column named 'source_abstract'.
    """
    abstract_dict = dict(zip(df['source_id'].unique(), df['source_abstract'].unique()))
    abstract_embeddings = {}
    for key, abstract in abstract_dict.items():
        abstract_embeddings[key] = process_text(tokenizer, model, abstract).cpu().numpy()
        gc.collect()
    
    filename = 'model/full_cls_source_abstract_embeddings.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(abstract_embeddings, file)

        
def get_source_abstract_embeddings():
    """
    Load the missing embeddings from a pickle file.
    
    This function reads the embeddings for missing or null contexts,
    which can be used as a fallback for missing data.
    
    Returns:
    - The missing embeddings as a numpy array.
    """
    with open('model/full_cls_source_abstract_embeddings.pkl', 'rb') as file:
        abstract_embeddings = pickle.load(file)
        
    # Adjust the dimensions of the embeddings
    abstract_embeddings = {key: val.squeeze(0) for key, val in abstract_embeddings.items()}
    return abstract_embeddings


def generate_citated_text_embeddings_dict(tokenizer, model, df):
    """
    Generate embeddings for each row in a DataFrame and save them to a file.
    
    Processes each text in the DataFrame through a BERT model to obtain [CLS] token embeddings
    and saves these embeddings in a pickle file.
    
    Parameters:
    - tokenizer (BertTokenizer): The tokenizer corresponding to the BERT model.
    - model (BertModel): The BERT model used for generating embeddings.
    - df (pd.DataFrame): DataFrame containing the texts to be processed in a column named 'cited'.
    """
    citated_text_dict = dict(zip(df['citated_text_id'].unique(), df['citated_text'].unique()))
    citated_embeddings = {}
    for key, abstract in citated_text_dict.items():
        citated_embeddings[key] = process_text(tokenizer, model, abstract).cpu().numpy()
        gc.collect()
        
    filename = 'model/full_cls_citated_text_embeddings.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(citated_embeddings, file)   

        
def get_target_embeddings(df):
    """
    Load embeddings from a pickle file and map them to target rows in the DataFrame.
    
    This function reads abstract embeddings from a file, processes them to adjust dimensions,
    and then aggregates embeddings for each target ID present in the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing 'target_id' columns.
    
    Returns:
    - A dictionary mapping each 'target_id' to its aggregated embedding.
    """
    
    # Load the cited embeddings from a file
    with open('model/full_cls_citated_text_embeddings.pkl', 'rb') as file:
        citated_embeddings = pickle.load(file)

    # Adjust the dimensions of the embeddings
    citated_embeddings = {key: val.squeeze(0) for key, val in citated_embeddings.items()}
           
    # # Aggregate embeddings for each 'target_id'
    target_embeddings = {}
    for i in range(len(df)):
        target_id = df.iloc[i]['target_id']
        citated_text_id = df.iloc[i]['citated_text_id']
        if target_id not in target_embeddings:
            target_embeddings[target_id] = citated_embeddings.get(citated_text_id)
        else:
            # Assuming aggregation means summing embeddings
            target_embeddings[target_id] += citated_embeddings.get(citated_text_id)
    return target_embeddings        


def generate_missing_embeddings(tokenizer, model):
    """
    Generate a [CLS] token embedding for an empty context and save it to a file.
    
    This function is used to generate an embedding for missing or null contexts by processing an empty string
    through the BERT model.
    
    Parameters:
    - tokenizer (BertTokenizer): The tokenizer corresponding to the BERT model.
    - model (BertModel): The BERT model used for generating embeddings.
    """
    inputs = tokenizer("", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    missing_cls_embeddings = outputs.last_hidden_state[:, 0, :]
    missing_embeddings = missing_cls_embeddings.squeeze(0).cpu().numpy()
    
    filename = 'model/missing_cls_cited_embeddings.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(missing_embeddings, file)

        
def get_missing_embeddings():
    """
    Load the missing embeddings from a pickle file.
    
    This function reads the embeddings for missing or null contexts,
    which can be used as a fallback for missing data.
    
    Returns:
    - The missing embeddings as a numpy array.
    """
    with open('model/missing_cls_cited_embeddings.pkl', 'rb') as file:
        missing_cls_embeddings = pickle.load(file)
    return missing_cls_embeddings.squeeze()
        

def generate_embeddings(df):
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
    
    # Load pre-trained model and tokenizer
    model_name = 'allenai/scibert_scivocab_uncased'#'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).eval()
    
    # Assuming each function below properly handles batch processing and saves embeddings to disk.
    generate_source_abstract_embeddings_dict(tokenizer, model, df)  # Embeddings for source abstract
    generate_citated_text_embeddings_dict(tokenizer, model, df)  # Embeddings for cited context
    generate_missing_embeddings(tokenizer, model)  # Embeddings for missing abstracts or contexts

    