---

# Citation Recommendation System

This project demonstrates a citation recommendation system using Node2Vec and BERT+GNN models on the PeerRead dataset. It provides instructions for preprocessing data, training models, and evaluating their performance.

## Installation

Ensure you have Python 3.9 + installed. Clone this repository and navigate to the project directory. Install dependencies using:

```bash
pip install -r requirements.txt
```

## Contents

* arxivCS_data_retrieval: Functions to extract data from arxivCS to build a dataset. 
* data: The folder to put the data
* model:
  * fine-tuning BERT
  * BERT Embeddings
  * GNN model(GCN, GAT)
  * baseline model using Node2vec
  * Data pre-processing and evaluation metric functions
  

## Data Preparation

1. [Full Context PeerRead](https://bert-gcn-for-paper-citation.s3.ap-northeast-2.amazonaws.com/PeerRead/full_context_PeerRead.csv)

   Download the dataset from this link.

2. Place the dataset in the `data/` directory. This project uses `full_context_PeerRead.csv` as an example.

   Columns:

   | Header                              |                    Description                    |
   | :---------------------------------- | :-----------------------------------------------: |
   | <strong>target_id</strong>          |                  citing paper id                  |
   | <strong>source_id</strong>          |                  cited paper id                   |
   | <strong>left_citated_text</strong>  | text to the left of the citation tag when citing  |
   | <strong>right_citated_text</strong> | text to the right of the citation tag when citing |
   | <strong>target_year</strong>        |             release target paper year             |
   | <strong>source_year</strong>        |             release source paper year             |


## Running the Notebook

To view and run the project:

1. Ensure you've installed Jupyter Notebook or JupyterLab.
2. Open `run.ipynb` in Jupyter Notebook/Lab:

```bash
jupyter notebook run.ipynb
```
or
```bash
jupyter lab run.ipynb
```

3. Execute the cells sequentially to preprocess data, train models, and evaluate their performance.

---

