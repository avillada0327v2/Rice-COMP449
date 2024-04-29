---

# Research Paper Recommendation System

This project demonstrates a citation recommendation system using Node2Vec and BERT+GNN models on the PeerRead dataset. It provides instructions for preprocessing data, training models, and evaluating their performance.

![PaperPilotLogo](https://github.com/avillada0327v2/Rice-COMP449/assets/142918860/8a23ddf3-61bf-4c6d-9c77-e810103795df)

## Paper Pilot Team Members

Sharath Giri, Judy Fang, Jerry Jiang, Jacky Jiang, James Murphy, & Andres Villada

## Installation

Ensure you have Python 3.9 + installed. Clone this repository and navigate to the project directory. Install dependencies using:

```bash
pip install -r requirements.txt
```

## Directory and Contents

* `data`: Folder to put data as well as data wrangling, scripting, and analysis.
  * `arxivCS_data_retrieval`: Functions to extract data from arxivCS to build a dataset.
    * `utils`: Folder that holds utility functions for arxivCS dataset wrangling and scraping script
      * `node.py`: Node functions for trie data structure used for wrangling and scraping
      * `paper_parser_utils.py`: Utility functions for column data processing the arxivCS dataset
      * `tree.py`: Trie functions for trie operations
    * `README.md`: Document outlining how to use the arxivCS data wrangling and scraping script
    * `data_wrangle_script.py`: Data wrangler and compiler for ArxivCS source and destination citation dataset.
  * `visual_explorations`: Notebooks outlining exploratory data analysis on datasets
    * `baseline_model_w_visuals.ipynb`: Notebook outlining baseline model compilation with data visuals
    * `peerread_data_pagerank.ipynb`: Notebook outlining various visualizations such as page rank distribution and graph communities
    * `topic_analysis.ipynb`: Notebook outlining various topic analysis functionalities such as word cloud and LDA
* `model_architecture`: This folder contains the pre-trained models and embeddings.
  * `bert_embeddings.py`: BERT embedding aggregation and functionality
  * `gnn.py`: Builds, runs and evaluates a GNN model using GCN layers for link prediction.
  * `node2vec.py`: Builds, runs and evaluates a node2vec model's ability to perform link prediction.
  * `utils.py`: Utility functions used throughout model architecture and evaluation
* `D2K Poster.pptx`: Poster presented for the D2K Spring 2024 Showcase
* `README.md`: Document being read that outlines the purpose and use of this repository
* `requirements.txt`: Text file to be run for all functionality requirement handling in this repository
* `run.ipynb`: Demo for model comparison via performance metrics of research paper recommendations

## Data Preparation

1. [Full Context PeerRead Dataset](https://bert-gcn-for-paper-citation.s3.ap-northeast-2.amazonaws.com/PeerRead/full_context_PeerRead.csv)

   Download the dataset from this link.

2. Place the dataset in the `data/` directory. Use `full_context_PeerRead.csv` as an example dataset.

   Dataset columns:

   | Header                              |                    Description                    |
   | :---------------------------------- | :-----------------------------------------------: |
   | <strong>target_id</strong>          |                  Citing paper ID                  |
   | <strong>source_id</strong>          |                  Cited paper ID                   |
   | <strong>left_citated_text</strong>  | Text to the left of the citation tag when citing  |
   | <strong>right_citated_text</strong> | Text to the right of the citation tag when citing |
   | <strong>target_year</strong>        |             Release year of citing paper          |
   | <strong>source_year</strong>        |             Release year of cited paper           |

## Model Preparation

To save time and not re-run BERT and the GNN model, you can download the pre-computed embeddings and models from here:
```
https://www.dropbox.com/scl/fi/65bp163njgpyogu5y94z7/resources.zip?rlkey=dl1gqvcybqugykrq8rwmowc6a&dl=0
```
Please download and place the content in the `model/` folder.

## Running the Notebook

To view and run the project:

1. Ensure you have Jupyter Notebook or JupyterLab installed.
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
