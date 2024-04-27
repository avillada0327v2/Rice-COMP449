---

# Research Paper Recommendation System

This project demonstrates a citation recommendation system using Node2Vec and BERT+GNN models on the PeerRead dataset. It provides instructions for preprocessing data, training models, and evaluating their performance.

![PaperPilotLogo](https://github.com/avillada0327v2/Rice-COMP449/assets/142918860/8a23ddf3-61bf-4c6d-9c77-e810103795df)

## Paper Pilot Team Members

Sharath Giri, Judy Fang, Jerry Jiang, Jacky Jiang, James Murphy, & Andres Villada

## Repository Structure
```bash
Rice-COMP449/
├── data
│   └── 
├── model
│   └── autoencoder
├── D2K Poster.pptx
│   
│
└── README.md
│
└── requirements.txt
└── run.ipynb

```

## Installation

Ensure you have Python 3.9 + installed. Clone this repository and navigate to the project directory. Install dependencies using:

```bash
pip install -r requirements.txt
```

## Contents

* `arxivCS_data_retrieval`: Functions to extract data from arxivCS to build a dataset. 
* `data`: The folder to put the data.
* `model`: This folder contains the pre-trained models and embeddings.
  * `fine-tuning BERT`
  * `BERT Embeddings`
  * `GNN model (GCN, GAT)`
  * `baseline model using Node2vec`
  * `Data pre-processing and evaluation metric functions`

To save time and not re-run BERT and the GNN model, you can download the pre-computed embeddings and models from here:
```
https://www.dropbox.com/scl/fi/65bp163njgpyogu5y94z7/resources.zip?rlkey=dl1gqvcybqugykrq8rwmowc6a&dl=0
```
Please download and place the content in the `model/` folder.

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
