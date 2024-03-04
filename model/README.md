Requirements:
* Python 3
* Have downloaded unzipped directory of citation data in this same directory (https://www.dropbox.com/s/iltvodnh2mldgub/dss.tar.gz?dl=0)

## run.py 

 The main script to get result.

```python
python3 run.py [options] 
```

* General Parameters:
  * `--method` (Required): train model or get the evaluate result. Possible values: `train` or `eval`

## bert_embeddings.py 

 The main script to get bert_embeddings

```python
python bert_embeddings.py 
```

* General Parameters:
  * `--model` (Required): The mode to run the `run_classifier.py` script in. Possible values: `bert` or `bert_gcn`
  * `--dataset` (Required): The dataset to run the `run_classifier.py` script in. Possible values: `AAN` or `PeerRead`
  * `--frequency` (Required): Parse datasets more frequently
  * `--max_seq_length` : Length of cited text to use 
  * `--gpu` : The gpu to run code

* BERT Parameters:
  You can refer to it [here](https://github.com/google-research/bert).
  * `--do_train`, `--do_predict`, `--data_dir`, `--vocab_file`, `--bert_config_file`, `--init_checkpoint`, ...

