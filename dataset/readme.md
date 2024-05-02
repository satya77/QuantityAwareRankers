#Dataset

Contains a collection of datasets and dataloaders used by models in this package.

- `bm25_dataloaders.py` contains a dataloader for the corpus and test queries to be used with the lexical models.
- `splade_dataloaders.py` and `splade_dataset.py` contain a dataloader and dataset class
  for loading a textual corpus or one with a numerical index for the splade models.

Data loaders for the ColBERT model are inside the respective module. 
