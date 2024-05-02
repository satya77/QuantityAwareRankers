# Data Generation for Joint Models
This module contains code for concept expansion on the queries (refer to the sub package`concepts`) and
query and sample generation pipeline. The order of the script is designated in the numbering.

## Transform Data
`01-transform_query_data.py`:

The pipeline starts with take a collection of sentences
and a CSV file containing unit/concept pairs and a list of values and sentences associated with them,
and generate dictionaries for easy access.
These files can be downloaded through the provided resources.
Basically, the data is transformed into a dictionary of dictionaries.
On the first level unit/concept pairs are the keys, pointing to a dictionary of values
associated with them, where the values of the dictionary are the list of sentences that
have that particular value.
This transformation is essential for the next steps.

Additionally, there are two versions of the collection created with masked units and
mask values used for evaluation of the joint models in the `evaluate` module.

Here is a sample on to run this script:
* `data-type`: either finance or clinical 
* `input-queries `: a csv file that has a column `keyword` indicating the concpets and `units` indicating units. It points to `values` and `sentences` and their positions in text with `values_char` and `unit_char`
* `input-collection`: input collection of all sentences in the corpus
* `generate-masked-value`: whether to generate a masked value collection
* `generate-masked-unit`: whether to generate a masked unit collection
* `output-path`: path to output folder, where the generation output will go
```
python 01-transform_query_data.py --data-type finance --input-queries ../data/finance/train_concepts_units_extended.csv --input-collection ../data/finance/collection.csv --generate-masked-value True  --generate-masked-unit True --output-path ../data/finance
```

## Generate Queries and Samples
`02-generate_data.py`:

This script is responsible for the data generation.
`02-generate_data.py` requires the output of `01-transform_query_data.py`  as well output of
the `concepts` module for data generation.

If one does not want to apply concept expansion, one can leave the output of the concept module `concepts` out.
The output of `02-generate_data.py` is a set of queries, an augment collection, and relevance in
terms of triplets (a positive and negative sample per query).
Data is generated in a variety of formats (refer to `02-generate_data.py`) to accommodate different
models and their input requirements.
An example output is as follows:
```
#training set 

all_queries_aug.tsv
collection_aug.tsv
jsonl_aug.json
jsonl_triplet_aug.json
qrel_aug.tsv
qrel_json_aug.json
queries_aug.tsv
triplets_aug.tsv

# validation data set 
val_collection_aug.tsv
val_qrel_aug.json
val_qrel_aug.tsv
val_queries_aug.tsv
```
For a detailed algorithm of how the query and sample generation is performed refer to the paper or comments
in the script.

`helper_functions.py`:
This script contains augmentation functions and helper functions used in `02-generate_data.py`.

`config_classes.py`:
This script contains classes and structures for passing data between the functions in `02-generate_data.py`.
The `GenerationConf` class defines all the parameters for data generation and `BatchSentences`
contains the positive and negative samples generated. Finally, `GenerationInput` is an abstraction
for the common inputs during data generation.

Here is a sample on to run this script: 
* `path-to-queries`: path to the pickled file containing concepts,units index from the pervious script
* `output-folder`: folder where the generated training data is stored
* `path-to-concepts`: path to the pickled file of all the concepts and their extensions
* `path-to-collection`:path to the collection of sentences in the corpus
* `path-to-validation`:path to a folder for validation set, if set to None, no validation set will be generated
* `samples-size`:number of samples to be generated
* `extend-concepts`:whether to apply concept expansion during query generation
* `permute-unit`:whether to apply unit permutation for sample generation
* `permute-value`:whether to apply value permutation for sample generation
```
python 02-generate_data.py --path-to-queries ../data/finance/querie_indicies_dict_extended.pickle --output-folder ../data/finance/generation_output --path-to-concepts ../data/finance/concepts_to_extentions.pickle --path-to-collection ../data/finance/collection.csv --samples-size 2 --extend-concepts  --permute-unit  --permute-value 
```