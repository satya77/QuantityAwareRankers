# Evaluation

This folder contains code for the evaluation of the quantity-aware retrieval systems and statistical testing.

Overview: 

- `test_cohere.py` contains code for connecting to Cohere API and indexing the collection. 
Then running evaluation similiar to `evaluate.py`.
- `permutation_significance_test.py` contains the code for permutation resampling, and testing
a model with the metrics `P@10`, `NDCG@10`, `R@10`, and `MRR@10` for significant improvements over a
baseline. To use this script you would require the rankings of a test model set in the form of a
JSON file. `evaluate.py` provides such an output.

- `evaluate.py` is the main evaluation script that uses the `pytrec_eval` package to evaluate all the models
  on different data splits. The output is two JSON file:
    - `model_name.json`: contains the raw predictions of the model.
    - `model_name_eval.json`: contains the evaluation metric and runtime.
      If you want to run an evaluation for a specific model, you can comment out the other rankers. 

It is possible that while evaluating the ColBERT models GCC throws an error, here is a solution that worked for us (beware of the cuda version): 
```
CUDAVER=cuda-11.2
export PATH=/usr/local/$CUDAVER/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/$CUDAVER/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/$CUDAVER/lib64:$LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/$CUDAVER
export CUDA_ROOT=/usr/local/$CUDAVER
export CUDA_HOME=/usr/local/$CUDAVER
export CUDA_HOST_COMPILER=/usr/bin/gcc-10
```
<hr>

Examples on how to run the scripts:

<hr>

`test_cohere.py`:
* `corpus-path`: the path to the sentence collection
* `test-queries-path`: path to the queries
* `api-key`: cohere api key
* `output-folder`: The folder to save the index and output of the evaluation to
* `gt-folder`: folder with ground truth relevances on different subsets in json files
* `create-index`: whether to create an index from scratch, if set to false the current index will be used for evaluation
```
python -m evaluate.test_cohere --corpus-path ./data/finance/collection.csv --gt-folder ./data/finance --test-queries-path ./data/finance/queries.tsv  --api-key aljerleawn --output-folder ./data/finance/test/ --create-index
```

<hr>

`evaluate.py`:
* `splade-ft-path`: path to fine-tuned splade model for SPLADE_ft checkpoint (.pt).
* `splade-pretrained`:path/or model name to the pretrained splade model (on general data) for QSPLADE model and SPLADE baseline.
* `colbert-ft-path`: path to fine-tuned colbert model for COLBERT_ft folder (containing all files).
* `colbert-pretrained`:path to the pretrained colbert model (on general data) for QCOLBERT model and COLBERT baseline.
* `gt-folder`: folder where the gt qrels are.
* `measures`: the measures to test, seperated by comma. 
* `corpus-path`: the path to the sentence collection.
* `queries-path`:path to the test queries.
* `output-folder`:The folder to save the index and output of the evaluation to.
* `load-cache-model`: whether to load model and index from cache.
* `load-cache-result`: whether to compute the metrics from the results in the out-folder.
* `model-cache-folder`:Where to save the model and indexes.
* `topk`:how many top-k to retrieve (choose based on the maximum value in recall).
* `bm25-param`:k1 parameter and b parameter of  bm25 seperated by comma.
* `qbm25-param`:k1 parameter and b parameter of  qbm25 seperated by comma.
* `bm25-filter-param`:k1 parameter and b parameter of  bm25 filter seperated by comma.
```
python -m evaluate.evaluate --splade-ft-path ./data/model_weights/finance/splade_ft/model_best.pt --colbert-ft-path ./data/model_weights/finance/colbert_ft  --splade-pretrained naver/splade-cocondenser-ensembledistil --colbert-pretrained ./data/model_weights/colbertv2.0
```
<hr>

`permutation_significance_test.py`:

* `sys-a`: name of system a (your system)
* `sys-b`: name of system b (systems to compare against), seperate by comma if more than one
* `gt-file`: the compelete qrel file (not the subsets)
* `input-folder`: the folder that contians the percomputed results for system a and b. The name of the systems should be same as name of the files
* `measures`: the metrics to be used for significant testing, comma seperated if more than one
```
python -m evaluate.permutation_significance_test --sys-a qcolbert --sys-b colbert --gt-file ./data/finance/qrel_compelete.json --input-folder ./data/finance/ --measures ndcg_cut@10
```
