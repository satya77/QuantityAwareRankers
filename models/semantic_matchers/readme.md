# Semantic Matchers
This module contains code for semantic matches, which are variants of the ColBERT and SPLADE models.
We briefly explain each.
Our semantic matcher contains disjoint and joint models. For training of the joint models
we refer to the repsective repositories of [ColBERT](https://github.com/stanford-futuredata/ColBERT) and [SPLADE](https://github.com/naver/splade?tab=readme-ov-file).
The code here is only for inference.

## ColBERT Rankers
The module `colbert` contains code from the respective repository without change for running
inference on a fine-tuned ColBERT model.
- `ColBERT_ranker.py` is the interface for the joint ColBERT model or base model used as a baseline
  in the evaluation. It loads trained checkpoints and returns the top-k results using FAISS.
- `QColBERT_ranker.py` is the interface for the disjoint ColBERT model, with an addition
  of a quantity index that performs reranking on the top-k results from ColBERT.

While running the ColBERT models on GPU and CUDA 11.2, if ninja fails to load the C++ files:
```
For GPU usage in case the 
 CUDAVER=cuda-11.2
 export PATH=/usr/local/$CUDAVER/bin:$PATH
 export LD_LIBRARY_PATH=/usr/local/$CUDAVER/lib:$LD_LIBRARY_PATH
 export LD_LIBRARY_PATH=/usr/local/$CUDAVER/lib64:$LD_LIBRARY_PATH
 export CUDA_PATH=/usr/local/$CUDAVER
 export CUDA_ROOT=/usr/local/$CUDAVER
 export CUDA_HOME=/usr/local/$CUDAVER
 export CUDA_HOST_COMPILER=/usr/bin/gcc-10
```

The trained checkpoints for ColBERT can be requested from the authors through their repository.

## SPLADE Rankers
The module `SPLADE` contains code from the respective repository with additional code for a
numerical sparse index and a combination of quantity scoring with SPLADE score.
- `SPLADE_ranker.py` is the interface for the joint SPLADE model or base model used as a baseline
  in the evaluation. It loads trained checkpoints and returns the top-k results using the sparse dot product.
- `QSPLADE_ranker.py` is the interface for the disjoint SPLADE model and the sparse numerical index.

SPLADE has multiple trained checkpoints, we used the best performing one `naver/splade-cocondenser-ensembledistil`,
which is also the default model that will be loaded with the scripts.

## General Structure
All the rankers have a similar interface to the lexical models.
- `_initialize`: initialize the class, loads model from catch, or computes statistics from the corpus.
  Each model also requires the path to a model checkpoint to be loaded. After initialization, we cache the
  indexes such that we do need to re computed them.
- `get_scores`: computes the score and ranking for documents given a query.
- `load_cache`: load the saved models from cache.
- `cache`: save the indexes to a file for later use.
- `get_top_n`: returns the top-n results from the scoring. All models accept `query` for query text.
  For the disjoint models, we have additional inputs as  `handler` for numerical condition, `amount` for value, and  `unit` for quantity unit.
