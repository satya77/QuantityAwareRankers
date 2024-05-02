# Lexical Matchers
This module contains code for lexical matches, which are variants of the BM25 model.
We briefly explain each.

## Okapi BM25
`BM25_base.py` contains the code for an Okapi BM25 implementation, mainly adapted from https://github.com/dorianbrown/rank_bm25.
This is the base model without any alteration that performs on top of corpus statistics.

## Filter BM25
`bm25_filter.py` contains code for an Okapi BM25 implementation including a quantity index.
The quantity index is supposed to filter the results of BM25 to filter the ones that do not
meet the query numerical condition.


## QBM25
`qbm25.py` contains code for a quantity-aware BM25 variant. Similar to the filter variant,
we create a quantity index, where value and unit pairs point to sentences that contain them.
However, this index is used for ranking and not filtering.
The quantity score based on heuristic functions is added to the normalized BM25 score.


## General Structure
All BM25 models inherit from the `BM25` class that computes the corpus statistics and should implement:

- `_initialize`: initialize the class, loads model from catch, or computes statistics from the corpus.
  For initialization, you can specify the parameters of BM25, namely `k1` and `b`. You also need to provide a corpus,
  using `dataset.bm25_dataloaders`, and specify if you want to load the model from the cache, using `load_cache` and defining a `cache_path`.
  QBM25 has one additional parameter, which is `ranker_type` specifying whether to use ratios or exponentials for quantity scoring functions.
- `get_scores`: computes the score and ranking for documents given a query.
- `load_cache`: load the saved models from cache.
- `cache`: save the indexes to a file for later use.
- `get_top_n`: returns the top-n results from the scoring. All the functions accept: `query` for query text, `handler` for numerical condition
  , `amount` for value and  `unit` for quantity unit. By default, they are set to `None` and have no
  impact on the scoring of the OkapiBM25. 
