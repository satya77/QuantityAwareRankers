#!/usr/bin/env python
from models.semantic_matchers.colbert import Indexer
from models.semantic_matchers.colbert.infra import Run, RunConfig, ColBERTConfig
from models.semantic_matchers.colbert import Searcher
import torch
from models.semantic_matchers.colbert.utils import utils
import time
from pathlib import Path
import os
"""
Implementation of the ColBERT ranking based on their repo colbert- https://github.com/stanford-futuredata/ColBERT/
"""


# remove cache file if it does not load the model from file correctly
# For GPU usage incase the ninja fails to load the c++ files
# CUDAVER=cuda-11.2
# export PATH=/usr/local/$CUDAVER/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/$CUDAVER/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/$CUDAVER/lib64:$LD_LIBRARY_PATH
# export CUDA_PATH=/usr/local/$CUDAVER
# export CUDA_ROOT=/usr/local/$CUDAVER
# export CUDA_HOME=/usr/local/$CUDAVER
# export CUDA_HOST_COMPILER=/usr/bin/gcc-10
class ColBERT_Ranker:
    def __init__(self, corpus, model_name='./data/model_weights/colbertv2.0', load_cache=False,
                 cache_path="./models_weights/embeddings_COLBERT", sent_col="sentence", rank=4):
        '''

        :param model_name: the name of the DPR model to use by default facebook/dpr-ctx_encoder-single-nq-base
        :param corpus: the corpus of sentences
        :param load_cache: if set to true it will load the pre-computed embedding from file
        :param cache_path : path to the embedding cache
        :normalize_sentences: whether to use normalized sentences

        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.corpus_path = corpus
        self.cache_path = cache_path
        self.index_creation_time = None
        self.sent_col = sent_col
        self.experiment = cache_path.split("/")[-1]
        self.rank = rank
        if not load_cache:
            self._initialize()

        with Run().context(RunConfig(nranks=self.rank, experiment=self.experiment)):
            self.config = ColBERTConfig(root=self.cache_path, index_root=self.cache_path)
            self.searcher = Searcher(index=self.cache_path.split("/")[-1], config=self.config,
                                     collection=self.corpus_path, data_type=sent_col)
            self.collection = self.searcher.collection

    def _initialize(self):
        print("creating index...")
        start_time = time.time()
        Path(os.path.dirname(self.cache_path)).mkdir(parents=True, exist_ok=True)

        with Run().context(RunConfig(nranks=self.rank, experiment=self.experiment)):
            config = ColBERTConfig(
                nbits=2,
                root=self.cache_path,
                index_root=self.cache_path
            )
            indexer = Indexer(checkpoint=self.model_name, config=config, data_type=self.sent_col)
            indexer.index(name=self.cache_path.split("/")[-1], collection=self.corpus_path, overwrite=True)

        end_time = time.time()
        print("creating the ColBERT features took :", utils.secondsToText(end_time - start_time))
        self.index_creation_time = utils.secondsToText(
            end_time - start_time)  # creating the ColBERT features took : 5.0 minutes, 19.53032 seconds,

    def _build_query(self, query, handler, amount, unit):

        query_text = query if not isinstance(query, list) else " ".join(query)
        unit_text = unit if not isinstance(unit, list) else " ".join(unit)

        amount_text = str(amount)
        if amount_text.endswith(".0"):
            amount_text = amount_text[:-2]
        new_text = ""
        if handler == "=":
            new_text = query_text + " " + amount_text + " " + unit_text
        elif handler == ">":
            new_text = query_text + " greater than " + amount_text + " " + unit_text
        elif handler == "<":
            new_text = query_text + " smaller than " + amount_text + " " + unit_text
        elif handler == "<<":
            new_text = query_text + " between " + str(amount[0]) + " and " + str(amount[1]) + " " + unit_text
        return new_text

    def get_top_n(self, query, handler=None, amount=None, unit=None, n=5, overload=None):
        """
        Return the top n based on the cosine similiarity of the query with the documents.
        :param query: the keywords in the query
        :param n: the number of top results to return
        :param handler: the handler : equal, greater than, smaller than or a range (ignored)
        :param amount: the exact amount or a list of two numbers for a range(ignored)
        :param unit: the unit to look for(ignored)
        :return: the top n results
        """

        pids, _, scores = self.searcher.search(query, k=n)
        results = {}
        overloaded_results = []
        for s, id in zip(scores, pids):
            results[id] = s
            if overload:
                overloaded_results.append({"doc_id": id, "doc": self.collection[id], "score": s})

        if overload:
            return overloaded_results
        return [self.collection[id] for id in pids]
