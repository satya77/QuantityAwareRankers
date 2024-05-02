#!/usr/bin/env python
import pickle
from models.semantic_matchers.colbert import Indexer
from models.semantic_matchers.colbert.infra import Run, RunConfig, ColBERTConfig
from models.semantic_matchers.colbert import Searcher
import torch
from models.semantic_matchers.colbert.utils import utils
import time
import math
from tqdm import tqdm
import pandas as pd
from ast import literal_eval
from collections import defaultdict
import numpy as np
from pathlib import Path
import os
"""
Implementation of the ColBERT ranking based on their repo colbert- https://github.com/stanford-futuredata/ColBERT/
"""

#remove cache file if it does not load the model from file correctly
#For GPU usage incase the ninja fails to load the c++ files
#CUDAVER=cuda-11.2
# export PATH=/usr/local/$CUDAVER/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/$CUDAVER/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/$CUDAVER/lib64:$LD_LIBRARY_PATH
# export CUDA_PATH=/usr/local/$CUDAVER
# export CUDA_ROOT=/usr/local/$CUDAVER
# export CUDA_HOME=/usr/local/$CUDAVER
# export CUDA_HOST_COMPILER=/usr/bin/gcc-10
class QColBERT_Ranker:
    def __init__(self, corpus, model_name='./data/model_weights/colbertv2.0', load_cache=False, cache_path="./models_weights/embeddings_DPR",sent_col="sentence",rank=2):
        '''

        :param model_name: the name of the DPR model to use by default facebook/dpr-ctx_encoder-single-nq-base
        :param corpus: the corpus of sentences
        :param load_cache: if set to true it will load the pre-computed embedding from file
        :param cache_path : path to the embedding cache
        :normalize_sentences: whether to use normalized sentences

        '''
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name=model_name
        self.corpus_path = corpus
        self.cache_path = cache_path
        self.index_creation_time=None
        self.map_unit_num = []  # unit and numbers for each document
        self.sent_col=sent_col
        self.rank=rank
        self.experiment=cache_path.split("/")[-1]
        if not load_cache:
            self._initialize()
        with Run().context(RunConfig(nranks=self.rank, experiment=self.experiment)):

            self.config = ColBERTConfig(root=self.cache_path )
            self.searcher = Searcher(index=self.cache_path.split("/")[-1], config=self.config,data_type=self.sent_col,collection=self.corpus_path)
            self.collection=self.searcher.collection

        if load_cache:
            self.map_unit_num,self.index_creation_time = pickle.load(open(self.cache_path+"_num", "rb"))

    def get_number_score_ratio(self, i, unit_num, handler, amount, unit):

        if unit not in unit_num:
            return 0

        if unit == None:  # if no unnit is given
            units_in_doc = [item for sublist in unit_num.values() for item in sublist]
        else:
            units_in_doc = unit_num.get(unit)

        if len(units_in_doc) == 0:  # if there is no unit detected in the documents
            return 0
        if handler == "=":
            dist = 0
            for number in units_in_doc:  # multiplied by position in do
                if number != 0 and amount != 0:
                    difference = amount - number
                    dist = dist+ math.exp(-1*abs(difference))
            return dist / len(units_in_doc)  # normalize by the number of units

        elif handler == ">":

            dist = 0
            for number in units_in_doc:
                difference = number - amount
                if number != 0:
                    dist = dist + (difference > 0) * (amount / number)
            return dist / len(units_in_doc)  # normalize by the number of units

        elif handler == "<":
            dist = 0
            for number in units_in_doc:
                difference = amount - number
                if amount != 0:
                    dist = dist + (difference > 0) * (number / amount)

            return dist / len(units_in_doc)  # normalize by the number of units

        elif handler == "<<":
            amount_bigger = amount[0]
            amount_smaller = amount[1]
            dist = 0
            for number in units_in_doc:
                amount_avg = (amount_smaller + amount_bigger) / 2.0
                dist = dist + math.exp(-1*abs(number-amount_avg))
            return dist / len(units_in_doc)  # normalize by the number of units


    def _initialize(self):
        print("creating index...")
        Path(os.path.dirname(self.cache_path)).mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        with Run().context(RunConfig(nranks=1, experiment=self.experiment)):

            config = ColBERTConfig(
                nbits=2,
                root=self.cache_path,
            )

            indexer = Indexer(checkpoint=self.model_name, config=config,data_type=self.sent_col)
            indexer.index(name=self.cache_path.split("/")[-1], collection=self.corpus_path,overwrite=True)

        df = pd.read_csv(self.corpus_path)
        for i, line in tqdm(df.iterrows()):
            value_list = literal_eval(line["values"])
            unit_list = literal_eval(line["units"])
            unit_num = defaultdict(list)
            for value,unit in zip(value_list,unit_list):
                unit_num[unit].append(value)  # create an index for the unit and values for each document
            self.map_unit_num.append(unit_num)
        end_time = time.time()
        print("creating the QColBERT features took :", utils.secondsToText(end_time - start_time))#creating the QColBERT features took : 5.0 minutes, 41.25297 seconds,
        self.index_creation_time=utils.secondsToText(end_time - start_time)
        pickle.dump((self.map_unit_num,self.index_creation_time),open(self.cache_path+"_num", "wb"))


    def _build_query(self,query,handler, amount, unit):

        query_text = query if not isinstance(query,list) else " ".join(query)
        unit_text = unit if not isinstance(unit,list) else " ".join(unit)

        amount_text=str(amount)
        if amount_text.endswith(".0"):
            amount_text=amount_text[:-2]
        new_text=""
        if handler=="=":
            new_text=query_text+" "+amount_text+ " "+unit_text
        elif handler==">":
            new_text=query_text+" greater than "+amount_text+ " "+unit_text
        elif handler=="<":
            new_text=query_text+" smaller than "+amount_text+ " "+unit_text
        elif handler=="<<":
            new_text=query_text+" between "+str(amount[0])+ " and "+str(amount[1])+" "+unit_text
        return new_text

    def get_top_n(self, query  , handler=None, amount=None, unit=None,n=5,overload=None,weight=1):
            """
            Return the top n based on the cosine similiarity of the query with the documents.
            :param query: the keywords in the query
            :param n: the number of top results to return
            :param handler: the handler : equal, greater than, smaller than or a range (ignored)
            :param amount: the exact amount or a list of two numbers for a range(ignored)
            :param unit: the unit to look for(ignored)
            :return: the top n results
            """


            pids,_,scores =self.searcher.search(query,k=1000)
            max_score=np.max(scores)
            scores=scores/max_score
            overloaded_results=[]
            new_scores= []
            for s,id in zip(scores,pids):
                num_score = weight * self.get_number_score_ratio(id, self.map_unit_num[id], handler, amount, unit)
                new_scores.append((id,s+num_score))
            new_scores.sort(key=lambda x: x[1],reverse=True)
            for id,s in new_scores[:n]:
                if overload :
                    overloaded_results.append({"doc_id":id,"doc":self.collection[id],"score":s})

            if overload:
                return overloaded_results
            return [self.collection[id] for id in pids]
