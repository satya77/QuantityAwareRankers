#!/usr/bin/env python
"""
This script contains the code for the Okapi BM25 with an additional index to filter based on numerical conditions
"""
import pickle
from .BM25_base import BM25
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined
and the github repo :https://github.com/dorianbrown/rank_bm25
we made minor change to adapt the algorithm to our needs 
"""

class FilterBM25(BM25):
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25, load_cache=False,
                 cache_path="./models_weights/bm25_filter"):
        self.map_doc_num=[]
        super().__init__(corpus, cache_path,load_cache,k1,b,epsilon)

    def _initialize(self):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for (document, quants,original_doc) in tqdm(self.corpus):
            self.docs.append(original_doc)# we save the original passage for later
            self.doc_len.append(len(document))
            num_doc += len(document)
            self.corpus_size=self.corpus_size+1

            frequencies = {}
            doc_num = []
            for word in document:
                frequencies=self.add_to_freq(word,frequencies)

            for quant in quants:
                unit, value=quant
                doc_num.append(value)
                frequencies=self.add_to_freq(unit,frequencies)
                frequencies=self.add_to_freq(str(value),frequencies)

            self.map_doc_num.append(doc_num)
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1

        self.avgdl = num_doc / self.corpus_size
        return nd



    def cache(self):
        print("Caching the pre-computed values. ")
        Path(os.path.dirname(self.cache_path)).mkdir(parents=True, exist_ok=True)

        pickle.dump(
            (self.corpus_size, self.avgdl, self.doc_freqs, self.idf, self.average_idf, self.doc_len, self.map_doc_num,self.docs,self.index_creation_time),
            open(self.cache_path, "wb"))

    def load_cache(self):
        print("Loading from cache. ")
        self.corpus_size, self.avgdl, self.doc_freqs, self.idf, self.average_idf, self.doc_len, self.map_doc_num,self.docs,self.index_creation_time = pickle.load(
            open(self.cache_path, "rb"))


    def get_scores(self, query, handler=None, amount=None,unit=None):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        q_exits = np.ones(len(self.doc_freqs))  # indicator if all the query terms are present

        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            q_exits *= np.array(q_freq > 0).astype(int)
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        itemindex = np.where(q_exits == 1)[0]
        for i in itemindex:
            condition_satisfied = False
            numbers = self.map_doc_num[i]
            for n in numbers:
                if handler == ">" and n > amount:
                    condition_satisfied = True
                elif handler == "<" and  n < amount:
                    condition_satisfied = True
                elif handler == "<<" and n > amount[1] and n < amount[2]:
                    condition_satisfied = True
            if not condition_satisfied and len(numbers) > 0:
                score[i] = score[i] * 0
        return score
