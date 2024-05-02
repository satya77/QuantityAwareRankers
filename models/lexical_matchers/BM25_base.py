
#!/usr/bin/env python
"""
This script contains the code for the Okapi BM25
"""
import time
from evaluate.utils import utils
import math
import  numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
import os
"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined
and the github repo :https://github.com/dorianbrown/rank_bm25
we made minor change to adapt the algorithm to our needs 
"""


class BM25:
    def __init__(self, corpus_generator, cache_path, load_cache=False,k1=2, b=0.1, epsilon=0.25):  # we added a caching option to avoid computing the indicies everytime
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.cache_path = cache_path
        self.docs=[]
        self.corpus = corpus_generator
        self.cache_path = cache_path
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.index_creation_time=None

        if load_cache:
            self.load_cache()
        else:
            start_time = time.time()
            nd = self._initialize()
            self._calc_idf(nd)
            end_time = time.time()
            print("creating the features took :", utils.secondsToText(end_time - start_time))  #
            self.cache()
            self.index_creation_time= utils.secondsToText(end_time - start_time)

    def add_to_freq(self,word,frequencies):
        if word not in frequencies:
            frequencies[word] = 0
        frequencies[word] += 1
        return frequencies

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_top_n(self, query, n=5, handler=None, amount=None, unit=None, overload=False):

        assert self.corpus_size == len(self.docs), "The documents given don't match the index corpus!"

        scores = self.get_scores(query, handler, amount, unit)

        top_n = np.argsort(scores)[::-1][:n]
        result=[]
        if overload:
            for i in top_n:
                result.append({"doc_id":i,"doc":self.docs[i],"score":scores[i]})
            return result

        else:
            return [self.docs[i] for i in top_n]


    # def retrieval_test(self):

    def _initialize(self):
        raise NotImplementedError()

    def get_scores(self, query, handler, amount, unit):
        raise NotImplementedError()

    def cache(self):
        raise NotImplementedError()
    def load_cache(self):
        raise NotImplementedError()

class BM25Okapi(BM25):
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25, load_cache=False, cache_path="../models_weights/bm25"):
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
            for word in document:
                frequencies=self.add_to_freq(word,frequencies)
            #add the numbers to the word index to increase the chance of a match
            for quant in quants:
                unit, value=quant
                frequencies=self.add_to_freq(unit,frequencies)
                frequencies=self.add_to_freq(str(value),frequencies)

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
            (self.corpus_size, self.avgdl, self.doc_freqs, self.idf, self.average_idf, self.doc_len, self.docs,self.index_creation_time),
            open(self.cache_path, "wb"))

    def load_cache(self):
        print("Loading from cache. ")
        self.corpus_size, self.avgdl, self.doc_freqs, self.idf, self.average_idf, self.doc_len,self.docs,self.index_creation_time= pickle.load(
            open(self.cache_path, "rb"))



    def get_scores(self, query,n=5, handler=None, amount=None, unit=None):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

