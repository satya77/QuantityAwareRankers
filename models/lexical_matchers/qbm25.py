#!/usr/bin/env python

import math

import numpy as np
from models.lexical_matchers.BM25_base import BM25
from collections import defaultdict
from tqdm import tqdm
import pickle
from pathlib import Path
import os
"""
Impelemenation of the QBM25 with three different numerical weighting function based on the BM25. The base of the impelementation is from Okapi BM25: https://github.com/dorianbrown/rank_bm25 
"""


class QBM25(BM25):
    def __init__(self, corpus, k1=1.2, b=0.75, epsilon=0.25, load_cache=False, cache_path="./models_weights/qbm25",ranker_type="ratio"):
        self.map_unit_num = []  # unit and numbers for each document
        self.ranker_type=ranker_type
        super().__init__(corpus,cache_path,load_cache, k1,b,epsilon)


    def _initialize(self):
        print("Initilize the index... ")
        nd = defaultdict(int)  # word -> number of documents with word
        num_doc = 0
        for (document, quants, original_doc) in tqdm(self.corpus):
            self.docs.append(original_doc)# we save the original passage for later
            self.doc_len.append(len(document))
            num_doc += len(document)
            self.corpus_size=self.corpus_size+1


            frequencies = defaultdict(int)
            unit_num = defaultdict(list)
            for word in document:
                frequencies[word] = 1  # we look at binary presence of the terms in the document

            for quant in quants:
                unit, value=quant
                frequencies[unit] = 1  # also binary presence of the unit in the document
                unit_num[unit].append(value)  # create an index for the unit and values for each document

            self.doc_freqs.append(frequencies)
            self.map_unit_num.append(unit_num)

            for word, freq in frequencies.items():
                nd[word] += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def get_scores(self, query, handler=None, amount=None, unit=None):
        """
        get QBM25 score based on the handler
        :param query: the keywords
        :param handler: the condition (=,<,<,<<)
        :param amounnt: number
        :param unit: unit of the numerical values
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        q_exits = np.ones(len(self.doc_freqs))  # indicator if all the query terms are present


        for q1 in query:

            q_freq = np.array([(doc.get(q1) or 0) for doc in self.doc_freqs])
            q_exits *= np.array(q_freq > 0).astype(int)
            score += (self.idf.get(q1) or 0) * (q_freq * (self.k1 + 1) /
                                                (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))



        max_score = np.max(score)
        if max_score!=0:
            score = score / max_score  # normalize the BM25 score

        t = 2  # weight of the numerical score
        itemindex = np.where(q_exits == 1)[0]
        if max_score==0:
            itemindex = np.where(q_exits == 0)[0]
        qscore = np.zeros(self.corpus_size)
        if handler and amount:
            for i in itemindex:
                if self.ranker_type=="ratio":
                    qscore[i] = t * self.get_number_score_ratio(i, self.map_unit_num[i], handler, amount, unit)

                else:
                    qscore[i] = t * self.get_number_score_expo(i, self.map_unit_num[i], handler, amount, unit)

        score += qscore

        return score

    def get_number_score_ratio(self, i, unit_num, handler, amount, unit):

        if isinstance(unit, str) and unit not in unit_num:
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

    def get_number_score_expo(self, i, unit_num, handler, amount, unit):

        if isinstance(unit, str) and unit not in unit_num:
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
                if (difference > 0):
                    dist = dist+ math.exp(-1*abs(difference))
            return dist / len(units_in_doc)  # normalize by the number of units

        elif handler == "<":
            dist = 0
            for number in units_in_doc:
                difference = amount - number
                if (difference > 0):
                    dist = dist+ math.exp(-1*abs(difference))

            return dist / len(units_in_doc)  # normalize by the number of units

        elif handler == "<<":
            amount_bigger = amount[0]
            amount_smaller = amount[1]
            dist = 0
            for number in units_in_doc:
                amount_avg = (amount_smaller + amount_bigger) / 2.0
                dist = dist + math.exp(-1*abs(number-amount_avg))
            return dist / len(units_in_doc)  # normalize by the number of units

    def cache(self):
        print("Caching the pre-computed values. ")
        Path(os.path.dirname(self.cache_path)).mkdir(parents=True, exist_ok=True)

        pickle.dump(
            (self.corpus_size, self.avgdl, self.doc_freqs, self.idf, self.average_idf, self.doc_len, self.map_unit_num,self.docs,self.index_creation_time),
            open(self.cache_path, "wb"))

    def load_cache(self):
        print("Loading from cache. ")
        self.corpus_size, self.avgdl, self.doc_freqs, self.idf, self.average_idf, self.doc_len, self.map_unit_num,self.docs,self.index_creation_time = pickle.load(
            open(self.cache_path, "rb"))


