#!/usr/bin/env python
import pickle

from transformers import AutoTokenizer
import torch

from dataset.splade_dataloaders import CollectionDataLoader
from dataset.splade_datasets import CollectionDatasetPreLoad
from models.semantic_matchers.SPLADE.SPLADE_indexer import SparseIndexing, SparseRetrieval
from evaluate.utils import utils
import time
from models.semantic_matchers.SPLADE.splade import Splade,SpladeDoc
"""
Implementation of the SPALDE ranking 
"""


class SPLADE_Ranker:
    def __init__(self, corpus, model_name='naver/splade-cocondenser-ensembledistil',
                 load_cache=False, cache_path="./models_weights/SPLADE", splade_type="splade",tokenizer_type='distilbert-base-uncased',batch_size=256,sent_col="sentence"):
        '''

        :param model_name: the name of the pre-trained splade model
        :param corpus: the corpus of sentences
        :param load_cache: if set to true it will load the pre-computed embedding from file
        :param cache_path : path to the embedding cache
        :param splade_type: which variant, splad or the splade_doc
        :param tokenizer_type: name of the tokenizer
        :param: batch_size: batch size for indexing
        :normalize_sentences: whether to use normalized sentences
        '''
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_map = {
            "splade": Splade,
            "splade_doc": SpladeDoc
        }

        model_class = model_map[splade_type]
        model_config= {'model_type_or_dir': 'naver/splade-cocondenser-ensembledistil', 'model_type_or_dir_q': None, 'freeze_d_model': 0, 'agg': 'max', 'fp16': False}
        self.model = model_class(**model_config)
        if not model_name.startswith("naver"):# it is a pretrained checkpoint you need to load the weights manually
            stat_dict=torch.load(model_name)
            new_dict={}
            for key,val in stat_dict.items():
                new_dict[key.replace("module.","")]=val
            del new_dict["transformer_rep.transformer.bert.embeddings.position_ids"]
            self.model.load_state_dict(new_dict)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        self.sent_col=sent_col
        d_collection = CollectionDatasetPreLoad(data_dir=corpus,sent_col=self.sent_col)
        self.tokenizer_type=tokenizer_type
        self.batch_size=batch_size
        self.id_to_doc=d_collection.data_dict
        self.corpus_path=corpus
        self.cache_path = cache_path
        self.evaluator=None
        self.index_creation_time=None
        self.model_name=model_name

        if not load_cache:
            self._initialize(d_collection)
        else:
            self.index_creation_time = pickle.load(open(self.cache_path+"_splademeta", "rb"))
        config={"index_dir":cache_path,"checkpoint_dir":"???","pretrained_no_yamlconfig":True,"out_dir":cache_path+"_index"}
        self.evaluator = SparseRetrieval(config=config, model=self.model, dataset_name="noname",
                                         compute_stats=True, dim_voc=self.model.output_dim)


    def _initialize(self,d_collection):
        print("creating index...")
        start_time = time.time()
        d_loader = CollectionDataLoader(dataset=d_collection, tokenizer_type=self.tokenizer_type,
                                        max_length=512,
                                        batch_size=self.batch_size,
                                        shuffle=False, num_workers=10, prefetch_factor=4)

        config={"index_dir":self.cache_path,"checkpoint_dir":"???","pretrained_no_yamlconfig":True,"out_dir":self.cache_path+"_index"}
        evaluator = SparseIndexing(model=self.model, config=config, compute_stats=True)
        evaluator.index(d_loader)
        end_time = time.time()
        print("creating the SPLADE features took :", utils.secondsToText(end_time - start_time))
        self.index_creation_time = utils.secondsToText(end_time - start_time)#creating the SPLADE features took : 34.0 minutes, 42.93047 seconds
        pickle.dump(self.index_creation_time,open(self.cache_path+"_splademeta", "wb"))


    def get_top_n(self, query, n=5,  overload=False):
        """
        Return the top n based on the cosine similiarity of the query with the documents.
        :param query: the keywords in the query
        :param n: the number of top results to return
        :param handler: the handler : equal, greater than, smaller than or a range (ignored)
        :param amount: the exact amount or a list of two numbers for a range (ignored)
        :param unit: the unit to look for (ignored)
        :return: the top n results
        """

        # print("query for splade is :", query_text)
        processed_query = self.tokenizer(query,
                                      add_special_tokens=True,
                                      padding="longest",  # pad to max sequence length in batch
                                      truncation="longest_first",  # truncates to self.max_length
                                      max_length=128,
                                      return_attention_mask=True)
        processed_query["input_ids"]=[processed_query["input_ids"]]
        processed_query["attention_mask"]=[processed_query["attention_mask"]]
        input = {**{k: torch.tensor(v) for k, v in processed_query.items()},
                 "id": torch.tensor([0], dtype=torch.long)}
        result = self.evaluator.retrieve_single(input, top_k=n, threshold=0)
        overloaded_results=[]
        sentences = list(result["retrieval"].keys())

        if overload:
            for id,score in result["retrieval"].items():
                overloaded_results.append({"doc_id":id,"doc":self.id_to_doc[id],"score":score})

            return overloaded_results
        selected_docs=[self.id_to_doc[s] for s in sentences]
        return selected_docs