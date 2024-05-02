import math
from numba.typed import List
import numba
import torch
import array
import json
import pickle
from collections import defaultdict
import time

import h5py
import numpy as np
from tqdm.auto import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class L0:
    """non-differentiable
    """

    def __call__(self, batch_rep):
        return torch.count_nonzero(batch_rep, dim=-1).float().mean()

def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def to_list(tensor):
    return tensor.detach().cpu().tolist()
def restore_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)
    # strict = False => it means that we just load the parameters of layers which are present in both and
    # ignores the rest
    if len(missing_keys) > 0:
        print("~~ [WARNING] MISSING KEYS WHILE RESTORING THE MODEL ~~")
        print(missing_keys)
    if len(unexpected_keys) > 0:
        print("~~ [WARNING] UNEXPECTED KEYS WHILE RESTORING THE MODEL ~~")
        print(unexpected_keys)
    print("restoring model:", model.__class__.__name__)

class IndexDictOfArray:
    def __init__(self, index_path=None, force_new=False, filename="array_index.h5py", dim_voc=None):
        if index_path is not None:
            self.index_path = index_path
            if not os.path.exists(index_path):
                os.makedirs(index_path)
            self.filename = os.path.join(self.index_path, filename)
            if os.path.exists(self.filename) and not force_new:
                print("index already exists, loading...")
                self.file = h5py.File(self.filename, "r")
                if dim_voc is not None:
                    dim = dim_voc
                else:
                    dim = self.file["dim"][()]
                self.index_doc_id = dict()
                self.index_doc_value = dict()
                for key in tqdm(range(dim)):
                    try:
                        self.index_doc_id[key] = np.array(self.file["index_doc_id_{}".format(key)],
                                                          dtype=np.int32)
                        # ideally we would not convert to np.array() but we cannot give pool an object with hdf5
                        self.index_doc_value[key] = np.array(self.file["index_doc_value_{}".format(key)],
                                                             dtype=np.float32)
                    except:
                        self.index_doc_id[key] = np.array([], dtype=np.int32)
                        self.index_doc_value[key] = np.array([], dtype=np.float32)
                self.file.close()
                del self.file
                print("done loading index...")
                doc_ids = pickle.load(open(os.path.join(self.index_path, "doc_ids.pkl"), "rb"))
                self.n = len(doc_ids)
            else:
                self.n = 0
                print("initializing new index...")
                self.index_doc_id = defaultdict(lambda: array.array("I"))
                self.index_doc_value = defaultdict(lambda: array.array("f"))
                self.index_doc_quants = defaultdict(lambda: array.array("f"))
        else:
            self.n = 0
            print("initializing new index...")
            self.index_doc_id = defaultdict(lambda: array.array("I"))
            self.index_doc_value = defaultdict(lambda: array.array("f"))

    def add_batch_document(self, row, col, data, n_docs=-1):
        """add a batch of documents to the index
        """
        if n_docs < 0:
            self.n += len(set(row))
        else:
            self.n += n_docs
        for doc_id, dim_id, value in zip(row, col, data):
            self.index_doc_id[dim_id].append(doc_id)# from term to lis of documents
            self.index_doc_value[dim_id].append(value)# from term to list of values

    def __len__(self):
        return len(self.index_doc_id)

    def nb_docs(self):
        return self.n

    def save(self, dim=None):
        print("converting to numpy")
        for key in tqdm(list(self.index_doc_id.keys())):
            self.index_doc_id[key] = np.array(self.index_doc_id[key], dtype=np.int32)
            self.index_doc_value[key] = np.array(self.index_doc_value[key], dtype=np.float32)
        print("save to disk")
        with h5py.File(self.filename, "w") as f:
            if dim:
                f.create_dataset("dim", data=int(dim))
            else:
                f.create_dataset("dim", data=len(self.index_doc_id.keys()))
            for key in tqdm(self.index_doc_id.keys()):
                f.create_dataset("index_doc_id_{}".format(key), data=self.index_doc_id[key])
                f.create_dataset("index_doc_value_{}".format(key), data=self.index_doc_value[key])
            f.close()
        print("saving index distribution...")  # => size of each posting list in a dict
        index_dist = {}
        for k, v in self.index_doc_id.items():
            index_dist[int(k)] = len(v)
        json.dump(index_dist, open(os.path.join(self.index_path, "index_dist.json"), "w"))


class IndexDictOfNumericalArray:
    def __init__(self, index_path=None, force_new=False, filename="array_index.h5py", dim_voc=None):
        if index_path is not None:
            self.index_path = index_path
            self.filename_unit = os.path.join(self.index_path, "units.pickle")
            if not os.path.exists(index_path):
                os.makedirs(index_path)
            self.filename = os.path.join(self.index_path, filename)
            if os.path.exists(self.filename) and not force_new:
                print("index already exists, loading...")
                self.file = h5py.File(self.filename, "r")
                if dim_voc is not None:
                    dim = dim_voc
                else:
                    dim = self.file["dim"][()]
                self.index_doc_id = dict()
                self.index_doc_value = dict()
                self.index_unit_docs = dict()
                self.index_unit_value = dict()

                for key in tqdm(range(dim)):
                    try:
                        self.index_doc_id[key] = np.array(self.file["index_doc_id_{}".format(key)],
                                                          dtype=np.int32)
                        # ideally we would not convert to np.array() but we cannot give pool an object with hdf5
                        self.index_doc_value[key] = np.array(self.file["index_doc_value_{}".format(key)],
                                                             dtype=np.float32)
                    except:
                        self.index_doc_id[key] = np.array([], dtype=np.int32)
                        self.index_doc_value[key] = np.array([], dtype=np.float32)

                units = pickle.load(open(self.filename_unit, "rb"))
                for key in tqdm(units):
                    try:
                        self.index_unit_docs[key] = np.array(self.file["index_unit_docs_{}".format(key)],
                                                             dtype=np.int32)
                        # ideally we would not convert to np.array() but we cannot give pool an object with hdf5
                        self.index_unit_value[key] = np.array(self.file["index_unit_value_{}".format(key)],
                                                              dtype=np.float32)
                    except:

                        self.index_unit_docs[key] = np.array([], dtype=np.int32)
                        self.index_unit_value[key] = np.array([], dtype=np.float32)

                self.file.close()
                del self.file
                print("done loading index...")
                doc_ids = pickle.load(open(os.path.join(self.index_path, "doc_ids.pkl"), "rb"))
                self.n = len(doc_ids)
            else:
                self.n = 0
                print("initializing new index...")
                self.index_doc_id = defaultdict(lambda: array.array("I"))
                self.index_doc_value = defaultdict(lambda: array.array("f"))
                self.index_unit_docs =defaultdict(lambda: array.array("I"))
                self.index_unit_value = defaultdict(lambda: array.array("f"))

        else:
            self.n = 0
            print("initializing new index...")
            self.index_doc_id = defaultdict(lambda: array.array("I"))
            self.index_doc_value = defaultdict(lambda: array.array("f"))
            self.index_unit_docs =defaultdict(lambda: array.array("I"))
            self.index_unit_value = defaultdict(lambda: array.array("f"))


    def add_batch_document(self, row, col, data, quants, n_docs=-1):
        """add a batch of documents to the index
        """
        if n_docs < 0:
            self.n += len(set(row))
        else:
            self.n += n_docs

        for doc_id, dim_id, value in zip(row, col, data):
            self.index_doc_id[dim_id].append(doc_id)# from term to lis of documents
            self.index_doc_value[dim_id].append(value)# from term to list of values
        for doc_id,quant in zip(np.unique(row),quants):
            for unit,list_of_values in quant.items():
                for l in list_of_values:
                    self.index_unit_docs[unit].append(doc_id)# if there are more than one we add them in repeation
                    self.index_unit_value[unit].append(l)# a parallel list for values exists



    def __len__(self):
        return len(self.index_doc_id)

    def nb_docs(self):
        return self.n

    def save(self, dim=None):
        print("converting to numpy")
        for key in tqdm(list(self.index_doc_id.keys())):
            self.index_doc_id[key] = np.array(self.index_doc_id[key], dtype=np.int32)
            self.index_doc_value[key] = np.array(self.index_doc_value[key], dtype=np.float32)
        for key in tqdm(list(self.index_unit_docs.keys())):
            self.index_unit_docs[key] = np.array(self.index_unit_docs[key], dtype=np.int32)
            self.index_unit_value[key] = np.array(self.index_unit_value[key], dtype=np.float32)
        print("save to disk")
        with h5py.File(self.filename, "w") as f:
            if dim:
                f.create_dataset("dim", data=int(dim))
            else:
                f.create_dataset("dim", data=len(self.index_doc_id.keys()))
            for key in tqdm(self.index_doc_id.keys()):
                f.create_dataset("index_doc_id_{}".format(key), data=self.index_doc_id[key])
                f.create_dataset("index_doc_value_{}".format(key), data=self.index_doc_value[key])
            for key in tqdm(self.index_unit_docs.keys()):
                try:
                    f.create_dataset("index_unit_docs_{}".format(key), data=self.index_unit_docs[key])
                    f.create_dataset("index_unit_value_{}".format(key), data=self.index_unit_value[key])
                except Exception as e:
                    print(key.replace(" ","_"))
                    print(self.index_unit_docs[key])
                    print(self.index_unit_value[key])
                    print(e)

            f.close()
        file = open(self.filename_unit, 'wb')

        pickle.dump(list(self.index_unit_docs.keys()), file)
        file.close()
        print("saving index distribution...")  # => size of each posting list in a dict
        index_dist = {}
        for k, v in self.index_doc_id.items():
            index_dist[int(k)] = len(v)
        json.dump(index_dist, open(os.path.join(self.index_path, "index_dist.json"), "w"))

class Evaluator:
    def __init__(self, model, config=None, restore=True):
        """base class for model evaluation (inference)
        """
        self.model = model
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if restore:
            if self.device == torch.device("cuda"):
                if "pretrained_no_yamlconfig" not in config or not config["pretrained_no_yamlconfig"]:
                    checkpoint = torch.load(os.path.join(config["checkpoint_dir"], "model.tar"))
                    restore_model(model, checkpoint["model_state_dict"])
                    print(
                        "restore model on GPU at {}".format(os.path.join(config["checkpoint_dir"], "model.tar")))
                self.model.eval()
                if torch.cuda.device_count() > 1:
                    print(" --- use {} GPUs --- ".format(torch.cuda.device_count()))
                    self.model = torch.nn.DataParallel(self.model)
                self.model.to(self.device)

            else:  # CPU
                if "pretrained_no_yamlconfig" not in config or not config["pretrained_no_yamlconfig"]:
                    checkpoint = torch.load(os.path.join(config["checkpoint_dir"], "model.tar"),
                                            map_location=self.device)
                    restore_model(model, checkpoint["model_state_dict"])
                    print(
                        "restore model on CPU at {}".format(os.path.join(config["checkpoint_dir"], "model.tar")))
        else:
            print("WARNING: init evaluator, NOT restoring the model, NOT placing on device")
        self.model.eval()  # => put in eval mode


class SparseIndexing(Evaluator):
    """sparse indexing
    """

    def __init__(self, model, config, compute_stats=False, dim_voc=None, is_query=False, force_new=True, **kwargs):
        super().__init__(model, config, **kwargs)
        self.index_dir = config["index_dir"] if config is not None else None
        self.sparse_index = IndexDictOfArray(self.index_dir, dim_voc=dim_voc, force_new=force_new)
        self.compute_stats = compute_stats
        self.is_query = is_query
        if self.compute_stats:
            self.l0 = L0()

    def index(self, collection_loader, id_dict=None):
        doc_ids = []
        if self.compute_stats:
            stats = defaultdict(float)
        count = 0
        with torch.no_grad():
            for t, batch in enumerate(tqdm(collection_loader)):

                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"id"}}
                if self.is_query:
                    batch_documents = self.model(q_kwargs=inputs)["q_rep"]
                else:
                    batch_documents = self.model(d_kwargs=inputs)["d_rep"]
                if self.compute_stats:
                    stats["L0_d"] += self.l0(batch_documents).item()
                row, col = torch.nonzero(batch_documents,
                                         as_tuple=True)  # row and col are the indicies of non-zero values in the batch

                data = batch_documents[row, col]
                row = row + count
                batch_ids = to_list(batch["id"])
                if id_dict:
                    batch_ids = [id_dict[x] for x in batch_ids]
                count += len(batch_ids)
                doc_ids.extend(batch_ids)
                self.sparse_index.add_batch_document(row.cpu().numpy(), col.cpu().numpy(), data.cpu().numpy(),
                                                     n_docs=len(batch_ids))
        if self.compute_stats:
            stats = {key: value / len(collection_loader) for key, value in stats.items()}
        if self.index_dir is not None:
            self.sparse_index.save()
            pickle.dump(doc_ids, open(os.path.join(self.index_dir, "doc_ids.pkl"), "wb"))
            print("done iterating over the corpus...")
            print("index contains {} posting lists".format(len(self.sparse_index)))
            print("index contains {} documents".format(len(doc_ids)))
            if self.compute_stats:
                with open(os.path.join(self.index_dir, "index_stats.json"), "w") as handler:
                    json.dump(stats, handler)
        else:
            # if no index_dir, we do not write the index to disk but return it
            for key in list(self.sparse_index.index_doc_id.keys()):
                # convert to numpy
                self.sparse_index.index_doc_id[key] = np.array(self.sparse_index.index_doc_id[key], dtype=np.int32)
                self.sparse_index.index_doc_value[key] = np.array(self.sparse_index.index_doc_value[key],
                                                                  dtype=np.float32)
            out = {"index": self.sparse_index, "ids_mapping": doc_ids}
            if self.compute_stats:
                out["stats"] = stats
            return out


class SparseNumericalIndexing(Evaluator):
    """sparse indexing
    """

    def __init__(self, model, config, compute_stats=False, dim_voc=None, is_query=False, force_new=True, **kwargs):
        super().__init__(model, config, **kwargs)
        self.index_dir = config["index_dir"] if config is not None else None
        self.sparse_index = IndexDictOfNumericalArray(self.index_dir, dim_voc=dim_voc, force_new=force_new)
        self.compute_stats = compute_stats
        self.is_query = is_query
        if self.compute_stats:
            self.l0 = L0()

    def index(self, collection_loader, id_dict=None):
        doc_ids = []
        if self.compute_stats:
            stats = defaultdict(float)
        count = 0
        with torch.no_grad():
            for t, batch in enumerate(tqdm(collection_loader)):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"id", "quants"}}
                if self.is_query:
                    batch_documents = self.model(q_kwargs=inputs)["q_rep"]
                else:
                    batch_documents = self.model(d_kwargs=inputs)["d_rep"]
                if self.compute_stats:
                    stats["L0_d"] += self.l0(batch_documents).item()
                row, col = torch.nonzero(batch_documents,
                                         as_tuple=True)  # row and col are the indicies of non-zero values in the batch

                data = batch_documents[row, col]
                row = row + count
                batch_ids = to_list(batch["id"])
                if id_dict:
                    batch_ids = [id_dict[x] for x in batch_ids]
                count += len(batch_ids)
                doc_ids.extend(batch_ids)
                self.sparse_index.add_batch_document(row.cpu().numpy(), col.cpu().numpy(), data.cpu().numpy(),
                                                     batch["quants"],
                                                     n_docs=len(batch_ids))
        if self.compute_stats:
            stats = {key: value / len(collection_loader) for key, value in stats.items()}
        if self.index_dir is not None:
            self.sparse_index.save()
            pickle.dump(doc_ids, open(os.path.join(self.index_dir, "doc_ids.pkl"), "wb"))
            print("done iterating over the corpus...")
            print("index contains {} posting lists".format(len(self.sparse_index)))
            print("index contains {} documents".format(len(doc_ids)))
            if self.compute_stats:
                with open(os.path.join(self.index_dir, "index_stats.json"), "w") as handler:
                    json.dump(stats, handler)
        else:
            # if no index_dir, we do not write the index to disk but return it
            for key in list(self.sparse_index.index_doc_id.keys()):
                # convert to numpy
                self.sparse_index.index_doc_id[key] = np.array(self.sparse_index.index_doc_id[key], dtype=np.int32)
                self.sparse_index.index_doc_value[key] = np.array(self.sparse_index.index_doc_value[key],
                                                                  dtype=np.float32)
            out = {"index": self.sparse_index, "ids_mapping": doc_ids}
            if self.compute_stats:
                out["stats"] = stats
            return out


class SparseRetrieval(Evaluator):
    """retrieval from SparseIndexing
    """


    @staticmethod
    def select_topk(filtered_indexes, scores, k):
        if len(filtered_indexes) > k:
            sorted_ = np.argpartition(scores, k)[:k]  # maybe negative scores because of this?
            filtered_indexes, scores = filtered_indexes[sorted_], -scores[sorted_]
        else:
            scores = -scores
        return filtered_indexes, scores

    @staticmethod
    @numba.njit(nogil=True, parallel=False, cache=True)
    def numba_score_float(inverted_index_ids: numba.typed.Dict,  # self.numba_index_doc_ids
                          inverted_index_floats: numba.typed.Dict,  # self.numba_index_doc_values
                          indexes_to_retrieve: np.ndarray,
                          query_values: np.ndarray,
                          threshold: float,
                          size_collection: int):
        scores = np.zeros(size_collection, dtype=np.float32)  # initialize array with size = size of collection
        n = len(indexes_to_retrieve)

        for _idx in range(n):
            local_idx = indexes_to_retrieve[_idx]  # which posting list to search
            query_float = query_values[_idx]  # what is the value of the query for this posting list
            retrieved_indexes = inverted_index_ids[local_idx]  # get indexes from posting list ( get doc ids form term)
            retrieved_floats = inverted_index_floats[local_idx]  # get values from posting list (get values from term)
            for j in numba.prange(len(retrieved_indexes)):  # for each document that has that term
                scores[retrieved_indexes[j]] += query_float * retrieved_floats[j]
        filtered_indexes = np.argwhere(scores > threshold)[:, 0]  # ideally we should have a threshold to filter
        # unused documents => this should be tuned, currently it is set to 0
        return filtered_indexes, -scores[filtered_indexes]

    def __init__(self, model, config, dim_voc, dataset_name=None, index_d=None, compute_stats=False, is_beir=False,
                 **kwargs):
        super().__init__(model, config, **kwargs)
        assert ("index_dir" in config and index_d is None) or (
                "index_dir" not in config and index_d is not None)
        if "index_dir" in config:
            self.sparse_index = IndexDictOfArray(config["index_dir"], dim_voc=dim_voc)
            self.doc_ids = pickle.load(open(os.path.join(config["index_dir"], "doc_ids.pkl"), "rb"))
        else:
            self.sparse_index = index_d["index"]
            self.doc_ids = index_d["ids_mapping"]
            for i in range(dim_voc):
                # missing keys (== posting lists), causing issues for retrieval => fill with empty
                if i not in self.sparse_index.index_doc_id:
                    self.sparse_index.index_doc_id[i] = np.array([], dtype=np.int32)
                    self.sparse_index.index_doc_value[i] = np.array([], dtype=np.float32)
        # convert to numba
        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()
        for key, value in self.sparse_index.index_doc_id.items():
            self.numba_index_doc_ids[key] = value
        for key, value in self.sparse_index.index_doc_value.items():
            self.numba_index_doc_values[key] = value
        self.out_dir = os.path.join(config["out_dir"], dataset_name) if (dataset_name is not None and not is_beir) \
            else config["out_dir"]
        self.doc_stats = index_d["stats"] if (index_d is not None and compute_stats) else None
        self.compute_stats = compute_stats
        if self.compute_stats:
            self.l0 = L0()


    def retrieve_single(self, single_query, top_k, id_dict=False, threshold=0):
        with torch.no_grad():
            q_id = to_list(single_query["id"])[0]
            if id_dict:
                q_id = id_dict[q_id]
            inputs = {k: v for k, v in single_query.items() if k not in {"id", "quants"}}
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)

            query = self.model(q_kwargs=inputs)["q_rep"]  # we assume ONE query per batch here

            # TODO: batched version for retrieval

            row, col = torch.nonzero(query, as_tuple=True)  # there is a single query, so row is all zero
            values = query[to_list(row), to_list(col)]
            filtered_indexes, scores = self.numba_score_float(inverted_index_ids=self.numba_index_doc_ids,
                                                              inverted_index_floats=self.numba_index_doc_values,
                                                              indexes_to_retrieve=col.cpu().numpy(),
                                                              query_values=values.cpu().numpy().astype(np.float32),
                                                              threshold=threshold,
                                                              size_collection=self.sparse_index.nb_docs())
            # threshold set to 0 by default, could be better

            filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=top_k)
            score_dict={}
            for index,score in zip(filtered_indexes, scores):
                score_dict[index]=score
            score_dict={k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1],reverse=True)}


        out = {"retrieval": score_dict}
        return out


class SparseNumericalRetrieval(Evaluator):
    """retrieval from SparseIndexing
    """

    @staticmethod
    def select_topk(filtered_indexes, scores, k):
        if len(filtered_indexes) > k:
            sorted_ = np.argpartition(scores, k)[:k]  # maybe negative scores because of this?
            filtered_indexes, scores = filtered_indexes[sorted_], -scores[sorted_]
        else:
            scores = -scores
        return filtered_indexes, scores

    @staticmethod
    @numba.njit(cache=True,nogil=True,parallel=True)
    def numba_score_float(inverted_index_ids: numba.typed.Dict,  # self.numba_index_doc_ids
                          inverted_index_floats: numba.typed.Dict,  # self.numba_index_doc_values
                          inverted_index_unit_ids: numba.typed.Dict,
                          inverted_index_unit_floats: numba.typed.Dict,
                          indexes_to_retrieve: np.ndarray,
                          query_values: np.ndarray,
                          query_unit: str,
                          query_quantity_value:float,
                          query_handler: str,
                          threshold: float,
                          size_collection: int,
                          query_quantity_value2=-1,
                          numerical_weight=1):

        scores = np.zeros(size_collection, dtype=np.float32)  # initialize array with size = size of collection
        numerical_scores= np.zeros(size_collection, dtype=np.float32)
        doc_with_unit = inverted_index_unit_ids[query_unit]
        doc_values = inverted_index_unit_floats[query_unit]

        n = len(indexes_to_retrieve)# numba has problem with python list and cheking if an element is present in it, it will through segfault with parallell true
        for _idx in range(n):
            local_idx = indexes_to_retrieve[_idx]  # which posting list to search
            query_float = query_values[_idx]  # what is the value of the query for this posting list
            retrieved_indexes = inverted_index_ids[local_idx]  # get indexes from posting list ( get doc ids form term)
            retrieved_floats = inverted_index_floats[local_idx]  # get values from posting list (get values from term)
            for j in numba.prange(len(retrieved_indexes)):  # for each document that has that term
                doc_id=retrieved_indexes[j]
                scores[doc_id]+= query_float * retrieved_floats[j]
        scores = (scores - np.min(scores))/(np.max(scores)-np.min(scores))#normalize the scores
        document_above_threshold=np.where(scores>np.quantile(scores, .25))[0]

        for j in numba.prange(len(document_above_threshold)):  # for each document that has that term
            doc_id=document_above_threshold[j]
            docs_with_id=np.where(doc_with_unit== doc_id)[0]
            if len(docs_with_id)>0:
                doc_values_with_id=doc_values[docs_with_id]
                for val_doc in numba.prange(len(doc_values_with_id)):
                    doc_value=doc_values_with_id[val_doc]
                    numerical_distance=0
                    difference = doc_value - query_quantity_value
                    if query_handler == "=":#equal
                        numerical_distance = numerical_distance+ math.exp(-1*abs(doc_value-query_quantity_value))
                    elif query_handler == ">" and difference >0:#bigger than
                        numerical_distance = numerical_distance + (query_quantity_value / doc_value)
                    elif query_handler == "<" and difference <0:#smaller than
                        numerical_distance = numerical_distance + (doc_value/query_quantity_value)
                    elif query_handler == "<<" and query_quantity_value2!=-1:#ranges
                        amount_avg = (query_quantity_value + query_quantity_value2) / 2.0
                        numerical_distance = numerical_distance + math.exp(-1*abs(doc_value-amount_avg))
                    numerical_scores[doc_id] += (numerical_distance)/len(doc_values_with_id)


        numerical_scores=(numerical_scores*numerical_weight)
        scores=scores+numerical_scores



        filtered_indexes = np.argwhere(scores > threshold)[:, 0]  # ideally we should have a threshold to filter
        # unused documents => this should be tuned, currently it is set to 0
        return filtered_indexes, -scores[filtered_indexes]




    def __init__(self, model, config, dim_voc, dataset_name=None, index_d=None, compute_stats=False, is_beir=False,
                 **kwargs):
        super().__init__(model, config, **kwargs)
        assert ("index_dir" in config and index_d is None) or (
                "index_dir" not in config and index_d is not None)
        if "index_dir" in config:
            self.sparse_index = IndexDictOfNumericalArray(config["index_dir"], dim_voc=dim_voc)
            self.doc_ids = pickle.load(open(os.path.join(config["index_dir"], "doc_ids.pkl"), "rb"))
        else:
            self.sparse_index = index_d["index"]
            self.doc_ids = index_d["ids_mapping"]
            for i in range(dim_voc):
                # missing keys (== posting lists), causing issues for retrieval => fill with empty
                if i not in self.sparse_index.index_doc_id:
                    self.sparse_index.index_doc_id[i] = np.array([], dtype=np.int32)
                    self.sparse_index.index_doc_value[i] = np.array([], dtype=np.float32)
        # convert to numba
        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()
        self.numba_index_unit_docs = numba.typed.Dict()
        self.numba_index_unit_values = numba.typed.Dict()
        for key, value in self.sparse_index.index_doc_id.items():
            self.numba_index_doc_ids[key] = value
        for key, value in self.sparse_index.index_doc_value.items():
            self.numba_index_doc_values[key] = value
            #########
        for key, value in self.sparse_index.index_unit_docs.items():# unit -> doc_ids that have that unit
            self.numba_index_unit_docs[key] = value
        for key, value in self.sparse_index.index_unit_value.items():# unit -> values parallel to the doc ids
            self.numba_index_unit_values[key] = value

        self.out_dir = os.path.join(config["out_dir"], dataset_name) if (dataset_name is not None and not is_beir) \
            else config["out_dir"]
        self.doc_stats = index_d["stats"] if (index_d is not None and compute_stats) else None
        self.compute_stats = compute_stats
        if self.compute_stats:
            self.l0 = L0()

    def retrieve(self, q_loader, top_k, name=None, return_d=False, id_dict=False, threshold=0):
        makedir(self.out_dir)
        if self.compute_stats:
            makedir(os.path.join(self.out_dir, "stats"))
        res = defaultdict(dict)
        if self.compute_stats:
            stats = defaultdict(float)
        with torch.no_grad():
            for t, batch in enumerate(tqdm(q_loader)):
                q_id = to_list(batch["id"])[0]
                if id_dict:
                    q_id = id_dict[q_id]
                inputs = {k: v for k, v in batch.items() if k not in {"id"}}
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                query = self.model(q_kwargs=inputs)["q_rep"]  # we assume ONE query per batch here
                if self.compute_stats:
                    stats["L0_q"] += self.l0(query).item()
                # TODO: batched version for retrieval
                row, col = torch.nonzero(query, as_tuple=True)  # there is a single query, so row is all zero
                values = query[to_list(row), to_list(col)]
                filtered_indexes, scores = self.numba_score_float(self.numba_index_doc_ids,
                                                                  self.numba_index_doc_values,
                                                                  col.cpu().numpy(),
                                                                  values.cpu().numpy().astype(np.float32),
                                                                  threshold=threshold,
                                                                  size_collection=self.sparse_index.nb_docs())
                # threshold set to 0 by default, could be better

                filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=top_k)
                for id_, sc in zip(filtered_indexes, scores):
                    res[str(q_id)][str(self.doc_ids[id_])] = float(sc)
        if self.compute_stats:
            stats = {key: value / len(q_loader) for key, value in stats.items()}
        if self.compute_stats:
            with open(os.path.join(self.out_dir, "stats",
                                   "q_stats{}.json".format("_iter_{}".format(name) if name is not None else "")),
                      "w") as handler:
                json.dump(stats, handler)
            if self.doc_stats is not None:
                with open(os.path.join(self.out_dir, "stats",
                                       "d_stats{}.json".format("_iter_{}".format(name) if name is not None else "")),
                          "w") as handler:
                    json.dump(self.doc_stats, handler)
        with open(os.path.join(self.out_dir, "run{}.json".format("_iter_{}".format(name) if name is not None else "")),
                  "w") as handler:
            json.dump(res, handler)
        if return_d:
            out = {"retrieval": res}
            if self.compute_stats:
                out["stats"] = stats if self.doc_stats is None else {**stats, **self.doc_stats}
            return out

    def retrieve_single(self, single_query, top_k, id_dict=False, threshold=0):
        with torch.no_grad():
            q_id = to_list(single_query["id"])[0]
            if id_dict:
                q_id = id_dict[q_id]
            inputs = {k: v for k, v in single_query.items() if k not in {"id", "quants"}}
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)

            query = self.model(q_kwargs=inputs)["q_rep"]  # we assume ONE query per batch here

            # TODO: batched version for retrieval
            row, col = torch.nonzero(query, as_tuple=True)  # there is a single query, so row is all zero
            values = query[to_list(row), to_list(col)]
            unit = single_query["quants"]["unit"]
            value = single_query["quants"]["value"]
            handler=single_query["quants"]["handler"]
            filtered_indexes, scores = self.numba_score_float(inverted_index_ids=self.numba_index_doc_ids,# term -> doc_ids
                                                              inverted_index_floats=self.numba_index_doc_values,# term -> weights in each docs
                                                              inverted_index_unit_ids=self.numba_index_unit_docs,# unit -> list of doc ids
                                                              inverted_index_unit_floats=self.numba_index_unit_values,# unit -> list of values in docs
                                                              indexes_to_retrieve=col.cpu().numpy(),
                                                              query_values=values.cpu().numpy().astype(np.float32),
                                                              query_unit=unit,
                                                              query_quantity_value=value,
                                                              query_handler=handler,
                                                              threshold=threshold,
                                                              size_collection=self.sparse_index.nb_docs())
            # threshold set to 0 by default, could be better
            filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=top_k)
            score_dict={}
            for index,score in zip(filtered_indexes, scores):
                score_dict[index]=score
            score_dict={k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1],reverse=True)}


        out = {"retrieval": score_dict}
        return out


