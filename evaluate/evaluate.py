"""
Run evaluation for variety of models and save the result
"""
import os
import time
from argparse import ArgumentParser

from nltk.stem import PorterStemmer
from collections import Counter
import json
import pytrec_eval

from models.lexical_matchers.BM25_base import BM25Okapi
from models.lexical_matchers.bm25_filter import FilterBM25
from models.lexical_matchers.qbm25 import QBM25
from models.semantic_matchers.ColBERT_ranker import ColBERT_Ranker
from models.semantic_matchers.QColBERT_ranker import QColBERT_Ranker
from models.semantic_matchers.QSPLADE_ranker import QSPLADE_Ranker
from models.semantic_matchers.SPLADE_ranker import SPLADE_Ranker
from .utils import utils
from pathlib import Path
import os

stemmer = PorterStemmer()
from dataset.bm25_dataloaders import read_test_queries, process_data_bm25
from pytrec_eval import RelevanceEvaluator

def mrr_k(run, qrel, k, agg=True):
    evaluator = RelevanceEvaluator(qrel, {"recip_rank"})
    truncated = truncate_run(run, k)
    mrr = evaluator.evaluate(truncated)
    if agg:
        mrr = sum([d["recip_rank"] for d in mrr.values()]) / max(1, len(mrr))
    return mrr


def evaluate(run, qrel, metric, agg=True, select=None):
    assert metric in pytrec_eval.supported_measures, print(
        "provide valid pytrec_eval metric"
    )
    evaluator = RelevanceEvaluator(qrel, {metric})
    out_eval = evaluator.evaluate(run)
    res = Counter({})
    if agg:
        for (
                d
        ) in (
                out_eval.values()
        ):  # when there are several results provided (e.g. several cut values)
            res += Counter(d)
        res = {k: v / len(out_eval) for k, v in res.items()}
        if select is not None:
            string_dict = "{}_{}".format(metric, select)
            if string_dict in res:
                return res[string_dict]
            else:  # If the metric is not on the dict, say that it was 0
                return 0
        else:
            return res
    else:
        return out_eval


def truncate_run(run, k):
    """truncates run file to only contain top-k results for each query"""
    temp_d = {}
    for q_id in run:
        sorted_run = {
            k: v
            for k, v in sorted(
                run[q_id].items(), key=lambda item: item[1], reverse=True
            )
        }
        temp_d[q_id] = {k: sorted_run[k] for k in list(sorted_run.keys())[:k]}
    return temp_d

def init_eval(metric):
    if metric not in ["MRR@10","MRR@100","MRR@1000", "recall@10",  "recall@100", "recall@1000","ndcg_cut@10",  "ndcg_cut@100", "ndcg_cut@1000","P@10",  "P@100", "P@1000"]:
        raise NotImplementedError("provide valid metric")
    if metric == "MRR@10":
        return lambda x, y: mrr_k(x, y, k=10, agg=True)
    if metric == "MRR@100":
        return lambda x, y: mrr_k(x, y, k=100, agg=True)
    if metric == "MRR@1000":
        return lambda x, y: mrr_k(x, y, k=1000, agg=True)
    else:
        return lambda x, y: evaluate(x, y, metric=metric.split('@')[0], agg=True, select=metric.split('@')[1])

def filter_results(run, qrel):
    new_run={}
    for line in qrel:
        new_run[line]=run[line]
    return new_run


def get_args():
    args = ArgumentParser()

    args.add_argument("--splade-ft-path", type=str, default="./data/model_weights/finance/splade_ft/model_best.pt",
                      help="path to fine-tuned splade model for SPLADE_ft.")
    args.add_argument("--colbert-ft-path", type=str,default="./data/model_weights/finance/colbert_ft",
                      help="path to fine-tuned colbert model for COLBERT_ft. ")
    args.add_argument("--splade-pretrained", type=str, default='naver/splade-cocondenser-ensembledistil',
                      help="path/or model name to the pretrained splade model (on general data) for QSPLADE model and SPLADE baseline.")
    args.add_argument("--colbert-pretrained", type=str, default='./data/model_weights/colbertv2.0',
                      help="path to the pretrained colbert model (on general data) for QCOLBERT model and COLBERT baseline.")
    args.add_argument("--gt-folder", type=str,
                      default="./data/finance/",
                      help="folder where the gt qrels are. ")
    args.add_argument("--measures", type=str, default="MRR@10,MRR@100,MRR@1000,recall@10,P@10,P@100,recall@100,recall@1000,ndcg_cut@10,ndcg_cut@100,ndcg_cut@1000",
                      help="the measures to test.  ")
    args.add_argument("--corpus-path", type=str,
                      default="./data/finance/collection.csv",
                      help="the path to the sentence collection.")
    args.add_argument("--queries-path", type=str,
                  default="./data/finance/queries.tsv",
                  help="path to the queries.  ")
    args.add_argument("--output-folder", type=str, default="./data/finance/test_out",
                      help="The folder to save the index and output of the evaluation to.  ")
    args.add_argument("--load-cache-model", action="store_true", default=False,
                      help="Load model and index from cache. ")
    args.add_argument("--load-cache-result", action="store_true", default=False,
                      help="Compute the metrics from the results in the out-folder. ")
    args.add_argument("--model-cache-folder", type=str, default="./data/finance/model_weights/",
                      help="Where to save the model and indexes. ")
    args.add_argument("--topk", type=int, default=1000,
                      help="how many top-k to retrieve (choose based on the maximum value in recall).")
    args.add_argument("--bm25-param", type=str, default="1.5,0.75",
                  help="k1 parameter and b parameter of  bm25 seperated by comma.")
    args.add_argument("--qbm25-param", type=str, default="1.5,0.75",
                      help="k1 parameter and b parameter of  qbm25 seperated by comma.")
    args.add_argument("--bm25-filter-param", type=str, default="1.5,0.75",
                      help="k1 parameter and b parameter of  bm25 filter seperated by comma.")
    args.add_argument("--num-gpus", type=int, default=1,
                      help="number of gpus available for evaluation.")
    return args.parse_args()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    args = get_args()
    measures=args.measures.split(",")
    bm25_param={k:float(i) for k,i in zip(["k1","b"],args.bm25_param.split(","))}
    qbm25_param={k:float(i) for k,i in zip(["k1","b"],args.qbm25_param.split(","))}
    bm25_filter_param={k:float(i) for k,i in zip(["k1","b"],args.bm25_filter_param.split(","))}

    # you can comment out models from here if you want to run dedicated evaluations
    ranker_models = {
         "bm25":BM25Okapi(process_data_bm25(args.corpus_path), k1=bm25_param["k1"],b=bm25_param["b"],load_cache=args.load_cache_model,cache_path=os.path.join(args.model_cache_folder,"bm25")),
        # "qbm25": QBM25(process_data_bm25(args.corpus_path),k1=qbm25_param["k1"],b=qbm25_param["b"], load_cache=args.load_cache_model,cache_path=os.path.join(args.model_cache_folder, "qbm25")),
        # "bm25_filter":FilterBM25(process_data_bm25(args.corpus_path),k1=bm25_filter_param["k1"],b=bm25_filter_param["b"],  load_cache=args.load_cache_model,cache_path=os.path.join(args.model_cache_folder,"bm25_filter")),
        # "colbert":ColBERT_Ranker(args.corpus_path,rank=args.num_gpus,model_name=args.colbert_pretrained,load_cache=args.load_cache_model,cache_path=os.path.join(args.model_cache_folder,"ColBERT")),
        # "qcolbert":QColBERT_Ranker(args.corpus_path,rank=args.num_gpus,model_name=args.colbert_pretrained,load_cache=args.load_cache_model,cache_path=os.path.join(args.model_cache_folder,"QColBERT")),
        # "colbert_ft":ColBERT_Ranker(args.corpus_path,rank=args.num_gpus,model_name=args.colbert_ft_path,load_cache=args.load_cache_model,cache_path=os.path.join(args.model_cache_folder,"ColBERT_ft")),
        # "splade":SPLADE_Ranker( args.corpus_path,model_name=args.splade_pretrained,load_cache=args.load_cache_model,cache_path=os.path.join(args.model_cache_folder,"SPLADE")),
        # "qsplade": QSPLADE_Ranker( args.corpus_path,model_name=args.splade_pretrained,load_cache=args.load_cache_model,cache_path=os.path.join(args.model_cache_folder,"QSPLADE")),
        # "splade_ft":SPLADE_Ranker(args.corpus_path, model_name=args.splade_ft_path,load_cache=args.load_cache_model,cache_path=os.path.join(args.model_cache_folder,"SPLADE_ft")),
    }

    qrels={}
    with open(os.path.join(args.gt_folder,"qrel_compelete.json")) as reader:
        qrel_compelete=json.load(reader)
        qrels["compelete"]=qrel_compelete
    with open(os.path.join(args.gt_folder,"qrel_in_training.json")) as reader:
        qrel_in_training=json.load(reader)
        qrels["in_training"]=qrel_in_training
    with open(os.path.join(args.gt_folder,"qrel_not_training.json")) as reader:
        qrel_not_training=json.load(reader)
        qrels["not_training"]=qrel_not_training
    with open(os.path.join(args.gt_folder,"qrel_expanded.json")) as reader:
        qrel_expanded=json.load(reader)
        qrels["expanded"]=qrel_expanded
    with open(os.path.join(args.gt_folder,"qrel_wo_surface.json")) as reader:
        qrel_expanded_wo_surface=json.load(reader)
        qrels["expanded_wo_surface"]=qrel_expanded_wo_surface

    with open(os.path.join(args.gt_folder,"qrels_equal.json")) as reader:
        qrels_equal=json.load(reader)
        qrels["qrels_equal"]=qrels_equal

    with open(os.path.join(args.gt_folder,"qrels_greater.json")) as reader:
        qrels_bigger=json.load(reader)
        qrels["qrels_bigger"]=qrels_bigger

    with open(os.path.join(args.gt_folder,"qrels_less.json")) as reader:
        qrels_smaller=json.load(reader)
        qrels["qrels_smaller"]=qrels_smaller

    run_q={}
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    for ranker_name,ranker_model in ranker_models.items():
        output={}
        print("evaluating:"+ranker_name)
        if args.load_cache_result:
            with open(os.path.join(args.out_folder,ranker_name+".json"), "r") as reader:
                run_q=json.load(reader)
        else:
            start_time = time.time()
            test_qs=read_test_queries(args.queries_path)
            for i,query in enumerate(test_qs):
                input=query["query_text_proccessed"] if ranker_name.startswith("bm") or ranker_name.startswith("qbm") else query["query_text"]
                if ranker_name=="bm25_filter":
                    input=input +[str(query["value"])]+query["unit"].split(" ")
                results = ranker_model.get_top_n(input, n=args.topk, handler=query["condition"], amount=query["value"], unit=query["unit"],overload=True)
                run_q[str(i)]={}
                for res in results:
                    run_q[str(i)][str(res["doc_id"])]= float(res["score"])
            end_time = time.time()
            total_time=end_time - start_time
            output["total_time"]= utils.secondsToText(total_time)
            output["total_time_per_query"]=total_time/float(len(run_q))
            #save the results and rankings
            with open(os.path.join(args.output_folder,ranker_name+".json"), "w") as outfile:
                json.dump(run_q, outfile, indent = 4)


        for test_type, qrel in qrels.items():
            filtered_run=filter_results(run_q, qrel)
            output[test_type]={}
            for measur in measures:
                metric=init_eval(measur)
                out=metric(filtered_run,qrel)
                output[test_type][measur]=out

        with open(os.path.join(args.output_folder,ranker_name+"_eval.json"), "w") as outfile:
            json.dump(output,outfile, indent=4)

