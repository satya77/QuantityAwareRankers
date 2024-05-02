"""
Code to test the cohere api and their embeddings.
"""
import os.path
from argparse import ArgumentParser

import cohere
import hnswlib
import pandas as pd
from tqdm import tqdm
from dataset.bm25_dataloaders import read_test_queries
from .evaluate import filter_results, init_eval
import json

def get_args():
    args = ArgumentParser()

    args.add_argument("--corpus-path", type=str,
                      default="./data/finance/collection.csv",
                      help="the path to the sentence collection.")
    args.add_argument("--test-queries-path", type=str,
                      default="./data/finance/queries.tsv",
                      help="path to the queries.  ")
    args.add_argument("--gt-folder", type=str,
                      default="./data/finance",
                      help="folder with ground truth relevances on different subsets in json files ")
    args.add_argument("--api-key", type=str,
                      help="cohere api key. ")
    args.add_argument("--output-folder", type=str, default="./data/finance/test",
                      help="The folder to save the index and output of the evaluation to")
    args.add_argument("--create-index",  default=True,action="store_true",
                      help="whether to create an index from scratch, if set to false the current index will be used for evaluation ")

    return args.parse_args()


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    args = get_args()

    create_index=True
    qrels={}
    gt_file=args.gt_folder

    measures=["MRR@10","MRR@100","MRR@1000", "recall@10", "P@10","P@100", "recall@100","recall@1000", "ndcg_cut@10",  "ndcg_cut@100", "ndcg_cut@1000"]

    with open(os.path.join(gt_file,"qrel_compelete.json")) as reader:
        qrel_compelete=json.load(reader)
        qrels["compelete"]=qrel_compelete
    with open(os.path.join(gt_file,"qrel_in_training.json")) as reader:
        qrel_in_training=json.load(reader)
        qrels["in_training"]=qrel_in_training
    with open(os.path.join(gt_file,"qrel_not_training.json")) as reader:
        qrel_not_training=json.load(reader)
        qrels["not_training"]=qrel_not_training
    with open(os.path.join(gt_file,"qrel_expanded.json")) as reader:
        qrel_expanded=json.load(reader)
        qrels["expanded"]=qrel_expanded
    with open(os.path.join(gt_file,"qrel_wo_surface.json")) as reader:
        qrel_expanded_wo_surface=json.load(reader)
        qrels["expanded_wo_surface"]=qrel_expanded_wo_surface

    with open(os.path.join(gt_file,"qrels_equal.json")) as reader:
        qrels_equal=json.load(reader)
        qrels["qrels_equal"]=qrels_equal

    with open(os.path.join(gt_file,"qrels_greater.json")) as reader:
        qrels_bigger=json.load(reader)
        qrels["qrels_bigger"]=qrels_bigger

    with open(os.path.join(gt_file,"qrels_less.json")) as reader:
        qrels_smaller=json.load(reader)
    qrels["qrels_smaller"]=qrels_smaller
    
    
    co = cohere.Client(args.api_key)
    df = pd.read_csv(args.corpus_path)

    ids=[]
    index = hnswlib.Index(space='ip', dim=1024)

    if not os.path.exists(args.output_folder+"/cohere_hnsw"):
        index.init_index(max_elements=len(df), ef_construction=512, M=64)
    else:
        index.load_index(args.output_folder+"/cohere_hnsw", max_elements = len(df))
    print(f"Index size is {index.element_count} and index capacity is {index.max_elements}")
    if create_index:
        documents=[]
        ids=[]
        batch_size=500
        batch_number=int(len(df)/float(batch_size))+2
        for b in range(1, batch_number):
            df_sub=df[(b-1)*batch_size:b*batch_size]
            doc_embs = co.embed(texts=list(df_sub["sentence"]), model='embed-english-v3.0', input_type="search_document").embeddings
            index.add_items(doc_embs, list(df_sub["Unnamed: 0"]))
            print(f"Index size is {index.element_count}")
            index.save_index(args.output_folder+"/cohere_hnsw")
            
    run_q={}
    output={}
    test_qs=read_test_queries(args.test_queries_path)
    for i,query in tqdm(enumerate(test_qs)):
        query_text=query["query_text"]

        query_emb = co.embed(texts=[query_text], model='embed-english-v3.0', input_type="search_query").embeddings
        labels, distances = index.knn_query(query_emb, k=10)

        run_q[str(i)]={}
        for la,dist in zip(labels[0], distances[0]):# 3:37 min
            run_q[str(i)][str(df.iloc[la]["Unnamed: 0"])]= float(dist)
    with open(args.output_folder+"/cohere.json", "w") as outfile:
        json.dump(run_q, outfile, indent = 4)
    
    for test_type, qrel in qrels.items():
        filtered_run=filter_results(run_q, qrel)
        output[test_type]={}
        for measur in measures:
            metric=init_eval(measur)
            out=metric(filtered_run,qrel)
            output[test_type][measur]=out
 
    with open(os.path.join(args.output_folder,"cohere_eval.json"), "w") as outfile:
        json.dump(output,outfile, indent=4)