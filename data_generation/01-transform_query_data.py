"""
Transforms a csv file with unit/concept pointing to sentences and values
to a dictionary of dictionary to be used in the next stage
"""
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm
import pickle
from collections import  defaultdict
from ast import literal_eval
import os

def get_args():
    args = ArgumentParser()

    args.add_argument("--data-type", type=str, default="finance",
                      help="either finance or clinical")
    args.add_argument("--input-queries", type=str, default="../data/finance/train_concepts_units_extended.csv",
                      help="a csv file that has a column `keyword` indicating the concpets and `units` indicating units. It points to `values` and `sentences` and their positions in text with `values_char` and `unit_char`")
    args.add_argument("--input-collection", type=str, default="../data/finance/collection.csv",
                      help="Input collection of sentences.")
    args.add_argument("--generate-masked-value", type=bool, default=False,
                      help="Whether to generated masked value collection. ")
    args.add_argument("--generate-masked-unit", type=bool, default=False,
                      help="Whether to generated masked unit collection. ")
    args.add_argument("--output-path", type=str, default="../data/finance",
                      help="Path to output folder. ")

    return args.parse_args()

if __name__ == "__main__":
    args = get_args()

    # This is the unit/concept index that has unit/concepts, pointing to a list of values and sentences
    # values_char and unit_char indicate the positions of the values and unit in text
    #this is essential for generation phase
    queries_df = pd.read_csv(args.input_queries, index_col=0,converters={"values": literal_eval,
                                                                      "values_char": literal_eval,
                                                                      "unit_char": literal_eval,
                                                                      "sentences": literal_eval})

    querie_indicies_dict= {}
    for index, item in tqdm(queries_df.iterrows(), total=queries_df.shape[0]):
        querie_indicies_dict[(item["keyword"],item["unit"])]= defaultdict(dict)
        for i,val in enumerate(item["values"]):# dictionary of values-> [list of sentences with that value]
            querie_indicies_dict[(item["keyword"],item["unit"])][val][item["sentences"][i]]=(item["values_char"][i],item["unit_char"][i])
    with open(os.path.join(args.output_path,"querie_indicies_dict_extended.pickle"), 'wb') as fp:
        pickle.dump(querie_indicies_dict, fp)



    if args.generate_masked_value:
        collection=pd.read_csv(args.input_collection, index_col=0,converters={
            "value_char_indices": literal_eval,
        })
        for index, item in tqdm(collection.iterrows(), total=collection.shape[0]):
            sent=item["sentence"]
            replace=[]
            for char_index in item["value_char_indices"]:
                replace.append(sent[char_index[0][0]:char_index[0][1]])
            for rep in replace:
                sent=sent.replace(rep,"[MASK]")
            collection.iloc[index]["sentence"]=sent
        collection.to_csv(os.path.join(args.output_path,'collection_masked.csv'))


    if args.generate_masked_unit:
        collection=pd.read_csv(args.input_collection, index_col=0,converters={
            "unit_char_indices": literal_eval,
        })
        for index, item in tqdm(collection.iterrows(), total=collection.shape[0]):
            sent=item["sentence"]
            replace=[]
            for char_index in item["unit_char_indices"]:
                if len(char_index)>0:
                    replace.append(sent[char_index[0][0]:char_index[0][1]])
            for rep in replace:
                sent=sent.replace(rep,"[MASK]")
            collection.iloc[index]["sentence"]=sent
        collection.to_csv(os.path.join(args.output_path,'collection_masked_unit.csv'))







