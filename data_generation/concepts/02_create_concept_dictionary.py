"""
Basically converts the csv file from the pervious stage to a pickle file
"""
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
import pickle

def get_args():
    args = ArgumentParser()

    args.add_argument("--data-type", type=str, default="finance",
                      help="either finance or clinical, used to select the correct prompt")
    args.add_argument("--input-file", type=str, default="../../data/finance/concepts_extended.csv",
                      help="a csv file that has the concepts and extensions in `extension` column")
    args.add_argument("--output-file", type=str, default="../../data/finance/concepts_to_extentions.pickle",
                      help="Path to output a pickle file including all the extensions in a dictionary. ")

    return args.parse_args()

if __name__ == "__main__":
    args = get_args()
    concepts_df=pd.read_csv(args.input_file, index_col=0)
    concepts_to_extentions={}# concept -> its extention
    extention_data=[]

    print("reading the concept file and creating a dictionary")
    for con,extension in tqdm(concepts_df.iterrows(), total=concepts_df.shape[0]):
        concept=con
        concepts_extention=extension["extension"]
        concepts_to_extentions[concept]=concepts_extention

    print(len(concepts_to_extentions))
    with open(args.output_file, 'wb') as fp:
        pickle.dump(concepts_to_extentions, fp)



