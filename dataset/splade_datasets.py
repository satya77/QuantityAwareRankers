
from ast import literal_eval
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm

class CollectionDatasetNumerical(Dataset):
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    we preload everything in memory at init
    """

    def __init__(self, data_dir,sent_col):
        self.data_dir = data_dir
        self.data_dict = {}
        self.line_dict = {}
        self.number_dict = {}
        print("Preloading dataset")
        self.df = pd.read_csv(data_dir)
        for i, line in tqdm(self.df.iterrows()):

            if len(line) > 1:
                data =line[sent_col]
                value_list=literal_eval(line["values"])
                unit_list=literal_eval(line["units"])

                tuples = [(key, value)  for i, (key, value) in enumerate(zip(unit_list, value_list)) if key!="-"]
                result={}
                for key, value in tuples:
                    result.setdefault(key, []).append(value)
                self.data_dict[i] = data.strip()
                self.number_dict[i] = result

        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        return str(idx), self.data_dict[idx], self.number_dict[idx]


class CollectionDatasetPreLoad(Dataset):
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    we preload everything in memory at init
    """

    def __init__(self, data_dir,sent_col):
        self.data_dir = data_dir
        self.data_dict = {}
        self.line_dict = {}
        print("Preloading dataset")
        self.df = pd.read_csv(data_dir)
        for i, line in tqdm(self.df.iterrows()):
            if len(line) > 1:
                data = line[sent_col]
                self.data_dict[i] = data.strip()

        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        return str(idx), self.data_dict[idx]
