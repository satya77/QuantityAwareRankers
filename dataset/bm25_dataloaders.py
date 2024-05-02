import re
from ast import literal_eval

import nltk
import unidecode
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
import pandas as pd

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
from os import listdir


def word_clean(words):
    '''
    remove non-latin words and perform simple clean ups to make the unit and number detection better
    :param words: a sentence
    :return:
    '''
    words= words.replace("<num>"," ")
    RE = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)
    line = re.sub(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', u'', words)
    line = unidecode.unidecode(line)
    line = line.encode("ascii", errors="ignore").decode()
    line = RE.sub('', line)
    return line


def isPunctuation(inputString):  # is token a punctuation
    return inputString in "!\"#&'()*+, -./:;<=>?@[\]^_`{|}~"


def tokenizer_stemmer(sentence):
    """
    Perform tokenization for BM25, where the numbers are also tokenized like the rest of the words
    :param sentence: a sentence
    :return:
    """
    cleaned = word_clean(sentence)
    list_terms = [stemmer.stem(w.lower()) for w in nltk.word_tokenize(cleaned) if w.lower() not in stop_words and not isPunctuation(w)]
    return (list_terms)


def find_tsv_filenames(path_to_dir, suffix=".tsv"):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def process_data_bm25(data_dir):
    """
    Reads the collection csv file and tokenizes them based on bm25 and retruns the quantities
    
    :param data_dir:  where the data collection is located
    :return:
    """
    print("Preloading dataset")
    df = pd.read_csv(data_dir)
    for i, line in tqdm(df.iterrows()):
        data = line["normalized_sentence"]
        value_list=literal_eval(line["values"])
        unit_list=literal_eval(line["units"])
        tuples = [(key, value) for i, (key, value) in enumerate(zip(unit_list, value_list))]
        data_processed = tokenizer_stemmer(data)
        yield data_processed, tuples, line["sentence"]

def read_test_queries(data_file):
    """
    Reads the test queries and creates a generator for paring it
    :param data_file: path to file
    :return:
    """
    print("Preloading test queries")
    df = pd.read_csv(data_file, sep="\t")
    for i, line in tqdm(df.iterrows()):
        query_text_proccessed = [stemmer.stem(w.lower()) for w in line["keywords"].split(" ")]
        test_query={"query_text":line["query_text"],"keywords":line["keywords"],'query_text_proccessed':query_text_proccessed,
                    "unit":line["unit"],"value":float(line["value"]),"condition":line["condition"]}
        yield test_query


