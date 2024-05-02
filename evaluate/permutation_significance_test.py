"""
Permutation testing taken from https://github.com/dennlinger/summaries/blob/main/summaries/evaluation/significance_testing.py
"""

import json
import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
from .evaluate import init_eval




def permute(permutation, system_a_labels, system_b_lables):
    '''
    Permutes the labels across the shared axis, by switching the output based on the labels of (zero for class a, one for class b)
    permutation:permuted list of classes
    system_a_labels: true labels of system a
    system_b_lables: true labels of system b
    returns: the permuted outputs
    '''
    system_a_labels_shuffled = {}
    system_b_labels_shuffled = {}
    for flip_prob, label_a in zip(permutation, system_a_labels.keys()):  # shows if we should flip or not
        if flip_prob == 0:
            system_a_labels_shuffled[label_a]=system_a_labels[label_a]
            system_b_labels_shuffled[label_a]=system_b_lables[label_a]
        else:  # then we should flip
            system_a_labels_shuffled[label_a]=system_b_lables[label_a]
            system_b_labels_shuffled[label_a]=system_a_labels[label_a]
    return system_a_labels_shuffled, system_b_labels_shuffled

def compute_differences(qrel,system_a_labels,system_b_lables,metrics):
    differences={}
    for m in metrics.keys():

        differences[m]=abs(metrics[m](system_a_labels,qrel) - metrics[m](system_b_lables,qrel))
    return  differences

def permutation_test(qrel,system_a_labels, system_b_lables,metrics,
                     n_resamples: int = 10_000, seed: int = 25) :
    """
    Method to compute a resampling-based significance test.
    It will be tested whether system A is significantly better than system B and return the p-value.
    Permutates the predictions of A and B with 50% likelihood, and scores the altered systems A' and B' again.
    The test counts the number of times the difference between permuted scores is larger than the observed difference
    between A and B, and returns the fraction of times this is the case.
    This implementation follows the algorithm by (Riezler and Maxwell III, 2005), see:
    https://aclanthology.org/W05-0908.pdf
    :param system_a: List of predictions of system A on the test set. Assumed to be the "better" system."
    :param system_b: List of predictions of system B on the test set. Assumed to be the "baseline" method.
    :param sys_a: the class for system a
    :param sys_b: the class for system b
    :param attribute_list: list of attributes to be considered
    :param n_resamples: Number of times to re-sample the test set.
    :param seed: Random seed to ensure reproducible results.
    :return: p-value of the statistical significance that A is better than B.
    """
    p_value={}
    n_outliers_vaule={}
    for m in metrics.keys():
        p_value[m]=0
        n_outliers_vaule[m]=0

    rng = np.random.default_rng(seed=seed)
    number_of_elements = len(system_a_labels)

    if  len(system_a_labels) != len(system_b_lables):
        raise ValueError("Ensure that gold and system outputs have the same lengths!")
    base_diff_value=compute_differences(qrel,system_a_labels,system_b_lables,metrics)


    for _ in tqdm(range(n_resamples)):
        # Randomly permutes the two lists along an axis
        permutation = rng.integers(0, 2, number_of_elements)
        system_a_labels_shuffled, system_b_labels_shuffled = permute(permutation, system_a_labels, system_b_lables)

        # Check whether the current hypothesis is a "greater outlier"
        permuted_diff_value=compute_differences(qrel,system_a_labels_shuffled, system_b_labels_shuffled,metrics)
        for m in metrics.keys():
            if permuted_diff_value[m]>=base_diff_value[m]:
                n_outliers_vaule[m]+= 1

    for m in metrics.keys():
        print(m,n_outliers_vaule[m])
        p_value[m] = (n_outliers_vaule[m] + 1) / (n_resamples + 1)


    # Return the offset p-value, following (Riezler and Maxwell, 2005)
    return p_value

def get_args():
    args = ArgumentParser()

    args.add_argument("--sys-a", type=str,
                      default="qcolbert",
                      help="name of system a (your system)")

    args.add_argument("--sys-b", type=str,
                      default="colbert",
                      help="name of system b (systems to compare against), seperate by comma if more than one. ")
    args.add_argument("--gt-file", type=str,
                      default="./data/finance/qrel_compelete.json",
                      help="the compelete qrel file. ")
    args.add_argument("--input-folder", type=str, default="./data/finance/",
                      help="the folder that contians the percomputed results for system a and b. The name of the systems should be same as name of the files. ")
    args.add_argument("--measures", type=str, default="ndcg_cut@10,P@10,recall@100,MRR@10",
                      help="the metrics to be used for significant testing  ")

    return args.parse_args()

if __name__ == '__main__':
    args = get_args()

    metrics= {}
    if len(args.measures.split(","))>1:

        for measur in args.measures.split(","):
            metric=init_eval(measur)
            metrics[measur]=(metric)
    else:
        metric=init_eval(args.measures)
        metrics[args.measures]=(metric)

    with open(args.gt_file) as reader:
        qrel_compelete=json.load(reader)

    with open(os.path.join(args.input_folder,args.sys_a+".json"), "r") as reader:
        system_a=json.load(reader)

    sys_bs={}
    for name in args.sys_b.split(","):
        with open(os.path.join(args.input_folder,name+".json"), "r") as reader:
            sys_bs[name]=json.load(reader)

    for name,sys_b in sys_bs.items():
        print("evaluating against", name)
        mertic_dict = permutation_test(qrel_compelete,system_a, sys_b , metrics)

        print("system a:{}, system b:{}".format(args["sys_a"], name))
        print(mertic_dict)