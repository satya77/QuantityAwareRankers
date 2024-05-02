import ujson
import pandas as pd
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from models.semantic_matchers.colbert.utils.runs import Run


from models.semantic_matchers.colbert.parameters import DEVICE
from models.semantic_matchers.colbert.modeling.colbert import ColBERT
from models.semantic_matchers.colbert.utils.utils import print_message, load_checkpoint

import re
import os
import ujson


def get_parts(directory):
    extension = '.pt'

    parts = sorted([int(filename[: -1 * len(extension)]) for filename in os.listdir(directory)
                    if filename.endswith(extension)])

    assert list(range(len(parts))) == parts, parts

    # Integer-sortedness matters.
    parts_paths = [os.path.join(directory, '{}{}'.format(filename, extension)) for filename in parts]
    samples_paths = [os.path.join(directory, '{}.sample'.format(filename)) for filename in parts]

    return parts, parts_paths, samples_paths


def load_doclens(directory, flatten=True):
    doclens_filenames = {}

    for filename in os.listdir(directory):
        match = re.match("doclens.(\d+).json", filename)

        if match is not None:
            doclens_filenames[int(match.group(1))] = filename

    doclens_filenames = [os.path.join(directory, doclens_filenames[i]) for i in sorted(doclens_filenames.keys())]

    all_doclens = [ujson.load(open(filename)) for filename in doclens_filenames]

    if flatten:
        all_doclens = [x for sub_doclens in all_doclens for x in sub_doclens]

    if len(all_doclens) == 0:
        raise ValueError("Could not load doclens")

    return all_doclens


def get_deltas(directory):
    extension = '.residuals.pt'

    parts = sorted([int(filename[: -1 * len(extension)]) for filename in os.listdir(directory)
                    if filename.endswith(extension)])

    assert list(range(len(parts))) == parts, parts

    # Integer-sortedness matters.
    parts_paths = [os.path.join(directory, '{}{}'.format(filename, extension)) for filename in parts]

    return parts, parts_paths


def load_model(args, do_print=True):
    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)
    colbert = colbert.to(DEVICE)

    print_message("#> Loading model checkpoint.", condition=do_print)

    checkpoint = load_checkpoint(args.checkpoint, colbert, do_print=do_print)

    colbert.eval()

    return colbert, checkpoint

def load_queries(queries_path):
    queries = OrderedDict()

    print_message("#> Loading the queries from", queries_path, "...")

    with open(queries_path) as f:
        for line in f:
            qid, query, *_ = line.strip().split('\t')
            qid = int(qid)

            assert (qid not in queries), ("Query QID", qid, "is repeated!")
            queries[qid] = query

    print_message("#> Got", len(queries), "queries. All QIDs are unique.\n")

    return queries


def load_collection(collection_path,data_type):
    print_message("#> Loading collection...")

    collection = []
    df = pd.read_csv(collection_path)
    for i, line in tqdm(df.iterrows()):
        data = line[data_type]
        collection.append(data)

    # with open(collection_path) as f:
        # for line_idx, line in enumerate(f):
        #     if line_idx % (1000*1000) == 0:
        #         print(f'{line_idx // 1000 // 1000}M', end=' ', flush=True)
        #
        #     pid, passage, *rest = line.strip('\n\r ').split('\t')
        #
        #     # assert pid == 'id' or int(pid) == line_idx, f"pid={pid}, line_idx={line_idx}"
        #     import pdb
        #     pdb.set_trace()
        #     if len(rest) >= 1:
        #         title = rest[0]
        #         passage = title + ' | ' + passage
        #
        #     collection.append(passage)

    print()
    print("len collection is",str(len(collection)))
    return collection


def load_colbert(args, do_print=True):
    colbert, checkpoint = load_model(args, do_print)

    # TODO: If the parameters below were not specified on the command line, their *checkpoint* values should be used.
    # I.e., not their purely (i.e., training) default values.

    for k in ['query_maxlen', 'doc_maxlen', 'dim', 'similarity', 'amp']:
        if 'arguments' in checkpoint and hasattr(args, k):
            if k in checkpoint['arguments'] and checkpoint['arguments'][k] != getattr(args, k):
                a, b = checkpoint['arguments'][k], getattr(args, k)
                Run.warn(f"Got checkpoint['arguments']['{k}'] != args.{k} (i.e., {a} != {b})")

    if 'arguments' in checkpoint:
        if args.rank < 1:
            print(ujson.dumps(checkpoint['arguments'], indent=4))

    if do_print:
        print('\n')

    return colbert, checkpoint
