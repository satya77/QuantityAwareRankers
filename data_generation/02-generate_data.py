import random
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
from helper_functions import *
from config_classes import *
import json
from sklearn.model_selection import train_test_split
random.seed(7676)
def get_args():
    args = ArgumentParser()

    args.add_argument("--path-to-queries", type=str, default="../data/finance/querie_indicies_dict_extended.pickle",
                      help="path to the pickled file containing concepts,units index from the pervious script")
    args.add_argument("--output-folder", type=str, default="../data/finance/generation_output", help="folder where the generated training data is stored")
    args.add_argument("--path-to-concepts", type=str, default="../data/finance/concepts_to_extentions.pickle",
                      help="path to the pickled file of all the concepts and their extensions ")
    args.add_argument("--path-to-collection", type=str, default="../data/finance/collection.csv",
                      help="path to the collection of sentences in the corpus ")
    args.add_argument("--path-to-validation", type=str, default=None,
                      help="path to a folder for validation set, if set to None, no validation set will be generated ")
    args.add_argument("--samples-size", type=int, default=2,
                      help="number of samples to be generated")
    args.add_argument("--extend-concepts",  default=True,action="store_true",
                      help="whether to apply concept expansion during query generation ")
    args.add_argument("--permute-unit",  default=True,action="store_true",
                      help="whether to apply unit permutation for sample generation ")
    args.add_argument("--permute-value",  default=True,action="store_true",
                      help="whether to apply value permutation for sample generation ")
    return args.parse_args()


queries = []  # id \t query \t query_text\t
collection = []  # id\t  sentence\t  sent_text
qrel = []  # <query ID, iteration=0, passage ID, relevance>
qrel_json={} #qrels in json format, query, list of relevant ones "":{"":1}
triplets = []  # query_text \t pos_text \t neg_text
jonsl= [] #jsonl data for dpr "dataset": "finnews","question": "","answers": [],"positive_ctxs": [{"title": "","text": "text","score": 0,"title_score": 0,"passage_id": "p_id"}],"negative_ctxs":[]
triplet_jsonl=[] # the triples in terms of jsonl {"qid": query_id, "pid+": positive_id, "pid-": negative_id}
query_id = 0
passage_id = 0
debug = False
all_sentences=[] # list of all sentences

down_sample_dict={}
down_sample_dict["equal_pos"]={"concept":0,"value":0,"unit":0,"data":0}
down_sample_dict["bigger_pos"]={"concept":0,"value":0,"unit":0,"data":0}
down_sample_dict["less_pos"]={"concept":0,"value":0,"unit":0,"data":0}

down_sample_dict["equal_neg"]={"concept":0,"value":0,"unit":0,"data":0}
down_sample_dict["bigger_neg"]={"concept":0,"value":0,"unit":0,"data":0}
down_sample_dict["less_neg"]={"concept":0,"value":0,"unit":0,"data":0}
def add_to_downsample(type,sample_function):
    global down_sample_dict
    down_sample_dict[type][sample_function]=down_sample_dict[type][sample_function]+1

def get_query_values(values):
    values_sorted_length = sorted(values, key=lambda k: len(values[k]),
                                  reverse=True)  # values sorted by how many instances we have of them (to sample common ones)
    values_sorted_amount = sorted(values, key=lambda k: k,
                                  reverse=True)  # values sorted by their values (to sample smaller and bigger than)

    equal_value = values_sorted_length[0]  # equal value
    equal_value2 = values_sorted_length[1]  # second equal value

    middle = math.trunc(len(values_sorted_amount) / 2)  # index of the middle of the values
    less_than_index = middle - 1
    bigger_than_index = middle + 1

    less_than_value = values_sorted_amount[less_than_index]
    bigger_than_value = values_sorted_amount[bigger_than_index]

    if len(values_sorted_amount) < 7:
        less_than_value2 = less_than_value
        bigger_than_value2 = bigger_than_value
    else:
        less_than_value2 = values_sorted_amount[less_than_index - 1]
        bigger_than_value2 = values_sorted_amount[bigger_than_index + 1]

    return equal_value, less_than_value, bigger_than_value, equal_value2, less_than_value2, bigger_than_value2

def choose_with_exception(values,exceptions):
    choice = random.choice(values)
    while choice in exceptions:
        choice = random.choice(values)
    return choice

def get_query_values_random(values,choosen_values):
    '''
    to add more examples that are ranodm and not based on count, only used with more than 20 elements
    :param values: list of all values
    :param choosen_values: choosen_values
    :return:
    '''
    all_vals=list(values.keys())[2:-2]# remove the ends so we have some support for bigger than and smaller than

    equal_value=choose_with_exception(all_vals,choosen_values["equal"]) # equal value
    equal_value2=choose_with_exception(all_vals,choosen_values["equal"]+[equal_value])# second equal value

    less_than_value=choose_with_exception(all_vals,choosen_values["less"]) # less value
    less_than_value2=choose_with_exception(all_vals,choosen_values["less"]+[less_than_value])# second less value

    bigger_than_value=choose_with_exception(all_vals,choosen_values["bigger"]) # less value
    bigger_than_value2=choose_with_exception(all_vals,choosen_values["bigger"]+[bigger_than_value])# second less value

    return equal_value, less_than_value, bigger_than_value, equal_value2, less_than_value2, bigger_than_value2



def generate_query_text(generation_input):
    bounds = {
        "=": ["exactly","exact" , "", "=", "equals", "equals to", "equal to", "at","for","with","of"],
        ">": ["more than", "above", "larger than", "greater than", "higher than", "over", "exceeding", "exceed",  ">"],
        "<": ["less than", "fewer than", "under", "below", "smaller than", "beneath", "<","lower than"]
    }
    conf=generation_input.conf
    unit_in_text = sample_unit_pos(generation_input.unit) if conf.augment_unit_forms else generation_input.unit
    handler_text = random.choice(bounds["="])  # get a random text for the handler selected
    value= generation_input.equal_value
    if conf.augment_value_forms:
        value=alterante_repr_value(value)
    query_text_equal = "{} {} {} {}".format(generation_input.keywords.strip(), handler_text, value,unit_in_text)

    unit_in_text = sample_unit_pos(generation_input.unit) if conf.augment_unit_forms else generation_input.unit
    handler_text = random.choice(bounds[">"])  # get a random text for the handler selected
    value= generation_input.bigger_than_value
    if conf.augment_value_forms:
        value=alterante_repr_value(value)
    query_text_bigger = "{} {} {} {}".format(generation_input.keywords.strip(), handler_text,value
                                             , unit_in_text)

    unit_in_text = sample_unit_pos(generation_input.unit) if conf.augment_unit_forms else generation_input.unit
    handler_text = random.choice(bounds["<"])  # get a random text for the handler selected
    value=generation_input.less_than_value
    if conf.augment_value_forms:
        value=alterante_repr_value(value)
    query_text_less = "{} {} {} {}".format(generation_input.keywords.strip(), handler_text,value ,unit_in_text)
    return query_text_equal, query_text_less, query_text_bigger


def get_sents(generation_input, only_sentences):
    '''
    get sentences with the same concept and unit pair that conform to a condition or violate it
    :param senteces_with_indicies: the dictionary of all sentences
    :param equal_value: the value of the equal query
    :param less_than_value: the value of the less than query
    :param bigger_than_value: the value of the bigger than  query
    :param only_sentences: if set to ture only the sentences will be returned
    :return:
    '''
    equal_sentences, not_equal_sentences, other_values, same_value = filter_dictioanry(
        generation_input.senteces_with_indicies,
        equal_condition, generation_input.equal_value)
    bigger_than_sentences, not_bigger_than_sentences, other_values_bigger, values_bigger = filter_dictioanry(
        generation_input.senteces_with_indicies, bigger_than_conditoin, generation_input.bigger_than_value)
    less_than_sentences, not_less_than_sentences, other_values_less, values_less = filter_dictioanry(
        generation_input.senteces_with_indicies, less_than_condition, generation_input.less_than_value)

    if only_sentences:
        return BatchSentences(equal_sentences=list(equal_sentences.keys()),
                              not_equal_sentences=list(not_equal_sentences.keys()),
                              not_bigger_than_sentences=list(not_bigger_than_sentences.keys()),
                              bigger_than_sentences=list(bigger_than_sentences.keys()),
                              less_than_sentences=list(less_than_sentences.keys()),
                              not_less_than_sentences=list(not_less_than_sentences.keys()))

    batch_sentences = BatchSentences(equal_sentences=equal_sentences, not_equal_sentences=not_equal_sentences,
                                     not_bigger_than_sentences=not_bigger_than_sentences,
                                     bigger_than_sentences=bigger_than_sentences,
                                     less_than_sentences=less_than_sentences,
                                     not_less_than_sentences=not_less_than_sentences)
    batch_sentences.set_values(values_equal=same_value, other_values_equal=other_values, values_bigger=values_bigger,
                               other_values_bigger=other_values_bigger, values_less=values_less,
                               other_values_less=other_values_less,all_values=generation_input.senteces_with_indicies.keys())
    return batch_sentences


def sample(sentences, conf, sample_function,generation_input=None):
    global equal_down_sample,bigger_down_sample,less_down_sample
    # positive samples
    size,down_sample=adjust_sample_size(sentences.equal_sentences, conf.samples_size)
    if down_sample:add_to_downsample("equal_pos",sample_function)
    equal_pos = random.sample(sentences.equal_sentences, size)

    size,down_sample=adjust_sample_size(sentences.bigger_than_sentences, conf.samples_size)
    if down_sample:add_to_downsample("bigger_pos",sample_function)
    bigger_pos = random.sample(sentences.bigger_than_sentences,size)

    size,down_sample=adjust_sample_size(sentences.less_than_sentences, conf.samples_size)
    if down_sample:add_to_downsample("less_pos",sample_function)
    less_pos = random.sample(sentences.less_than_sentences,size)

    # negative samples
    size,down_sample=adjust_sample_size(sentences.not_equal_sentences, conf.samples_size)
    if down_sample:add_to_downsample("equal_neg",sample_function)
    equal_neg = random.sample(sentences.not_equal_sentences,size)

    size,down_sample=adjust_sample_size(sentences.not_bigger_than_sentences, conf.samples_size)
    if down_sample:add_to_downsample("bigger_neg",sample_function)
    bigger_neg = random.sample(sentences.not_bigger_than_sentences,size)

    size,down_sample=adjust_sample_size(sentences.not_less_than_sentences, conf.samples_size)
    if down_sample:add_to_downsample("less_neg",sample_function)
    less_neg = random.sample(sentences.not_less_than_sentences,size)

    



    return {'equal_pos': equal_pos, 'equal_neg': equal_neg, 'bigger_pos': bigger_pos, 'bigger_neg': bigger_neg,
            'less_pos': less_pos, 'less_neg': less_neg}


def sample_deterministic(sentences, conf, sample_function,generation_input=None):
    global equal_down_sample,bigger_down_sample,less_down_sample
    # positive samples
    sample_size=conf.samples_size*3
    size,down_sample=adjust_sample_size(sentences.equal_sentences, sample_size)
    if down_sample:add_to_downsample("equal_pos",sample_function)
    equal_pos = sentences.equal_sentences[:size]



    size,down_sample=adjust_sample_size(sentences.bigger_than_sentences, sample_size)
    if down_sample:add_to_downsample("bigger_pos",sample_function)
    bigger_pos = sentences.bigger_than_sentences[:size]

    size,down_sample=adjust_sample_size(sentences.less_than_sentences, sample_size)
    if down_sample:add_to_downsample("less_pos",sample_function)
    less_pos = sentences.less_than_sentences[:size]


    # negative samples
    size,down_sample=adjust_sample_size(sentences.not_equal_sentences, sample_size)
    if down_sample:add_to_downsample("equal_neg",sample_function)
    equal_neg = sentences.not_equal_sentences[:size]

    size,down_sample=adjust_sample_size(sentences.not_bigger_than_sentences, sample_size)
    if down_sample:add_to_downsample("bigger_neg",sample_function)
    bigger_neg = sentences.not_bigger_than_sentences[:size]

    size,down_sample=adjust_sample_size(sentences.not_less_than_sentences, sample_size)
    if down_sample:add_to_downsample("less_neg",sample_function)
    less_neg = sentences.not_less_than_sentences[:size]



    return {'equal_pos': equal_pos, 'equal_neg': equal_neg, 'bigger_pos': bigger_pos, 'bigger_neg': bigger_neg,
            'less_pos': less_pos, 'less_neg': less_neg}


def pos_neg_same_concept(generation_input):
    '''
    Create pos and neg examples form the same set with the same unit and concept pair
    :return:
    '''
    sentences = get_sents(generation_input, only_sentences=True)
    return sample(sentences, conf,"concept",generation_input)

def pos_neg_same_from_data(generation_input):
    '''
    Create pos and neg examples form the same set with the same unit and concept pair
    :return:
    '''
    sentences = get_sents(generation_input, only_sentences=True)

    returned_samples= sample(sentences, conf,"data",generation_input)
    exteneded_samples={"equal_neg":[],"bigger_neg":[],"less_neg":[]}
    #add random negative examples
    for sent in returned_samples["equal_pos"]:
        slice=random.randint(1,len(all_sentences)-1)
        exteneded_samples["equal_neg"].append(all_sentences[slice])
    for sent in returned_samples["bigger_pos"]:
        slice=random.randint(1,len(all_sentences)-1)
        exteneded_samples["bigger_neg"].append(all_sentences[slice])
    for sent in returned_samples["less_pos"]:
        slice=random.randint(1,len(all_sentences)-1)
        exteneded_samples["less_neg"].append(all_sentences[slice])

    
    returned_samples["less_neg"]=returned_samples["less_neg"]+exteneded_samples["less_neg"]
    returned_samples["bigger_neg"]=returned_samples["bigger_neg"]+exteneded_samples["bigger_neg"]
    returned_samples["equal_neg"]=returned_samples["equal_neg"]+exteneded_samples["equal_neg"]

    returned_samples["equal_pos"]=returned_samples["equal_pos"]+returned_samples["equal_pos"]
    returned_samples["bigger_pos"]=returned_samples["bigger_pos"]+returned_samples["bigger_pos"]
    returned_samples["less_pos"]=returned_samples["less_pos"]+returned_samples["less_pos"]
    return returned_samples



def get_value_augmented_sent(generation_input):
    '''
    get sentences with the same concept and unit pair that values have been agumented to accept a condition or reject it
    :param senteces_with_indicies: the dictionary of all sentences
    :param equal_value: the value of the equal query
    :param less_than_value: the value of the less than query
    :param bigger_than_value: the value of the bigger than  query
    :return:
    '''
    sentences = get_sents(generation_input, only_sentences=False)

    # negative_equal
    not_equal_sents_aug = create_augmentation_value(sentences.equal_sentences, sentences.other_values_equal,generation_input.conf.augment_value_forms)
    # positive_equal
    equal_sent_aug = create_augmentation_value(sentences.not_equal_sentences, sentences.values_equal,generation_input.conf.augment_value_forms)

    # positive bigger than
    bigger_sent_aug = create_augmentation_value(sentences.not_bigger_than_sentences, sentences.values_bigger,generation_input.conf.augment_value_forms)
    # negative bigger than
    not_bigger_sent_aug = create_augmentation_value(sentences.bigger_than_sentences, sentences.other_values_bigger,generation_input.conf.augment_value_forms)

    # positive less than
    less_sent_aug = create_augmentation_value(sentences.not_less_than_sentences, sentences.values_less,generation_input.conf.augment_value_forms)
    # negative less than
    not_less_sent_aug = create_augmentation_value(sentences.less_than_sentences, sentences.other_values_less,generation_input.conf.augment_value_forms)

    return BatchSentences(equal_sentences=equal_sent_aug, not_equal_sentences=not_equal_sents_aug,
                          not_bigger_than_sentences=not_bigger_sent_aug, bigger_than_sentences=bigger_sent_aug,
                          less_than_sentences=less_sent_aug, not_less_than_sentences=not_less_sent_aug)




def pos_neg_value_permutation(generation_input):
    '''
    Create pos and neg examples form the same set with the same unit and concept pair
    '''
    sentences = get_value_augmented_sent(generation_input)
    return sample(sentences, conf,"value",generation_input)

def get_unit_augmented_sent(generation_input):
    '''
    get sentences with the same concept and unit pair that values have been agumented to accept a condition or reject it
    :param senteces_with_indicies: the dictionary of all sentences
    :param equal_value: the value of the equal query
    :param less_than_value: the value of the less than query
    :param bigger_than_value: the value of the bigger than  query
    :param unit: the unit in question
    :return:
    '''

    # if isinstance(other_surfaceforms,tuple):
    #     other_surfaceforms_as_set=set(other_surfaceforms[0])+ set(other_surfaceforms[1])
    # unrelevant_units = list(all_surfaceforms- set(other_surfaceforms_as_set))
    sentences = get_sents(generation_input, only_sentences=False)

    # positive equal
    equal_sent_aug = create_augmentation_unit(sentences.equal_sentences, generation_input.unit)

    # negative equal
    not_equal_sentences = create_augmentation_unit(sentences.equal_sentences, generation_input.unit, negative=True)

    # positive bigger than
    bigger_sent_aug = create_augmentation_unit(sentences.bigger_than_sentences, generation_input.unit)
    # negative bigger than
    not_bigger_sent_aug = create_augmentation_unit(sentences.bigger_than_sentences, generation_input.unit,
                                                   negative=True)

    # positive less than
    less_sent_aug = create_augmentation_unit(sentences.less_than_sentences, generation_input.unit)
    # negative less than
    not_less_sent_aug = create_augmentation_unit(sentences.less_than_sentences, generation_input.unit, negative=True)

    return BatchSentences(equal_sentences=equal_sent_aug, not_equal_sentences=not_equal_sentences,
                          not_bigger_than_sentences=not_bigger_sent_aug, bigger_than_sentences=bigger_sent_aug,
                          less_than_sentences=less_sent_aug, not_less_than_sentences=not_less_sent_aug)


def pos_neg_unit_permutation(generation_input):
    '''
    Create pos and neg examples form the same set with the same unit and concept pair
    :return:
    '''

    sentences = get_unit_augmented_sent(generation_input)
    return sample(sentences, conf,"unit",generation_input)


def save_data(path, subscrip,validation_path=None):
    print("generating validation set...")
    print("writing to file...")
    queries_df = pd.DataFrame(queries)
    queries_df.to_csv(os.path.join(path, 'all_queries_' + str(subscrip) + '.tsv'), index=False, sep="\t", header=False)

    if validation_path:
        queries_df.to_csv(os.path.join(path, 'queries_' + str(subscrip) + '.tsv'), index=False, sep="\t", header=False)
        queries_df.columns=["id","query"]
        collection_df = pd.DataFrame(collection)
        validation_queries=pd.read_csv(validation_path+"val_queries.tsv",sep="\t",names=["id","query"])
        validation_queries_unique=list(validation_queries["query"].unique())
        q_in_val=list(queries_df[queries_df["query"].isin(validation_queries_unique)]["id"])

        qrel_df = pd.DataFrame(qrel)
        qrel_df=qrel_df[~qrel_df['query_id'].isin(q_in_val)]
        print("number of queries existing:",str(len(q_in_val)))
        full_qrel={}
        val_qrel={}
        for line in qrel_json:
            if int(line) in q_in_val:
                val_qrel[line]=qrel_json[line]
            else:
                full_qrel[line]=qrel_json[line]
        triplet_df = pd.DataFrame(triplets)
        triplet_df=triplet_df[~triplet_df['query_text'].isin(validation_queries_unique)]

    else:
        train_queries, val_queries= train_test_split(queries_df, test_size=100, random_state=77, shuffle=True)
        train_queries.to_csv(os.path.join(path, 'queries_' + str(subscrip) + '.tsv'), index=False, sep="\t", header=False)
        val_queries.to_csv(os.path.join(path, 'val_queries_' + str(subscrip) + '.tsv'), index=False, sep="\t", header=False)
        val_queries.columns=["id","query_text"]

        qrel_df = pd.DataFrame(qrel)
        val_qrels=qrel_df[qrel_df['query_id'].isin(val_queries.id.tolist())]
        qrel_df=qrel_df[~qrel_df['query_id'].isin(val_queries.id.tolist())]

        val_qrels.to_csv(os.path.join(path, 'val_qrel_' + str(subscrip) + '.tsv'), index=False, sep="\t", header=False)


        collection_df = pd.DataFrame(collection)
        val_collection=collection_df[collection_df['id'].isin(val_qrels.sent_id.tolist())]
        val_collection.to_csv(os.path.join(path, 'val_collection_' + str(subscrip) + '.tsv'), index=False, sep="\t",header=False)

        queries_S=list(val_queries["id"].unique())
        full_qrel={}
        val_qrel={}
        for line in qrel_json:
            if int(line) in queries_S:
                val_qrel[line]=qrel_json[line]
            else:
                full_qrel[line]=qrel_json[line]

        triplet_df = pd.DataFrame(triplets)
        triplet_df=triplet_df[~triplet_df['query_text'].isin(val_queries.query_text.tolist())]
        with open(os.path.join(path, 'val_qrel_' + str(subscrip) + '.json'), 'w') as f:
            f.write(json.dumps(val_qrel, indent=4))


    collection_df.to_csv(os.path.join(path, 'collection_' + str(subscrip) + '.tsv'), index=False, sep="\t",
                         header=False)
    qrel_df.to_csv(os.path.join(path, 'qrel_' + str(subscrip) + '.tsv'), index=False, sep="\t", header=False)

    triplet_df.to_csv(os.path.join(path, 'triplets_' + str(subscrip) + '.tsv'), index=False, sep="\t", header=False)

    with open(os.path.join(path, 'jsonl_' + str(subscrip) + '.json'), 'w') as f:
        f.write("[\n")
        for line in jonsl:
            f.write(json.dumps(line, indent=4) + ",\n")
        f.write("]")

    with open(os.path.join(path, 'jsonl_triplet_' + str(subscrip) + '.json'), 'w') as f:
        for line in triplet_jsonl:
            f.write(json.dumps(line) + "\n")
    with open(os.path.join(path, 'qrel_json_' + str(subscrip) + '.json'), 'w') as f:
        f.write(json.dumps(full_qrel, indent=4))




def add_queries(new_queries):
    ids = []
    global query_id
    for new_query in new_queries:
        queries.append({"id": query_id, "query_text": new_query})
        ids.append(query_id)
        query_id = query_id + 1
    return ids


def add_passages(passages):
    ids = []
    global passage_id
    for passage in passages:
        collection.append({"id": passage_id, "sentence": passage})
        ids.append(passage_id)
        passage_id = passage_id + 1
    return ids


def add_qrel(q_id, p_ids, tag):
    if q_id not in qrel_json:
        qrel_json[q_id]={}
    for p_id in p_ids:
        qrel.append({"query_id": q_id, "number": 0, "sent_id": p_id, "relevence": tag})
        if tag==1:
            qrel_json[q_id][p_id]=1

def add_triples(query_text, pos_paassges, neg_passages):
    for pos,neg in zip(pos_paassges,neg_passages):
        triplets.append({"query_text":query_text,"pos_passage":pos,"negative_passage":neg})

def add_jsonl(q_id,query_text, pos_paassges, neg_passages,current_p_ids_pos,current_p_ids_neg):
    positive_ctxs=[]
    negative_ctxs=[]
    for pos,pos_id in zip(pos_paassges,current_p_ids_pos):
        positive_ctxs.append({"title": "","text": pos,"score": 0,"title_score": 0,"passage_id": pos_id})
    for neg,neg_id in zip(neg_passages,current_p_ids_neg):
        negative_ctxs.append({"title": "","text": neg,"score": 0,"title_score": 0,"passage_id": neg_id})
    jonsl.append({"dataset": "finnews","question": query_text,"answers": [],"positive_ctxs":positive_ctxs,"negative_ctxs":negative_ctxs})
    for p_id,n_id in zip(current_p_ids_pos,current_p_ids_neg):
        triplet_jsonl.append({"qid":q_id,"pid+":p_id,"pid-":n_id})

def add_to_dataset_per_query(q_id,query_text, pos, neg):
    # add the passages to the collection
    current_p_ids_pos = add_passages(pos)
    current_p_ids_neg = add_passages(neg)
    # add the tags for positive passages
    add_qrel(q_id, current_p_ids_pos, 1)
    add_qrel(q_id, current_p_ids_neg, 0)
    #add triples
    add_triples(query_text, pos, neg)
    #add jsonl
    add_jsonl(q_id,query_text,pos,neg,current_p_ids_pos,current_p_ids_neg)
    return passage_id


def generate_data(generation_functions, generation_input):
    # generate text for the concept as it is
    query_text_equal, query_text_less, query_text_bigger = generate_query_text(generation_input)
    # add the queries to the dataset
    current_q_ids = add_queries([query_text_equal, query_text_less, query_text_bigger])
    # accumulate the pos and neg samples in a list
    equal_pos = []
    bigger_than_pos = []
    less_than_pos = []
    equal_neg = []
    bigger_than_neg = []
    less_than_neg = []

    # positives and negatives from the data

    for func in generation_functions:
        results = func(generation_input)
        equal_pos.extend(results["equal_pos"])
        equal_neg.extend(results["equal_neg"])
        less_than_pos.extend(results["less_pos"])
        less_than_neg.extend(results["less_neg"])
        bigger_than_pos.extend(results["bigger_pos"])
        bigger_than_neg.extend(results["bigger_neg"])

    add_to_dataset_per_query(current_q_ids[0],query_text_equal, equal_pos, equal_neg)
    add_to_dataset_per_query(current_q_ids[1],query_text_less, less_than_pos, less_than_neg)
    add_to_dataset_per_query(current_q_ids[2],query_text_bigger, bigger_than_pos, bigger_than_neg)

    if debug == True:
        print("Equal: ", query_text_equal)
        print("positives:")
        for p in equal_pos: print(p)
        print("****")
        print("negatives:")
        for p in equal_neg: print(p)
        print("****")
        print("bigger: ", query_text_bigger)
        print("positives:")
        for p in bigger_than_pos: print(p)
        print("****")
        print("negatives:")
        for p in bigger_than_neg: print(p)
        print("****")
        print("less: ", query_text_less)
        print("positives:")
        for p in less_than_pos: print(p)
        print("****")
        print("negatives:")
        for p in less_than_neg: print(p)
        print("****")

if __name__ == "__main__":
    arg = get_args()

    path_to_validation=arg.path_to_validation
    output_folder =arg.output_folder+"/"
    # read the query and sentences
    with open(arg.path_to_queries, 'rb') as fp:
        querie_indicies_dict = pickle.load(fp)

    # read the concept dictionaries for query expansion
    with open(arg.path_to_concepts, 'rb') as fp:
        concepts_to_extentions = pickle.load(fp)
    augment_value_forms=True if arg.permute_value else False
    augment_unit_forms=True if arg.permute_unit else False


    conf = GenerationConf(samples_size=arg.samples_size,extend_concepts=arg.extend_concepts,permute_unit_surface_forms=arg.permute_unit,permute_values=arg.permute_value,
                          augment_value_forms=augment_value_forms,augment_unit_forms=augment_unit_forms)

    all_sentences=list(pd.read_csv(arg.path_to_collection)["sentence"].unique())


    save_name=""
    generation_functions = [pos_neg_same_from_data]
    #concept premutation


    if conf.extend_concepts:
        save_name=save_name+"concept_"
        generation_functions.append(pos_neg_same_concept)
        output_folder=output_folder+"concept_"
    # positives and negatives from value permutation
    if conf.permute_values:
        save_name=save_name+"value_"
        generation_functions.append(pos_neg_value_permutation)
        output_folder=output_folder+"value_"
    if conf.augment_value_forms:
        save_name=save_name+"surf_"
    # positives and negatives from unit permutation
    if conf.permute_unit_surface_forms:
        save_name=save_name+"unit_"
        generation_functions.append(pos_neg_unit_permutation)
        output_folder=output_folder+"unit_"

    save_name= save_name+"aug"
    output_folder=output_folder+"aug"

    isExist = os.path.exists(output_folder)
    if not isExist:
        os.makedirs(output_folder)
        print("The new output directory is created! "+output_folder)


    counter=0
    for key, senteces_with_indicies in tqdm(querie_indicies_dict.items()):
        counter=counter+1

        keyword = key[0]
        unit = key[1]

        equal_value, less_than_value, bigger_than_value, equal_value2, less_than_value2, bigger_than_value2 = get_query_values(
            senteces_with_indicies)

        #normal generation
        generation_input = GenerationInput(senteces_with_indicies, equal_value, less_than_value, bigger_than_value,
                                           unit, conf, keyword)
        generate_data(generation_functions, generation_input)


        # generate text for the extented concept
        if conf.extend_concepts:
            if debug == True:
                print("extended concept for {} is {}: ".format(keyword, is_in_dictionary(concepts_to_extentions, keyword)))
            extention = is_in_dictionary(concepts_to_extentions, keyword)
            generation_input = GenerationInput(senteces_with_indicies, equal_value2, less_than_value2,
                                               bigger_than_value2, unit, conf, extention)
            generate_data(generation_functions, generation_input)

        if len(senteces_with_indicies)>8 and conf.permute_values:#12 before
            all_choosen_vals={"equal":[equal_value,equal_value2],"less":[less_than_value,less_than_value2],"bigger":[ bigger_than_value, bigger_than_value2]}
            function_value=[pos_neg_value_permutation]
            equal_value3, less_than_value3, bigger_than_value3, equal_value4, less_than_value4, bigger_than_value4=get_query_values_random(senteces_with_indicies,all_choosen_vals)
            generation_input_random = GenerationInput(senteces_with_indicies, equal_value3, less_than_value3, bigger_than_value3,
                                               unit, conf, keyword)

            generate_data(function_value, generation_input_random)
            if conf.extend_concepts:
                extention = is_in_dictionary(concepts_to_extentions, keyword)

                generation_input_random2 = GenerationInput(senteces_with_indicies, equal_value4, less_than_value4, bigger_than_value4,
                                                      unit, conf, extention)
                generate_data(function_value, generation_input_random2)

    print(down_sample_dict)
    # save the data
    save_data(output_folder, save_name,path_to_validation)




