import itertools
import math
import random
from collections import defaultdict
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from CQE import CQE
import pickle
import humanize

noun_path= "./noun_units.pickle"

suffixes = {
    'billion':['billion','B','b','bn'],
    'million':['M','million'],
    'thousand':['thousand','K','k'],
    'trillion':['tn','trillion']
}
millnames = ['','thousand','million','billion','trillion']

parser = CQE.CQE(overload=True)
all_forms = parser.get_unit_hierarchy()
all_surfaceforms = defaultdict(list)
all_symbols = defaultdict(list)
hierarchy = defaultdict(list)
hierarchy["radioactivity"]=["centigray"]
hierarchy["electric current"]=["milliampere"]
hierarchy["power"]=["milliwatt"]

for type, surf in all_forms.items():
    hierarchy[type].extend(list(surf.keys()))
    for s in surf.values():
        all_surfaceforms[type].extend(s["surfaces"])
        all_symbols[type].extend(s["symbols"])

total_down_sample=0

def is_scientific_unit(unit):
    other_units = parser.get_specific_unit_surface_forms(unit)
    if len(other_units) < 1:
        return False
    return True


def get_unit_grouping(unit):
    for type, typed_unit in hierarchy.items():
        if unit in typed_unit:
            return type

if os.path.isfile(noun_path):
    with open(noun_path, 'rb') as fp:
        noun_units = pickle.load(fp)
else:
    noun_units=set()

def sample_unit_pos(unit,unit_surface_forms=None,symbol=False):
    global noun_units
    if unit_surface_forms is None:
        unit_surface_forms = parser.get_specific_unit_surface_forms(unit)
    if len(unit_surface_forms)==0:
        noun_units.add(unit)
        with open(noun_path, 'wb') as fp:
            pickle.dump(noun_units, fp)
        return unit
    if unit in unit_surface_forms:
        if symbol:# if in the text there is symbol replace it with another symbol (if possible) otherwise with a surface form
            if "symbols" in unit_surface_forms[unit] and len(unit_surface_forms[unit]["symbols"])>0:
                return random.choice(unit_surface_forms[unit]["symbols"])

            else:
                return random.choice(unit_surface_forms[unit]["surfaces"])
        else:
            return random.choice(unit_surface_forms[unit]["surfaces"]+unit_surface_forms[unit]["symbols"])

def is_in_dictionary(dictionary, concept):
    concept_parts = concept.split(" ")
    if concept in dictionary:
        return dictionary[concept]
    if concept[:-1] in dictionary:
        return dictionary[concept[:-1]]
    if len(concept_parts)>4:
        return False
    all_combos = itertools.product(concept_parts, repeat=len(concept_parts))
    all_combos = list(all_combos)
    all_combo_strings = [" ".join(list(b)) for b in all_combos]
    if len(all_combo_strings) > 0:
        for combo in all_combo_strings:
            if combo in dictionary:
                return dictionary[combo]
    else:
        return False
def get_same_type_unit(unit_name):
    first_unit_dict=parser.get_specific_unit_surface_forms(unit_name)
    if len(first_unit_dict)>0:
        first_unit_dict=first_unit_dict[unit_name]
        combined_first_unit=first_unit_dict["symbols"]+first_unit_dict["surfaces"] if "symbols" in first_unit_dict else first_unit_dict["surfaces"]
        unit_type_1=get_unit_grouping(unit_name)
        same_type_all_surfaces_1=list(set(all_surfaceforms[unit_type_1])-set(combined_first_unit))
        same_type_all_symboles_1=list(set(all_symbols[unit_type_1])-set(combined_first_unit))
        same_type_all_1=same_type_all_surfaces_1+same_type_all_symboles_1

    else:
        same_type_all_1=[unit_name]
    return  same_type_all_1


def sample_unit_neg(unit,unit_surface_forms,symbol=False):
    if unit_surface_forms is None:
        unit_surface_forms = parser.get_specific_unit_surface_forms(unit)

    if len(unit_surface_forms)==0:#if it is non-scientific
        return random.choice(list(noun_units))

    if len(unit_surface_forms)==1:
            unit_dict=unit_surface_forms[unit]
            combined=unit_dict["symbols"]+unit_dict["surfaces"] if "symbols" in unit_dict else unit_dict["surfaces"]
            unit_type=get_unit_grouping(unit)
            same_type_all_surfaces=list(set(all_surfaceforms[unit_type])-set(combined))
            same_type_all_symboles=list(set(all_symbols[unit_type])-set(combined))
            if symbol and len(same_type_all_symboles)>0:
                return random.choice(same_type_all_symboles)
            else:
                return random.choice(same_type_all_surfaces)



    if len(unit_surface_forms)>1:
        unit_parts = list(unit_surface_forms.keys())
        unit_parts.remove(unit)
        first_unit,second_unit=unit_parts[0],unit_parts[1]
        connector=unit.replace(unit_parts[0],"").replace(unit_parts[1],"")# might be "a" "per" ....
        same_type_all_1=get_same_type_unit(first_unit)
        same_type_all_2=get_same_type_unit(second_unit)
        flip = random.randint(0, 1)# randomly replaces the first or the second unit in the combined unit examples to make it more difficult
        if connector in unit and connector not in first_unit and connector not in second_unit:
            if (flip == 0):
                unit_in_text = first_unit + connector + random.choice(same_type_all_2)
            else:
                unit_in_text = random.choice(same_type_all_1) + connector + second_unit
        else:
            if (flip == 0):
                unit_in_text = first_unit + " " + random.choice(same_type_all_2)
            else:
                unit_in_text = random.choice(same_type_all_1) + " " + second_unit

        return unit_in_text.replace("  ","")



def adjust_sample_size(sentences, samples_size):
    global total_down_sample
    if len(sentences) < samples_size:
        total_down_sample = total_down_sample +1
        return len(sentences),True
    else:
        return samples_size, False

def formatNumber(num):
    if num % 1 == 0:
        return int(num)
    else:
        return num


def alterante_repr_value(value):

    n = float(value)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    mil_exp=millnames[millidx]
    if mil_exp=="":# if there is no million or billions
        replace_str=str(formatNumber(n))
        return replace_str
    else:# if there are post fixes
        flip = random.randint(0, 1)
        if (flip == 0):# either return one with the commas
            return humanize.intcomma(formatNumber(n))
        else:# or return on with the post fix
            surface=random.choice(suffixes[mil_exp])
            origional_number=str(formatNumber(n))
            string_format=humanize.intword(formatNumber(n))

            for char in origional_number:
                if char not in string_format:
                    return origional_number
            if surface==mil_exp:
                #normlized form
                return string_format.replace(".0 "," ")
            else:
               # one of the other surface forms
                return string_format.replace(".0 "," ").replace(" "+mil_exp,surface)

def create_augmentation_value(sentences, replace_values,augment_value_forms=False):
    sents_aug = []

    for sent, indicies in sentences.items():
        replace_val = random.choice(replace_values)

        if augment_value_forms:
            replace_str=alterante_repr_value(replace_val)
        else:
            replace_str=str(formatNumber(replace_val))
        value_idx = indicies[0][0]
        new_sent = sent[:value_idx[0]] + replace_str + sent[value_idx[1]:]
        sents_aug.append(new_sent)
    return sents_aug




def create_augmentation_unit(sentences,unit, negative=False):
    sents_aug = []
    unit_surface_forms=parser.get_specific_unit_surface_forms(unit)
    for sent, indicies in sentences.items():

        value_idx = indicies[1][0]
        if negative:
            if (value_idx[1] - value_idx[0]) == 1:
                replace_val=sample_unit_neg(unit,unit_surface_forms,symbol=True)
            else:
                replace_val=sample_unit_neg(unit,unit_surface_forms)
        else:
            if (value_idx[1] - value_idx[0]) == 1:
                replace_val=sample_unit_pos(unit,unit_surface_forms,symbol=True)
            else:
                replace_val=sample_unit_pos(unit,unit_surface_forms)

        new_sent = sent[:value_idx[0]] + str(replace_val) + sent[value_idx[1]:]
        sents_aug.append(new_sent)
    return sents_aug


def filter_dictioanry(senteces_with_indicies, filter_conditon, filter_value):
    with_conditon = {}
    not_condition = {}
    other_values = []
    values = []
    for k, v in senteces_with_indicies.items():
        if filter_conditon(k, filter_value):
            with_conditon.update(senteces_with_indicies[k])
            values.append(k)
        else:
            if k!=filter_value:#@TODO added newly
                not_condition.update(senteces_with_indicies[k])
                other_values.append(k)
    return with_conditon, not_condition, other_values, values


def equal_condition(k, filter_value):
    return k == filter_value


def bigger_than_conditoin(k, filter_value):
    return k > filter_value


def less_than_condition(k, filter_value):
    return k < filter_value




