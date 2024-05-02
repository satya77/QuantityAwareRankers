class GenerationConf:
    def __init__(self, samples_size:int, extend_concepts:bool=True, permute_unit_surface_forms:bool=True,
                 permute_values:bool=True,augment_value_forms:bool=True,
                 augment_unit_forms:bool=True):
        '''

        :param samples_size: number of samples pos and neg that come from the same concept and unit pair
        :param:extend_concepts: whether to extend the concepts for queries
        :param:permute_unit_surface_forms:whether to permute the unit surface forms for negative samples
        :param:permute_values: whether to permute the values for negative samples
        :param:augment_value_forms: should the value representation be augmented if false only the normalized value is represnted
        :param augment_unit_forms: should units be repsented in various form, if set to false then only normalized units are considered
        '''
        self.samples_size = samples_size
        self.extend_concepts = extend_concepts
        self.permute_unit_surface_forms = permute_unit_surface_forms
        self.permute_values = permute_values
        self.augment_value_forms=augment_value_forms
        self.augment_unit_forms=augment_unit_forms
        if permute_unit_surface_forms:# if we permute them the we are using different surfaces
            self.augment_unit_forms=True
        else:
            self.augment_unit_forms=False

# for passing a batch of positive and negative examples between the functions
class BatchSentences:
    def __init__(self, equal_sentences:list, not_equal_sentences:list, not_bigger_than_sentences:list, bigger_than_sentences:list,
                 less_than_sentences:list, not_less_than_sentences:list):
        '''

        :param equal_sentences: sentences that satisfy the equal condition
        :param not_equal_sentences: sentences that do not satisfy the equal condition
        :param not_bigger_than_sentences: sentences that do not satisfy the bigger than condition
        :param bigger_than_sentences: sentences that satisfy the bigger than condition condition
        :param less_than_sentences: sentences that satisfy the less than condition condition
        :param not_less_than_sentences: sentences that do not satisfy the less than condition
        '''
        self.equal_sentences = equal_sentences
        self.not_equal_sentences = not_equal_sentences
        self.other_values_equal = None
        self.values_equal = None
        self.not_bigger_than_sentences = not_bigger_than_sentences
        self.bigger_than_sentences = bigger_than_sentences
        self.other_values_bigger = None
        self.values_bigger = None
        self.less_than_sentences = less_than_sentences
        self.not_less_than_sentences = not_less_than_sentences
        self.other_values_less = None
        self.values_less = None
        self.all_values=None

    def set_values(self, values_equal:list[float], other_values_equal:list[float], values_bigger:list[float], other_values_bigger:list[float], values_less:list[float],
                   other_values_less:list[float],all_values:list[float]):
        '''
        to pass the values inside the sentences that satisfy or do not satisfy a condition for value permutation functions
        :param values_equal: values that satisfy the equal condition
        :param other_values_equal: values that do not satisfy the equal condition
        :param values_bigger: values that do not satisfy the bigger than condition
        :param other_values_bigger: values that satisfy the bigger than condition condition
        :param values_less: values that satisfy the less than condition condition
        :param other_values_less: values that do not satisfy the less than condition
        :return:
        '''
        self.other_values_equal = other_values_equal
        self.values_equal = values_equal
        self.other_values_less = other_values_less
        self.values_less = values_less
        self.other_values_bigger = other_values_bigger
        self.values_bigger = values_bigger
        self.all_values=all_values


def __str__(self):
        txt = ""
        txt += "*****equal_sentences****\n"
        for sent in self.equal_sentences: txt += sent + "\n"
        txt += "*****not_equal_sentences****\n"
        for sent in self.equal_sentences: txt += sent + "\n"
        txt += "*****bigger_than_sentences****\n"
        for sent in self.bigger_than_sentences: txt += sent + "\n"
        txt += "*****not_bigger_than_sentences****\n"
        for sent in self.not_bigger_than_sentences: txt += sent + "\n"
        txt += "*****less_than_sentences****\n"
        for sent in self.less_than_sentences: txt += sent + "\n"
        txt += "*****not_less_than_sentences****\n"
        for sent in self.not_less_than_sentences: txt += sent + "\n"
        return txt


#for passing data to different generation functions
class GenerationInput:
    def __init__(self,senteces_with_indicies:list, equal_value:float, less_than_value:float, bigger_than_value:float, unit:str, conf:GenerationConf, keywords:str):
        '''

        :param senteces_with_indicies:
        :param equal_value:
        :param less_than_value:
        :param bigger_than_value:
        :param unit:
        :param keywords:
        :param conf: generation configuration with the number of samples and so one, type of class Conf
        '''
        self.senteces_with_indicies=senteces_with_indicies
        self.equal_value=equal_value
        self.less_than_value=less_than_value
        self.bigger_than_value=bigger_than_value
        self.unit=unit
        self.conf=conf
        self.keywords=keywords
        self.values=list(senteces_with_indicies.keys())
    def __str__(self):
            txt=self.keywords+" "+self.unit+" "
            txt=txt+" bigger: "+str(self.bigger_than_value)
            txt=txt+" less: "+str(self.less_than_value)
            txt=txt+" equal: "+str(self.equal_value)
            txt=txt+"\n list of values:"+str(self.values)
            return txt
