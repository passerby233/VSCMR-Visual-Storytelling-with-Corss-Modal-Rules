from __future__ import print_function

from collections import Counter
import utils as ut
from matplotlib import pyplot as plt


class VsDetector(object):
    def __init__(self, rules_file, id2words):
        self.imrange = 2048
        self.id2words = id2words
        self.rules = ut.load_npz_dict(rules_file)

        self.pattern_to_set = {}
        for pattern in self.rules.keys():
            self.pattern_to_set[pattern] = set(pattern)

        self.rules_num = len(self.rules)

        counter = Counter()
        for pattern, elem in self.rules.items():
            elem_set = [self.id2words[str(int(idx)-self.imrange)] for idx in elem]
            counter.update(elem_set)
        # The cartegories list of the concepts
        self.categories = counter.keys()
        # The number of the categories
        self.elem_kinds = len(counter)

    def show_rules(self, ):
        counter = Counter()

        print("Total rules:", self.rules_num)
        print("kinds of elements:", self.elem_kinds)
        for pattern, elem in self.rules.items():
            elem_set = [self.id2words[str(int(idx) - self.imrange)] for idx in elem]
            counter.update(elem_set)
            # print(pattern, elem_set)

        print(counter)

    def detect(self, image_trans):
        """
        :param image_trans: imgae transaction, list of 10 int.
        :return: a list of words inferred by the rules
        """
        ve_id_set = set()

        for rule, elem_set in self.rules.items():
            if self.pattern_to_set[rule].issubset(image_trans):
                ve_id_set |= elem_set
        ve_words = [self.id2words[str(int(idx)-self.imrange)] for idx in ve_id_set]
        return ve_words

    def detect_id(self, image_trans):
        """
        :param image_trans: imgae transaction, list of 10 int.
        :return: a list of word id inferred by the rules
        """
        ve_id_set = set()

        for rule, elem_set in self.rules.items():
            if self.pattern_to_set[rule].issubset(image_trans):
                ve_id_set |= elem_set
        ve = list(ve_id_set)
        return ve
