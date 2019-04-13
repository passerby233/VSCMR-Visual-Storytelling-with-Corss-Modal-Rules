"""
Must be run in python3
"""
from __future__ import division

from collections import defaultdict
import itertools
import os
import argparse

import utils as ut
import config as cfg


def generate_association_rules(patterns, confidence_threshold):
    """
    Given a set of frequent itemsets, return a dict
    of association rules in the form
    {(left): ((right), confidence)}
    """
    rules = {}
    for itemset in patterns.keys():
        # print "itemset in
        # patterns.keys",itemset,"patterns[itemset]",patterns[itemset]
        upper_support = patterns[itemset]

        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))
                if antecedent in patterns:
                    lower_support = patterns[antecedent]
                    confidence = float(upper_support) / lower_support

                    if confidence >= confidence_threshold:
                        rule1 = (consequent, confidence)
                        rule1 = list(rule1)
                        if antecedent in rules:
                            rules[antecedent].append(rule1)
                        else:
                            rules[antecedent] = rule1

    return rules


def adaptive_generate_association_rules(patterns, confidence_threshold):
    """
    Given a set of frequent itemsets, return a dictof association rules
    in the form {(left): (right)}
    It has a check with 2048 thus will only retain multimodal rules.
    """
    missed = 0
    rules = defaultdict(set)
    for setn, support in patterns.items():
        if len(setn) > 1:
            itemset = list(setn)  # the itemset I with n element

            for i in range(len(itemset)-1, -1, -1):
                # the last pos is the inference item i for I->i
                # every elem go to the last once, the itemset remains sorted
                itemset[i], itemset[-1] = itemset[-1], itemset[i]
                setn_1 = tuple(itemset[:-1])

                if max(itemset[:-1]) < 2048 <= itemset[-1]:
                    if setn_1 in patterns:
                        confidence = patterns[setn] / patterns[setn_1]
                        if confidence >= confidence_threshold:
                            rules[setn_1].add(itemset[-1])
                    else:
                        missed += 1
                        print("missed", setn_1)
    print('%d freq missed.' % missed)

    return rules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minsupc', type=int, default=3)
    parser.add_argument('--minconf', type=float, default=0.6)
    opt = parser.parse_args()

    pattern_file = os.path.join(cfg.rule_data_path, 'patterns{}.npz'.format(str(opt.minsupc)))
    rule_file = os.path.join(cfg.rule_data_path, 'rules{}_{}.npz'.format(str(opt.minsupc).zfill(2), str(opt.minconf)))
    conf_rule_file = os.path.join(cfg.rule_data_path, 'conf_rules{}_{}.npz'.format(str(opt.minsupc), str(opt.minconf)))

    patterns = ut.load_npz_dict(pattern_file)
    #print(len(patterns))
    ut.printn("Got freq_tuples, minging the rules...")

    rules = adaptive_generate_association_rules(patterns, opt.minconf)
    #rules = generate_association_rules(patterns, opt.minconf)
    #print(len(rules))

    ut.save_dict_npz(rule_file, rules)
    ut.printn("Rules saved to %s" % rule_file)


if __name__ == '__main__':
    main()
