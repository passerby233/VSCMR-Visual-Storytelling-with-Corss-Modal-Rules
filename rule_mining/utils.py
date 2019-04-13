from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import string
import json
from datetime import datetime
from matplotlib import pyplot as plt
import re

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')  # Split on any non-alphanumeric character


def split_sentence(sentence):
    """ break sentence into a list of words and punctuation """
    toks = []
    for word in [s.strip().lower() for s in SENTENCE_SPLIT_REGEX.split(
            sentence.strip()) if len(s.strip()) > 0]:
        # Break up any words containing punctuation only, e.g. '!?', unless it
        # is multiple full stops e.g. '..'
        if all(c in string.punctuation for c in word) and not all(
                        c in '.' for c in word):
            toks += list(word)
        else:
            toks.append(word)
    if toks[-1] != '.':
        return toks
    return toks[:-1]


def save_dict_npz(save_path, dictdata):
    np.savez_compressed(save_path, dict=dictdata)


def load_npz_dict(npz_file):
    dictdata = np.load(npz_file, encoding='bytes')['dict'][()]
    return dictdata


def save_json(save_path, dictdata):
    with open(save_path, 'wb') as f:
        json.dump(dictdata, f)


def load_json(save_path):
    with open(save_path, 'rb') as f:
        dictdata = json.load(f)
    return dictdata


def printn(*args):
    print("%s:" % str(datetime.now())[:19], end="")
    print(*args)


def plt_texts(strs, bias, shift=200, fontsize=15):
    for idx, line in enumerate(strs):
        plt.text(0, bias + idx * shift, line, fontsize=fontsize)


