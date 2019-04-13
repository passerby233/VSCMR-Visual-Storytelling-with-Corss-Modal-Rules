"""
This script calculates the inter and intra repetition of generated stories.
The repetition is defined in the paper:
@article{DBLP:journals/corr/abs-1811-05701,
  author    = {Lili Yao and
               Nanyun Peng and
               Ralph M. Weischedel and
               Kevin Knight and
               Dongyan Zhao and
               Rui Yan},
  title     = {Plan-And-Write: Towards Better Automatic Storytelling},
  journal   = {CoRR},
  volume    = {abs/1811.05701},
  year      = {2018},
  url       = {http://arxiv.org/abs/1811.05701},
  archivePrefix = {arXiv},
  eprint    = {1811.05701},
  timestamp = {Wed, 06 Mar 2019 07:05:24 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1811-05701},
  bibsource = {dblp computer science bibliography, https://dblp.org}
"""

from __future__ import division
from rule_mining import utils as ut
import nltk

"""
ref_file_list = ["/home/lijiacheng/AREL-master/data/save/XE/XE_b3.json",
                 "/home/lijiacheng/AREL-master/data/save/IRL-XE-ss-init/challenge_b3.json",
                 "/home/lijiacheng/AREL/data/save/seman_XE/seman_XE55_b3.json",
                 "/home/lijiacheng/AREL/data/save/default/XE_b3.json"]
"""
ref_file_list = ["/home/lijiacheng/AREL-master/data/save/IRL-XE-ss-init/prediction_test.json",
                 "/home/lijiacheng/AREL/data/save/seman_XE/prediction_test.json",
                 "/home/lijiacheng/AREL-master/data/save/XE_score/prediction_test.json"]
txtdata_file = "/home/lijiacheng/AREL/VIST/rule_data/txtdata.json"


txtdata = ut.load_json(txtdata_file)
words2id = txtdata['words2id']
id2words = txtdata['id2words']


def split(text):
    """
    Split each sentence in the prediction to a list of word.
    '[','male',']' has been taken into consideration. It will be concatenated to '[male]'
    """
    split_text = [[nltk.tokenize.word_tokenize(sentence) for sentence in story] for story in text]
    for story in split_text:
        story_list = []
        for words in story:
            for index, word in enumerate(words):
                if word == '[':
                    words[index] = ''.join(words[index:index+3])
                    del words[index+1]
                    del words[index+1]
    return split_text


def tokenize(text):
    """
    Similar to split. The sentence will be split to a list of word id.
    """
    split_text = [[nltk.tokenize.word_tokenize(sentence) for sentence in story] for story in text]
    prediction_token = []
    for story in split_text:
        story_list = []
        for words in story:
            for index, word in enumerate(words):
                if word == '[':
                    words[index] = ''.join(words[index:index+3])
                    del words[index+1]
                    del words[index+1]
            story_list.append([words2id[word] for word in words])
        prediction_token.append(story_list)
    return prediction_token


def inter_rep(prediction, m=5, n=4):
    """
    calculate the inter repetition
    :param prediction:list of stories, each story contain m sentences, each story is a list of split word.
    :param m: number of sentences in a story
    :param n: n_gram
    :return:
    """
    distinct, T_all, inter_score = [], [], []
    for i in xrange(m):
        T_all.append(0)
        distinct.append(set())
        for story in prediction:
            if i < len(story):
                sentence_i = story[i]
                for k in xrange(len(sentence_i) - n + 1):
                    ngram = tuple(sentence_i[k:k + n])
                    distinct[i].add(ngram)
                    T_all[i] += 1
        T = len(distinct[i])
        inter_score.append(1-T/T_all[i])

    distinct_all = set()
    for i in xrange(m):
        distinct_all = distinct_all | distinct[i]
    inter_score_agg = 1-len(distinct_all) / sum(T_all)
    return inter_score, inter_score_agg


def intra_rep(prediction, m=5, n=4):
    """
    calculate the intra repetition
    :param prediction:list of stories, each story contain m sentences, each story is a list of split word.
    :param m: number of sentences in a story
    :param n: n_gram
    :return:
    """
    distinct = [set(), set(), set(), set(), set()]
    intra_scores = [[], [], [], [], []]
    for story in prediction:
        for s in distinct:
            s.clear()
        for i in xrange(m):
            if i < len(story):
                sentence_i = story[i]
                for k in xrange(len(sentence_i) - n + 1):
                    ngram = tuple(sentence_i[k:k + n])
                    distinct[i].add(ngram)
        for i in xrange(m):
            if i == 0:
                intra_scores[i].append(0)
            else:
                if len(distinct[i]) == 0:
                    intra_scores[i].append(0)
                else:
                    intra_scores[i].append(
                        sum([len(distinct[k] & distinct[i]) for k in xrange(i)]) / (i * len(distinct[i])))

    intra_score = [sum(intra_score_i) / len(prediction) for intra_score_i in intra_scores]
    intra_score_agg = sum(intra_score) / m
    return intra_score, intra_score_agg


def rep_eval(ref_file):
    print "evaluating for %s" % ref_file
    #ref = ut.load_json(ref_file)["output_stories"]
    #text = [item["story_text_normalized"].split('.')[:-1] for item in ref]
    ref = ut.load_json(ref_file)
    text = [sentence[0].split('.')[:-1] for sentence in ref.values()]
    split_text = split(text)
    inter_score, inter_score_agg = inter_rep(split_text, n=5)
    intra_score, intra_score_agg = intra_rep(split_text, n=5)
    print inter_score, inter_score_agg
    print intra_score, intra_score_agg


def main():
    for ref_file in ref_file_list:
        rep_eval(ref_file)


if __name__ == '__main__':
    main()

