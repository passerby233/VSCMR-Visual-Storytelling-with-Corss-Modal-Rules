"""
This script is for multi-modal transaction creation. The argv is the mode for dataset.
'train','val' and 'test' are valid. The default value is 'train'.
Takes about 3 min.
"""
import numpy as np
import os
import nltk
import csv
import sys
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

import utils as ut
import config as cfg
# Please uncomment the following two line when running at first time
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')


im_keep = 10


def get_txtdata():
    story_line = ut.load_json(cfg.story_line_file)
    txtdata = {'words2id': story_line['words2id'],
               'id2words': story_line['id2words'],
               'words': story_line['words2id'].keys(),
               'word_tag': {}}

    tagged_words = nltk.corpus.brown.tagged_words()
    for word, tag in tagged_words:
        txtdata['word_tag'][word] = tag

    def _get_tag(word):
        if word in txtdata['word_tag']:
            wtag = txtdata['word_tag'][word]
        else:
            wtag = nltk.pos_tag([word])[0][1]
        return wtag

    # get diasbled_list
    allowed = ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS',
               'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    categories = defaultdict(list)
    txtdata['disabled_list'] = ['ed', 'n\'t', 'get']

    for word in txtdata['words']:
        tag = _get_tag(word)
        if tag not in allowed or len(word) <= 1:
            txtdata['disabled_list'].append(word)
        categories[tag].append(word)
    print("Found %d non semantic words." % len(txtdata['disabled_list']))

    # get word_count
    txtdata['word_count'] = defaultdict(int)
    for mode in ['train', 'val', 'test']:
        annofile = os.path.join(cfg.dataset_path, "annotations/sis/{}.story-in-sequence.json".format(mode))
        annotations = ut.load_json(annofile)['annotations']

        for item in annotations:
            annotation = item[0]
            text_list = ut.split_sentence(annotation['text'])
            for word in text_list:
                txtdata['word_count'][word] += 1

    # save txtdata
    ut.save_json(cfg.txtdata_file, txtdata)


class TextProcesser(object):
    def __init__(self):
        if not os.path.exists(cfg.txtdata_file):
            get_txtdata()
        txtdata = ut.load_json(cfg.txtdata_file)

        self.words2id = txtdata['words2id']
        self.id2words = txtdata['id2words']
        self.words = txtdata['words']
        self.word_tag = txtdata['word_tag']
        self.disabled_list = txtdata['disabled_list']
        self.word_count = txtdata['word_count']
        self.wnl = WordNetLemmatizer()

    def get_tag(self, word):
        if word in self.word_tag:
            tag = self.word_tag[word]
        else:
            tag = nltk.pos_tag([word])[0][1]
        return tag

    def lemmatize(self, word):
        tag = self.get_tag(word)
        if tag.startswith('NN'):
            lemm = self.wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            lemm = self.wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            lemm = self.wnl.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            lemm = self.wnl.lemmatize(word, pos='r')
        else:
            lemm = word

        return lemm


def create_transactions(mode='train'):
    #split_per = 0.1
    rule_folder = "VIST/rule_data"
    annofile = os.path.join(cfg.dataset_path, "annotations/sis/{}.story-in-sequence.json".format(mode))
    trans_file = os.path.join(cfg.rule_data_path,  "{}_transactions.txt".format(mode))
    trans_info_file = os.path.join(cfg.rule_data_path,  "{}_trans_info.npz".format(mode))
    #mining_flickr_list_file = os.path.join(cfg.proj_dir, "VIST/mining_flickr_list_{}.json".format(split_per))

    # open annotation file and list
    annotations = ut.load_json(annofile)['annotations']
    ut.printn("%d annotations in total" % len(annotations))
    #mining_flickr_list = ut.load_json(mining_flickr_list_file)
    #ut.printn("%d images for transaction creation in total" % len(mining_flickr_list))

    # process text
    tp = TextProcesser()
    text_elem = defaultdict(set)

    # create img_id -> text
    for item in annotations:
        annotation = item[0]
        img_id = annotation['photo_flickr_id']
        text_list = ut.split_sentence(annotation['text'])

        for word in text_list:
            lemm = tp.lemmatize(word)
            if word in tp.disabled_list or lemm in tp.disabled_list:
                continue
            elif lemm in tp.words:
                text_elem[img_id].add(lemm)
            elif word in tp.words:
                text_elem[img_id].add(word)
            else:
                continue

    # create transactions
    id2trans = {}
    for img_id, words_set in text_elem.items():
        feature_file = os.path.join(cfg.feature_dir, mode, "{}.npy".format(img_id))
        if os.path.exists(feature_file):
            im_feature = np.load(feature_file)
        else:
            continue
        im_trans = np.sort(np.argsort(-im_feature)[:im_keep])

        words_list = sorted(list(words_set), key=lambda x: tp.word_count[x], reverse=True)
        txt_trans = np.sort(np.array([tp.words2id[word] for word in words_list], dtype=np.int64) + 2048)

        id2trans[img_id] = np.concatenate((im_trans, txt_trans), axis=0)

    # write transactions
    trans_len = []
    with open(trans_file, 'w') as f:
        ut.printn("Writing %d transactions to %s" % (len(id2trans), trans_file))
        writer = csv.writer(f, delimiter='\t')
        for img_id, trans in id2trans.items():
            writer.writerow(trans)
            trans_len.append(len(trans))
    avg_len = np.mean(trans_len)
    ut.printn("Done. Average len %f" % avg_len)

    # save transaction info
    trans_info = {'id2trans': id2trans,
                  'trans_len': trans_len,
                  'avg_len': avg_len}
    np.savez(trans_info_file, dict=trans_info)

    """
    # output partial transactions
    for img_id, words in text_elem.items()[:5]:
        print(img_id, words)

    for img_id, trans in id2trans.items()[:5]:
        print(img_id, trans[:10], [tp.id2words[str(idx-2048)] for idx in trans[10:]])
    """


if __name__ == '__main__':
    if len(sys.argv) > 1:
        create_transactions(sys.argv[1])
    else:
        create_transactions()
