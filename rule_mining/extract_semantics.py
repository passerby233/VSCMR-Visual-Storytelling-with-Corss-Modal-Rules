"""
This script extract semantic for each photo stream in the dataset.
argv1 is the number threads, default is 4. Lager, faster.
"""
import numpy as np
import os
import sys
from multiprocessing import Process
from collections import defaultdict
from detector import VsDetector

import utils as ut
import config as cfg

split_per = 0.7
#data_path = os.path.join(proj_dir, 'VIST/rule_data_{}'.format(split_per))


def get_partial_dict(process_index, partial_story_ids, story_line, id2trans, detector):
    id2seman = defaultdict(list)

    for story_id in partial_story_ids:
        story = story_line[story_id]
        for i in range(5):
            img_id = story['flickr_id'][i]
            if img_id in id2trans:
                id2seman[story_id].append(detector.detect_id(id2trans[img_id][:10]))
            else:
                id2seman[story_id].append([])
        ut.printn("Process %d Processed story_%s" % (process_index, story_id))

    np.savez("{}semantic{}.npz".format(cfg.rule_data_path, process_index), dict=id2seman)
    ut.printn("Process %d finished detecting %d stories "
              % (process_index, len(partial_story_ids)))


def extract_semantics(detector, total_story_line, id2semantic_file, id2semanset_file, num_process):
    id2semanlsts = {}
    for mode in ['train', 'val', 'test']:
        trans_info_file = os.path.join(cfg.rule_data_path, "{}_trans_info.npz".format(mode))
        id2trans = ut.load_npz_dict(trans_info_file)['id2trans']

        story_line = total_story_line[mode]
        story_ids = story_line.keys()
        story_len = len(story_ids)
        ut.printn("%d stories in %s set" % (story_len, mode))

        spacing = np.linspace(0, story_len, num_process + 1).astype(np.int)
        ranges, processes = [], []

        for i in range(len(spacing) - 1):
            ranges.append((spacing[i], spacing[i + 1]))

        for process_index in range(len(ranges)):
            left, right = ranges[process_index]
            partial_story_ids = story_ids[left:right]
            args = (process_index, partial_story_ids, story_line, id2trans, detector)
            p = Process(target=get_partial_dict, args=args)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        id2semanlst = {}
        ut.printn("Merging %s set to result..." % mode)
        for i in range(num_process):
            partial_dict = ut.load_npz_dict("{}semantic{}.npz".format(cfg.rule_data_path, i))
            os.remove("{}semantic{}.npz".format(cfg.rule_data_path, i))
            id2semanlst.update(partial_dict)

        id2semanlsts[mode] = id2semanlst

    ut.save_json(id2semantic_file, id2semanlsts)
    ut.printn("Saved semanlst file to %s" % id2semantic_file)

    ut.printn("Converting semanlst to semanset")
    id2semansets = {}
    for mode in ['train', 'val', 'test']:
        id2semanset = {}
        tempset = set()

        for story_id, story_seman in id2semanlsts[mode].items():
            tempset.clear()
            for img_seman in story_seman:
                tempset.update(img_seman)
            id2semanset[story_id] = list(tempset)

        id2semansets[mode] = id2semanset

    ut.save_json(id2semanset_file, id2semansets)
    ut.printn("Saved semanlst file to %s" % id2semanset_file)


def test_semantic_result(id2words, total_story_line, id2semantic_file):
    mode = 'val'
    story_line = total_story_line[mode]
    trans_info_file = os.path.join(cfg.rule_data_path, "rule_data/{}_trans_info.npz".format(mode))
    id2trans = ut.load_npz_dict(trans_info_file)['id2trans']
    id2seman = ut.load_json(id2semantic_file)[mode]

    for story_id, story_seman in id2seman.items():
        origin_words = [id2words[str(idx-2048)] for img_id in story_line[story_id]['flickr_id']
                        for idx in id2trans[img_id][10:]]
        detected_words = [id2words[str(idx-2048)] for img_seman in story_seman
                          for idx in img_seman]

        print(origin_words)
        print(detected_words)
        print('--------------------------------------------------------------------')


def main():
    num_process = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    id2semanlst_file = os.path.join(cfg.rule_data_path, "semantic_list.json")
    id2semanset_file = os.path.join(cfg.rule_data_path, "semantic_set.json")
    #story_line_file = os.path.join(cfg.proj_dir, "VIST/story_line_{}.json".format(split_per))
    rule_file = os.path.join(cfg.rule_data_path, 'rules03_0.6.npz')

    # Load files
    id2words = ut.load_json(cfg.txtdata_file)['id2words']
    total_story_line = ut.load_json(cfg.story_line_file)

    # Create detector and extract semantic
    detector = VsDetector(rule_file, id2words)
    ut.printn("rules_num = %d, categories_kinds = %d" % (detector.rules_num, detector.elem_kinds))
    extract_semantics(detector, total_story_line, id2semanlst_file, id2semanset_file, num_process)

    #test_semantic_result(id2words, total_story_line, id2semanlst_file)


if __name__ == '__main__':
    main()

