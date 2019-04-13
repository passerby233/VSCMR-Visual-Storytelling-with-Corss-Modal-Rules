import os

import utils as ut
from detector import VsDetector

proj_dir = "/home/lijiacheng/AREL/"
data_path = os.path.join(proj_dir, 'VIST/')
mode = 'train'

trans_info_file = os.path.join(data_path, "rule_data/{}_trans_info.npz".format(mode))
txtdata_file = os.path.join(data_path, "rule_data/txtdata.json")
story_line_file = os.path.join(data_path, "story_line.json")
id2trans = ut.load_npz_dict(trans_info_file)['id2trans']

id2words = ut.load_json(txtdata_file)['id2words']
story_line = ut.load_json(story_line_file)[mode]

path = "/home/lijiacheng/AREL/VIST/rule_data/"
basenames = ['rules10_0.6.npz', 'rules05_0.6.npz', 'rules03_0.6.npz', 'rules03_0.7.npz', 'rules03_0.8.npz']
for img_id in story_line['5248']['flickr_id']:
    print img_id,
print
for name in basenames:
    rule_file = path + name
    detector = VsDetector(rule_file, id2words)
    print rule_file
    print "rules_num = %d, categories_kinds = %d" % (detector.rules_num, detector.elem_kinds)

    origin_words = set()
    for img_id in story_line['5248']['flickr_id']:
        origin_words.update([id2words[str(idx - 2048)].encode('utf-8')
                             for idx in id2trans[img_id][10:]])

    semantic_list = []
    detected_words = set()

    for img_id in story_line['5248']['flickr_id']:
        im_trans = id2trans[img_id][:10]
        semantic_words = detector.detect(im_trans)
        semantic_words = [word.encode('utf-8') for word in semantic_words]
        semantic_list.append(semantic_words)
        detected_words.update(semantic_words)
    print('intersection')
    inter = detected_words & origin_words
    print(inter, len(inter))
    print('remainder')
    diff = detected_words - inter
    print(diff, len(diff))
