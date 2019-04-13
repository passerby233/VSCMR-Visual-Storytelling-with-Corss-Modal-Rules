# vist_eval_scripts
__author__ = Licheng
lift from https://github.com/lichengunc/vist_eval

## Usage:
### stimgids_eval.py
instant a class Story_in_Sequence with vist dataset path
```
  sis = Story_in_Sequence(
      '/your_dir/vist/images/',
      '/your_dir/vist/annotations/')
```
instant StimgidsEvaluator with Story_in_Sequence and aplly evaluate:
```
  seval = StimgidsEvaluator(sis)
  seval.evaluate(preds)
```
- The preds is a list of dict, each dict is in the form [{'stimgids': stimgids, 'pred_story_str': pred_story_str}]<br>
- stimgids is a str, the 5 photo_flickr_id joined by '_'.<br>
- pred_story_str is a str, the 5 sentences joined by '_'.<br>

### album_eval.py
instant a class and evalute
```
eval = AlbumEvaluator()
eval.evaluate(reference, predictions)
```
- the references and predictions are a dict of {album_id:prediction}
- album_id is a str, corresponding to album_id; prediction is a str, 5 sentences joined by '_'<br>
- The result is saved in eval.eval_overall

```
@article{DBLP:journals/corr/abs-1708-02977,
  author    = {Licheng Yu and
               Mohit Bansal and
               Tamara L. Berg},
  title     = {Hierarchically-Attentive {RNN} for Album Summarization and Storytelling},
  journal   = {CoRR},
  volume    = {abs/1708.02977},
  year      = {2017},
  url       = {http://arxiv.org/abs/1708.02977},
  archivePrefix = {arXiv},
  eprint    = {1708.02977},
  timestamp = {Mon, 13 Aug 2018 16:47:29 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1708-02977},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
