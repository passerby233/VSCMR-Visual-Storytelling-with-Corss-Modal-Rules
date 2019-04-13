# Cross-Modal Rules
- config.py | save the global value of data path.
- create_transactions.py | create cross-modal transactions with resnet features and text annotations.
- fpgrowth_py3.py | find frequent patterns in transactions 
- get_rules.py | get cross-modal rules from patterns
- detector.py | class VsDetector for concepts inference
- extract_semantics.py | extratic semantic concepts with rules by multi threads.
Adapated from https://github.com/vukk/amdm-fpgrowth-python/blob/master/fpgrowth.py
- the fpgrowth_py3.py and get_rules.py must be run in python3

### utils.py
#### function description
```
split_sentence(sentence)
```
- Split a sentence into a list of word, a little different from nltk.tokenize<br>

```
save_json(save_path, dictdata)
load_json(save_path)
```
- Quickly save and load dict to json file. Convenient for simple data type.

```
save_dict_npz(save_path, dictdata)
load_npz_dict(npz_file)
```
- Quickly save and load dict to npz file. Support dict containing numpy array.

```
printn(*args)
```
- Print content with time stamp.
