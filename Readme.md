# Visual Storytelling with Corss-Modal Rules
Code for: Informative Visual Storytelling with Cross-modal Rules
In procedings of ACM Multimedia 2019
@inproceedings{li2019informative,
  title={Informative Visual Storytelling with Cross-modal Rules},
  author={Li, Jiacheng and Shi, Haizhou and Tang, Siliang and Wu, Fei and Zhuang, Yueting},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={2314--2322},
  year={2019},
  organization={ACM}
}
The storytelling code is adpated from https://github.com/eric-xw/AREL
![](https://github.com/passerby233/VSCMR/blob/master/illustration.pdf)


## Prerequisites 
- Python 2.7
- Python 3.x
- PyTorch 0.3
- TensorFlow (optional, only using the fantastic tensorboard)
- cuda & cudnn

## Usage
### 1. Setup
- clone this code
- Download the dataset from http://www.visionandlanguage.net/workshop2018/<br>
Create a folder 'vist' and unzip the dataset to the folder<br>
The dataset folder should follow the structure:<br>
```
vist
　|--annotatioons
　　|--dii
　　|--sis
  |--images
    |--test
    |--train
    |--val
```
- Get resnet features from http://nlp.cs.ucsb.edu/data/VIST_resnet_features.zip <br>
The features should be unzipped to folder 'VIST'<br>
or process it from dataset by 'extract_feature_max.py'<br>
- Change values in the 'rule_ming/config.py'

### 2. Cross-Modal Rule Mining
In The following decription, [] denotes option
- Enter the folder rule_ming:
``` 
cd rule_mining
```
- Create multi-modal transactions:
```
python2 create_transactions.py [mode]
```
mode can be 'train', 'val', 'test'; default is 'train' if not designated.
- Find the frequent itemset:
```
python3 fpgrowth_py3.py [--minsupc 3]
```
- Get Cross-Modal Rules:
```
python3 get_rules.py [--minsupc 3,--conf 0.6]
```
- Extract semantic concepts with CMR:
```
python2 extract_semantics.py [4]
```
The option is the number of threads; Larger, faster; Please set according to the number of your CPU cores.

### 3. Visual Storytelling
These scripts are adapted from AREL<br>, we add a attention mechanism to attend the inferred concepts.
To train the VIST model:
```
python2 train.py --beam_size 3 [--id model_name]
```
To test the performance on metrics:
```
python2 train.py --beam_size 3 --option test --start_from_model data/save/XE/model.pth [--id score_save_path]
```
