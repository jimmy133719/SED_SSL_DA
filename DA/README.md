# Domain Adaptation
A sound event detection system designed for DCASE 2020 task 4, which consists of large amount of weak label and unlabel audio clips.

This work is based on the baseline of DCASE task 4 (https://github.com/turpaultn/dcase20_task4/tree/public_branch/baseline). 

**Note:** Check if the baseline code works before using our code.

-------------------------------
### Two-stage domain adaptation
#### First stage: pretrain the model with labeled data
```
python main.py -stage=pretrain
```
#### Second stage: Do domain adaptation on the pretrained model
Load the pretrained model as initialization (Copy and Rename the model as baseline_epoch_0), and assign start_epoch = 1 in line 525

Next,

```
python main.py -stage=adaptation -level=frame
```
Switch to clip-level domain adaptation --> -level=clip

**Note:**
If you want to change below argument, you need to restart from first stage

-mt: use the mean-teacher approach 

-ISP: use the proposed strategies

-fpn: switch the backbone model to FP-CRNN

Ex. Use all the above settings
```
python main.py -stage=pretrain -level=frame -mt -ISP -fpn

python main.py -stage=adaptation -level=frame -mt -ISP -fpn
```

#### Evaluate well-trained model
Choose a well-trained model (Ex. ./stored_data/ADDA_with_synthetic_fpn_clipD/model/baseline_best) and evaluate

```
python TestModel.py -m=model_path -g=../dataset/metadata/validation/validation.tsv -pd -fpn 
```
**Note:** 
-pd: use label prediction, **necessary**.
-sf=embedded feature path: Directory path for saving embedded feature

-------------------------------
### Embedded feature analysis
Analyze the domain invariance of adapted models 

You need to extract embedded features from both real and synthetic data first, ex.

for real data,
```
python TestModel.py -m=./stored_data/ADDA_with_synthetic_fpn_clipD/model/baseline_best -g=../dataset/metadata/validation/validation.tsv -pd -fpn -sf=embedded_feature/ADDA_with_synthetic_fpn_clipD
```

for synthetic data,
```
python TestModel.py -m=./stored_data/ADDA_with_synthetic_fpn_clipD/model/baseline_best -g=../dataset/metadata/train/synthetic20/soundscapes.tsv -pd -fpn -sf=embedded_feature/ADDA_with_synthetic_fpn_clipD
```

Next,

do analysis on frame-level features
```
python frameDA_analysis.py
```

or do analysis on clip-level features
```
python clipDA_analysis.py
```