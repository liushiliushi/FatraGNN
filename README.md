# FatraGNN

This repository (without datasets due to space limit) is the official implementation of Fairness Learning on Graphs under Distribution Shifts. 


The [complete repository](https://anonymous.4open.science/r/FatraGNN-118F) with all the datasets is preferred.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset

Datasets can be found in https://anonymous.4open.science/r/FatraGNN-118F.


## Training

To train FatraGNN on the five datasets, run these commands:

```train
python fatragnn.py --gpu 3 --ood 2 --dataset=bail --encoder=GCN --inid=_B0 --outid=all --times=config1
python fatragnn.py --gpu 3 --ood 2 --dataset=credit --encoder=GCN --inid=_C0 --outid=all --times=config1
python fatragnn.py --gpu 3 --ood 2 --dataset=pokec --encoder=GCN --inid=_z --outid=all --times=config1
```

Train on sync-B1s and sync-B2s

```train
python fatragnn.py --gpu 3 --ood 1 --dataset=bail --encoder=GCN --inid=_B0 --outid=_md0 --times=config2
python fatragnn.py --gpu 3 --ood 1 --dataset=bail --encoder=GCN --inid=_B0 --outid=_md3 --times=config3
```


