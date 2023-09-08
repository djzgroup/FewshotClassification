# Title

## Description

This repository contains the code for our paper: **title**

## Environment setup

This project is built upon the following environment:

* Python 3.8
* CUDA 11.3
* PyTorch 1.12.0

## Dataset

* [ModelNet40](https://modelnet.cs.princeton.edu/)
* [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)
  
  ## Train
  
  Train a model on the ModelNet40 dataset by
  
  ```
  python main.py --phase train --dataset modelnet40 --data_path ./dataset/modelnet40_normal_resampled
  ```

Train a model on the ScanObjectNN dataset by

```
python main.py --phase train --dataset scanobjectn --data_path ./dataset/ScanObjectNN/main_txt
```

## Evaluate

Evaluate a model on the ModelNet40 dataset by

```
python main.py --phase eval --dataset modelnet40 --model_path ./log/modelnet40/checkpoints/5_way_1_shot.pth
```

Evaluate a model on the ScanObjectNN dataset by

```
python main.py --phase eval --dataset scanobjectnn --model_path ./log/scanobjectn/S_0/checkpoints/5_way_1_shot.pth
```

## Acknowledgment

Our implementation is mainly based on the following codebase. We gratefully thank the authors for their wonderful works.

[Cross-Modality-Feature-Fusion-Network](https://github.com/LexieYang/Cross-Modality-Feature-Fusion-Network)
