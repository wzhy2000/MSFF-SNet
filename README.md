# MSFF-SNet

This paper proposes MSFF-SNet, which directly work with grasp poses as learning targets, while simultaneously recognizing the objects to be sorted. Specifically, multi-scale features from different layers of a ResNet50 are first extracted and intra-scale features are fused using a self-attention mechanism, emphasizing the global contextual information. The features of multiple scales are then fused through a cross-scale feature fusion module. Finally, a multi-task detection head is used to directly make pixel-level predictions of grasping poses and object categories. MSFF-SNet achieved a grasp accuracy of 99.62% on the Cornell Grasping Dataset and a mean Average Precision with Grasp of 79.52% on the Visual Manipulation Relationship Dataset, achieving the state-of-the-art performance. Besides, it achieved the highest inference speed on the Visual Manipulation Relationship Dataset.



**MSFF-SNet: An End-to-end Object Sorting Model with Multi-head Self-attention and Multi-scale Feature Fusion**

Xiaoya Fan, Jiaxiao Wang, Zilu Wang, Zhonghua Huang, Wenjie Yang, and Zhong Wang



## Installation

This code was developed with Python 3.9 on Ubuntu 20.04. The main Python requirements:

```
pytorch==1.2 or higher version
opencv-python
mmcv
numpy
```

## Datasets

1. Download and extract Cornell and VMRD Dataset

2. run `generate_grasp_mat.py`，convert `pcd*Label.txt` to `pcd*grasp.mat`, they represent the same label, but the format is different, which is convenient for MSFF-SNet to read.

3. Put all the samples of the Cornell  datasets in the same folder, and put train-test folder in the upper directory of the dataset, as follows

   ```
   D:\dataset\
   ├─cornell
   │  ├─pcd0100grasp.mat
   │  └─pcd0100r.png
   │  |
   |  └─pcd2000grasp.mat
   |  └─pcd2000r.png
   |
   ├─train-test
   │  ├─train-test-all
   │  ├─train-test-cornell
   │  └─train-test-mutil
   │  └─train-test-single
   |
   ├─other_files
   ```



## Training

Training is done by the `train_net.py` script.

Some basic examples:

```shell
python train_net.py --dataset-path <Path To Dataset>
```

Trained models are saved in `output/models` by default, with the validation score appended.



## Visualisation

visualisation of the trained networks are done using the `demo.py` script. 

Some output examples of MSFF-SNet is under the `demo\output`.



## Acknowledgement

Our work is inspired by **AFFGA-Net**, whose approach to adaptive feature fusion greatly motivated the design of our multi-scale fusion strategy. We appreciate the contributions of the researchers behind AFFGA-Net.



## How to cite

Xiaoya Fan, Jiaxiao Wang, Zilu Wang, Zhonghua Huang, Wenjie Yang, and Zhong Wang, MSFF-SNet: An End-to-end Object Sorting Model with Multi-head Self-attention and Multi-scale Feature Fusion, IJCNN 2025

