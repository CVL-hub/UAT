
# **`[TIP2025]Uncertainty-Aware Transformer for Referring Camouflaged Object Detection`**

Authors: Ranwan Wu, [Tian-Zhu Xiang](https://scholar.google.com/citations?user=5uQEWX0AAAAJ&hl=en&oi=ao), [Guo-Sen Xie](https://scholar.google.com/citations?user=LKaWa9gAAAAJ&hl=en&oi=ao), [Rongrong Gao](https://scholar.google.com/citations?user=MwdwZ_kAAAAJ&hl=en&oi=ao), [Xiangbo Shu](https://scholar.google.com/citations?user=FQfcm5oAAAAJ&hl=en&oi=ao), [Fang Zhao](https://scholar.google.com/citations?hl=en&user=4C7mvOwAAAAJ) and [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en&oi=ao)


Welcome to the official PyTorch implementation repository of our paper **Uncertainty-Aware Transformer for Referring Camouflaged Object Detection**, accepted to IEEE TIP 2025.

# Framework
![image](figs/UAT.png)  
Figure.1 Architecture of uncertainty-aware transformer (UAT) for Ref-COD. UAT takes a camouflaged image and several referring images as input, respectively. Next, basic feature extraction on these images is performed. Then, the extracted features are fed into referring feature aggregation (RFA), cross-attention encoder (CAE), and transformer probabilistic decoder (TPD) to integrate visual reference into camouflage feature, aggregate multi-layer camouflage features, and model the dependencies between patches/tokens via Bayesian uncertainty learning, respectively. Finally, the predictions from all four stages are supervised by $L_{total}$ collaboratively.


# Requirements
Python v3.6, Pytorch 0.4.0+, Cuda 10.0, TensorboardX 2.0, opencv-python

# Get Start
### 1. Data Preparation
- Please visiting [RefCOD](https://github.com/zhangxuying1004/RefCOD) for training and testing data. Thanks for their contributions.

### 2. Training
- Download the training and testing dataset, and place them in the *dataset* floder.
- Download the pre-trained weights of [pvtv2](https://pan.baidu.com/s/1etvyFSv9nFrWKHxwHcSHJA?pwd=2025)[code:2025] on Baidu Netdisk, and place them in the *pvt_weights* floder.
- Run `python train.py` to train the model.
- You can also download the our pre-trained [UAT.pth](https://pan.baidu.com/s/1BiDFrYWoAbGmAdC9MTcCUQ?pwd=2025) with access code 2025 on Baidu Netdisk directly.

### 3. Inference
- After training, run `python infer.py` to generate the prediction maps of UAT.
- You can also download our prediction maps [UAT-Maps](https://pan.baidu.com/s/1HvcGqGkATsGnPXpv7NwVtw?pwd=2025)[code:2025] on Baidu Netdisk.

### 4. Testing
- After training, run `python test.py` to evaluate the performance of UAT.

### 5. Results
* **Qualitative comparison**

![image](figs/qulities_results.png)  
Table.1 Quantitative comparison with some SOTA models on referring camouflaged bbject detection benchmark datasets. 


# Acknowlegement
This repo is mainly built based on [R2CNet](https://github.com/zhangxuying1004/RefCOD). Thanks for the great work! If you have any technical questions, feel free to contact [wuranwan2020@sina.com](wuranwan2020@sina.com). If our work inspires your research, please cite it and start this project. We appreciate your support!
