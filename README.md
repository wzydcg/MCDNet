# MCDNet: Morphological-Conditional Dual-view Fusion for 3D Tubular Structure Segmentation

[![stars - MCDNet](https://img.shields.io/github/stars/wzydcg/MCDNet?style=social)](https://github.com/wzydcg/MCDNet)
[![forks - MCDNet](https://img.shields.io/github/forks/wzydcg/MCDNet?style=social)](https://github.com/wzydcg/MCDNet)
![language](https://img.shields.io/github/languages/top/wzydcg/MCDNet?color=lightgrey)
![license](https://img.shields.io/github/license/wzydcg/MCDNet)
---

## Approach

![MCDNet.png](picture/MCDNet.png)

## Morphological-Conditional Convolution
![MCConv.png](picture/MCConv.png)

## Dataset

Download the BCIC IV-2A and IV-2B dataset from [here](https://www.bbci.de/competition/iv/index.html).

Download the ZuCo-TSR dataset from [here](https://osf.io/q3zws/).

MNRED dataset will be released in the near future.

## Preprocessing Data

Each dataset corresponds to a dataloader and a preprocessing scripts. 
For example, ```smr_preprocess()``` in ```data/smr.py``` process BCIC IV-2A to ```SMR128.npy``` 

## Training

### Default Scripts
All default hyperparameters among these models are tuned for RAOS datasets.

Wandb is needed if visualization of training parameters is wanted

### Customized Execution

run script like this:
```bash
python main.py \
--model Our_UNet \
--dataset RAOS \
--batch_size 4 \
--num_epochs 200 \
--learning_rate 1e-4 \
--dropout 0.1 \
--do_train \
--do_evaluate
```

## Dependencies
- python==3.12
- opencv-python==4.7.0.68
- einops
- nilearn==0.10.4
- scikit-learn==1.3.2
- scipy
- torch==2.3.0
- pydicom==2.4.4
- pandas==1.5.3
- nibabel==5.2.1
- wandb

## Citation

```
@ARTICLE{
  author={Wang, Zhiyan and Wang, Changjian and Xu, Kele and Tang, Zhongshun and Zhuang, Yan and Zou, Jiani and Liu, Fangyi},
  journal={}, 
  title={MCDNet: Morphological-Conditional Dual-view Fusion for 3D Tubular Structure Segmentation}, 
  year={2025},
  volume={},
  number={},
  pages={},
  keywords={Tubular Structure Segmentation;Conditional Convolution;Dual-view Architecture},
  doi={}}

```

## Contact Us

If you are interested to leave a message, please feel free to send any email to us at ```wangzhiyan24@nudt.edu.cn```
