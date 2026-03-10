# Synapse_Baseline_allbackbones

Baseline training pipelines for **multi-class medical image segmentation** on the **Synapse** dataset using several backbone families, including **UNet**, **UNeXt**, and a **ResNet-based encoder-decoder**. The repository is organized around training scripts, evaluating, dataset utilities, losses, metrics, plotting utilities, and experiment outputs.

---

## Overview

This project provides a set of baseline experiments for **2D slice-based segmentation** on the Synapse dataset.  
The main idea is to train different segmentation backbones under a similar pipeline and compare their performance using common medical segmentation metrics such as:

- **Dice score**
- **Jaccard / IoU**
- **HD95**

The current repository includes dedicated training scripts for:

- **UNet**
- **UNeXt**
- **ResNet**
- **TransUnet** 
- **EfficientUnet**
- **HiFormer** 
- **DeepLab**

and contains supporting folders for datasets, losses, metrics, networks, plots, and experiment outputs.

---

## Repository Structure

```bash
Synapse_Baseline_allbackbones/
├── Datasets/                         # Dataset loading and preprocessing utilities
├── losses/                           # Loss functions
├── metrics/                          # Evaluation metrics
├── networks/                         # Model definitions
├── plot/                             # Plotting scripts for training & testing history / results
├── experiments/                      # Saved experiment outputs (not all commited (large files))
├── main_Unet.py                      # UNet evaluating script
├── main_Unext.py                     # UNeXt evaluating script
├── main_DeepLab.py                   # DeepLab-based segmentation evaluating script
├── main_Resnet.py                    # ResNet-based segmentation evaluating script
├── main_TransUnet.py                 # TransUnet-based segmentation evaluating script
├── main_HiFormer.py                  # HiFormer-based segmentation evaluating script
├── main_EfficientUnet.py             # EfficientUnet-based segmentation evaluating script
├── Old_code/                         # Older experiments / archived code
├── results.txt                       # results recap 
└── __init__.py
```

## Features 
- Multi-class segmentation on Synapse
- Unified training pipeline across several backbones
- Automatic class-weight estimation for CrossEntropy loss
- Combined training loss
- CSV logging of training and test history
- Best and last checkpoint saving
- Plot utilities for result visualization

## Future Improvements
- Add a full requirements.txt
- Add pretrained checkpoint download links 
- Add qualitative visualization examples
- Add augmentation benchmarks
- Add support for more backbones
- Add inference / prediction scripts
- Add 3D evaluation utilities

