# STF-CrowdMamba

## STF-CrowdMamba: Spatio-Temporal Mamba and Spatial-Frequency Refinement for Video Crowd Counting

**基于时空 Mamba 和频域增强的视频人群计数**

Official PyTorch implementation of **STF-Mamba** for video crowd counting.

**Authors:** Zheng-Yan Wu and Miao-Gen Ling

**Affiliation:** School of Computer Science, Nanjing University of Information
Science & Technology, Nanjing 210044, China

## Overview

STF-Mamba is designed to model long-range spatio-temporal dependencies while
maintaining efficient computation and complementary feature representations.
The network contains two parallel branches:

- **Multi-Scale Spatio-Temporal Mamba Block (MS-STMB):** organizes consecutive
  frames as a joint temporal-height-width sequence, models long-range
  spatio-temporal relationships with bidirectional selective scans, and
  aggregates multi-scale crowd features through multiple paths.
- **Spatial-Frequency Refinement Block (SFRB):** separately refines the
  amplitude and phase spectra of frame-level features and reconstructs enhanced
  spatial features through the inverse Fourier transform.
- **Dual-branch density regression:** fuses the spatio-temporal and frequency
  representations to predict crowd density maps.

Experiments on five public video crowd-counting datasets demonstrate the
effectiveness of STF-Mamba. On the Bus and Canteen datasets, its MAE/RMSE are
respectively 6.1%/4.2% and 5.5%/7.2% lower than those of the second-best method.

## Repository Structure

```text
STF-Mamba/
├── dataset/       # Dataset loading and preprocessing
├── model/         # STF-Mamba and backbone modules
├── utils/         # Training, testing, logging, and reproducibility utilities
├── train.py       # Training entry point
├── test.py        # Single-checkpoint evaluation
├── auto_test.py   # Multi-checkpoint evaluation
└── requirements.txt
```

## Installation

Install Python 3, PyTorch, and the required packages:

```bash
pip install -r requirements.txt
```

`mamba-ssm` requires a compatible PyTorch/CUDA build when GPU selective-scan
kernels are used.

## Datasets

- Bus images: [Baidu Netdisk](https://pan.baidu.com/s/1DQGiTBQF3S9vCpy_EMUzsw?pwd=8ybx)
- Bus ground truth: [Baidu Netdisk](https://pan.baidu.com/s/1i3KorhBNSmQ_AsiXmynpYA?pwd=xr8h)

Using the Bus dataset as an example, arrange the files as follows:

```text
Bus/
├── train/
│   ├── ground_truth/
│   │   ├── xxx_0.h5
│   │   └── xxx_10.h5
│   └── images/
│       ├── xxx_0.jpg
│       └── xxx_10.jpg
├── test/
│   ├── ground_truth/
│   └── images/
└── bus_roi.npy
```

## Pretrained Backbone

Download the
[ShiftViT-Tiny weights](https://pan.baidu.com/s/1faf5lFmemvptlAaYd19EZw?pwd=ftvt),
create a directory named `wight` in the project root, and save the file as:

```text
wight/shiftvit-T-G.pth
```

Model weights and datasets are intentionally excluded from Git.

## Training

```bash
python train.py /path/to/Bus ./checkpoints/bus /path/to/Bus/bus_roi.npy
```

The default model uses two MS-STMB blocks in each spatio-temporal stage and one
SFRB block in each frequency stage. These depths can be changed with
`--ms-stmb-depth-1`, `--ms-stmb-depth-2`, `--sfrb-depth-1`, and
`--sfrb-depth-2`.

## Evaluation

Evaluate one checkpoint:

```bash
python test.py \
  -dataset_path /path/to/Bus \
  -pth_path ./checkpoints/bus/best_model.pth \
  -m /path/to/Bus/bus_roi.npy
```

Evaluate every checkpoint in a directory:

```bash
python auto_test.py \
  /path/to/Bus \
  ./checkpoints/bus \
  /path/to/Bus/bus_roi.npy
```

Use `--seed` to reproduce experiments with a fixed random seed.

## Citation

If this code is useful in your research, please cite:

> Zheng-Yan Wu and Miao-Gen Ling. *Video Crowd Counting Based on
> Spatio-Temporal Mamba and Frequency Enhancement*.
