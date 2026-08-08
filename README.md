# STF-CrowdMamba: Spatio-Temporal Mamba and Spatial-Frequency Refinement for Video Crowd Counting

## Overview
- **We propose a video crowd counting method, termed STF-CrowdMamba, that combines spatio-temporal Mamba modeling with frequency-domain enhancement. Parallel spatio-temporal and frequency-domain branches are constructed to capture long-range dependencies and frequency-domain information, thereby improving complementarity across feature domains. To the best of our knowledge, this is the first study to apply Mamba to multi-frame video crowd counting.
- **We develop MS-STMB, which arranges consecutiveframe features into a joint temporal-height-width sequence and employs bidirectional selective scanning to establish long-range spatio-temporal associations across frames. Average- and max-pooling paths are further introduced to extract crowd features at different spatial scales.
- **Experiments on five public video crowd counting datasets demonstrate that STF-CrowdMamba achieves competitive counting performance.

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
