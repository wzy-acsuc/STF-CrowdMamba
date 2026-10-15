# STF-CrowdMamba: Spatio-Temporal Mamba and Spatial-Frequency Refinement for Video Crowd Counting

## Repository Structure

```text
STF-Mamba/
├── dataset/       
├── model/         
├── utils/         
├── train.py       
├── test.py        
├── auto_test.py   
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
├── test/
├── val/
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

