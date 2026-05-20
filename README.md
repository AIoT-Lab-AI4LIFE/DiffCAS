# DiffCAS: Inference-time CT-free Diffusion Model for Physics-aware Multi-slice Attenuation Correction in Cardiac SPECT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

DiffCAS generates attenuation-corrected (AC) cardiac SPECT images directly from
non-attenuation-corrected (NAC) inputs **without requiring CT at inference time**.
It is built on a Brownian Bridge Diffusion Model (BBDM) and trained with a
two-stage Teacher–Student framework:

- **DiffCas-Teacher** — a CT-conditioned diffusion model that learns
  physics-informed attenuation priors from paired NAC/CT/AC training data.
- **DiffCas-Student** — a CT-free student distilled from the Teacher, combining
  a reconstruction loss with a cosine-similarity feature-alignment loss so the
  student replicates the Teacher's internal representations at inference time.

## Repository Structure

```
DiffCAS/
├── DiffCas-Teacher/          # CT-conditioned teacher model
│   ├── main.py               # Entry point (train / test)
│   ├── configs/
│   │   └── DiffCas-Teacher.yaml
│   ├── datasets/             # Dataset loaders
│   ├── runners/
│   │   └── BaseRunner.py     # Training/validation/test loop
│   ├── evaluation/           # FID, LPIPS, diversity metrics
│   ├── calculate_metrics.py  # SSIM, PSNR, RMSE
│   ├── preprocess_and_evaluation.py
│   ├── Register.py
│   ├── utils.py
│   └── environment.yml
│
└── DiffCas-Student/          # CT-free student model
    ├── main.py               # Entry point (train / test)
    ├── configs/
    │   └── Template-DiffCas-S.yaml
    ├── datasets/             # Dataset loaders
    ├── runners/
    │   └── BaseRunner.py     # Training loop with distillation losses
    ├── evaluation/           # FID, LPIPS, diversity metrics
    ├── model/                # UNet and registration (Reg / Transformer_2D)
    ├── shell/
    │   └── Template-shell.sh # Example training / test commands
    ├── calculate_metrics.py  # SSIM, PSNR, RMSE
    ├── preprocess_and_evaluation.py
    ├── Register.py
    ├── utils.py
    └── environment.yml
```

## Key Features

- **Brownian Bridge Diffusion Model (BBDM)**: models the NAC→AC translation as
  a stochastic bridge between two image distributions rather than pure noise.
- **Physics-aware Teacher**: conditions on CT images via a `CT_transformer`
  UNet conditioning key, encoding patient-specific attenuation maps.
- **Multi-slice contextual learning**: each sample spans the target slice ±8
  neighboring slices (18 input/output channels), capturing volumetric context.
- **Knowledge distillation**: the Student is trained with a combined loss —
  pixel-level L1 reconstruction and cosine-similarity alignment of Teacher
  feature maps at three UNet stages (input block, middle block, output block).
- **Spatial registration**: a learnable deformable registration network (`Reg` /
  `Transformer_2D`) is co-trained with the Student to handle slice-alignment.
- **Exponential Moving Average (EMA)**: EMA weights (decay 0.995) are used
  during validation and inference for stable outputs.
- **Multi-GPU training**: native PyTorch DDP with NCCL backend.

## Environment Setup

Both modules share the same conda environment (`BBDM`):

```bash
conda env create -f DiffCas-Teacher/environment.yml
conda activate BBDM
```

Key dependencies:

| Package | Version |
|---|---|
| Python | 3.9 |
| PyTorch | 1.12.1+cu113 |
| torchvision | 0.13.1+cu113 |
| pytorch-lightning | 1.9.3 |
| lpips | 0.1.4 |
| einops | 0.6.0 |
| transformers | 4.26.1 |
| scikit-image | 0.20.0 |

## Data Preparation

Both Teacher and Student expect a dataset in `custom_aligned` format.
Set `dataset_path` in the config YAML to point to your data directory.
The default image resolution is **64×64** with **3 channels**, normalized
to `[-1, 1]` (`to_normal: True`).

## Training

### Stage 1 — Train the Teacher (CT-conditioned)

```bash
cd DiffCas-Teacher
python main.py \
  --config configs/DiffCas-Teacher.yaml \
  --train \
  --save_top \
  --gpu_ids 0
```

Multi-GPU (e.g., 4 GPUs):
```bash
python main.py \
  --config configs/DiffCas-Teacher.yaml \
  --train \
  --save_top \
  --gpu_ids 0,1,2,3
```

Resume from a checkpoint:
```bash
python main.py \
  --config configs/DiffCas-Teacher.yaml \
  --train \
  --resume_model path/to/last_model.pth \
  --resume_optim path/to/last_optim_sche.pth \
  --gpu_ids 0
```

### Stage 2 — Train the Student (CT-free)

Edit `configs/Template-DiffCas-S.yaml` and set `teacher_ckpt` to the
Teacher checkpoint path, then:

```bash
cd DiffCas-Student
python main.py \
  --config configs/Template-DiffCas-S.yaml \
  --train \
  --sample_at_start \
  --save_top \
  --gpu_ids 0 \
  --resume_model path/to/model_ckpt \
  --resume_optim path/to/optim_ckpt
```

### Key Training Hyperparameters

| Hyperparameter | Teacher | Student |
|---|---|---|
| Epochs | 200 | 200 |
| Max steps | 400 000 | 400 000 |
| Batch size (train) | 4 | 8 |
| Optimizer | Adam | Adam |
| Learning rate | 1e-4 | 1e-4 |
| LR scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| Diffusion timesteps | 500 | 500 |
| Sampling steps | 100 | 100 |
| EMA decay | 0.995 | 0.995 |
| Grad accumulation | 2 | 2 |
| Neighbor slices | 8 | 8 |

## Inference / Testing

```bash
# Teacher (requires CT at test time)
cd DiffCas-Teacher
python main.py \
  --config configs/DiffCas-Teacher.yaml \
  --sample_to_eval \
  --gpu_ids 0 \
  --resume_model path/to/model_ckpt

# Student (CT-free inference)
cd DiffCas-Student
python main.py \
  --config configs/Template-DiffCas-S.yaml \
  --sample_to_eval \
  --gpu_ids 0 \
  --resume_model path/to/model_ckpt
```

Results are written to the `results/` directory under the path derived from
`dataset_name` and `model_name` in the config.

## Evaluation

Compute image quality metrics on saved predictions:

```bash
# SSIM / PSNR / RMSE  (computed inline during sampling via calculate_metrics.py)

# LPIPS
python preprocess_and_evaluation.py \
  --func_name LPIPS \
  --source_dir path/to/predictions \
  --target_dir path/to/ground_truth \
  --num_samples 5

# Diversity across stochastic samples
python preprocess_and_evaluation.py \
  --func_name diversity \
  --source_dir path/to/predictions \
  --num_samples 5
```

Supported metrics: **SSIM**, **PSNR**, **RMSE**, **LPIPS**, **Diversity**.

## Citation

If you find this work useful, please cite:

```bibtex
@article{vu2026diffcas,
  title={DiffCAS: Inference-time CT-free Diffusion Model for Physics-aware
         Multi-slice Attenuation Correction in Cardiac SPECT},
  author={Vu, Hoang Minh and Pham, Trung Kien and Nguyen, Thi Ha Chi and
          Nguyen, Hai Dang and Nguyen, Dac Thai and Son, Mai Hong and
          Nguyen, Thanh Trung and Nguyen, Trung Thanh and Nguyen, Phi Le},
  year={2026}
}
```

## Contact

For questions or collaborations, please open an issue or contact the
corresponding authors.
