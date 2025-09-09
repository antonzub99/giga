# <center>GIGA: Generalizable Sparse Image-driven Gaussian Humans</center>

This is an official repository for the GIGA project.

## Table of Contents

- [Prepare the environment](#prepare-the-environment)
  - [0. Requirements](#0-requirements)
  - [1. Conda/Mamba: CUDA setup](#1-condamamba-cuda-setup)
  - [(Optional) 1a. System-wide CUDA setup](#optional-1a-system-wide-cuda-setup)
  - [2. SMPLX setup](#2-smplx-setup)
  - [3. GIGA setup](#3-giga-setup)
- [Training](#training)
  - [Datasets](#datasets)
  - [Preparing static textures](#preparing-static-textures)
  - [Training command](#training-command)
- [Testing](#testing)
  - [Evaluation](#evaluation)
  - [Render-only (no metrics)](#render-only-no-metrics)
  - [Freeview rendering](#freeview-rendering)
- [Other pytorch/CUDA versions](#other-pytorchcuda-versions)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)


## Prepare the environment

### 0. Requirements

* NVIDIA GPU (tested on RTX 3090, A100, H100).
* A working CUDA installation.
* Preferrably Linux (should work on Windows too though).
* Install [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) to manage Python packaging, your life will become much easier.

### 1. Conda/Mamba: CUDA setup

This repo was tested with CUDA 12.8. The easiest way is to install it using `conda`, `mamba` or `micromamba`:

```bash
mamba create -f environment.yml
```

We will have to link some of the CUDA headers to an appropriate place, so please run this command too:

```bash
mamba activate giga
ln -s $CONDA_PREFIX/targets/x86_64-linux/include/* $CONDA_PREFIX/include
```

### (Optional) 1a. System-wide CUDA setup

If you have CUDA 12.8 already installed on your system, or you are on Windows and symlinking might be an issue, you can avoid installing CUDA from the anaconda packages.

Make sure to set the following environment variables correctly:

```bash
export CUDA_HOME=/usr/local/cuda-12.8 # might be in /usr/lib/cuda-12.8
# or on Windows, in /c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64
```

To verify that CUDA is set correctly, run `nvcc -V`. You should see that your CUDA compiler has version 12.8


### 2. SMPLX setup

Register at the official website of [SMPLX](https://smpl-x.is.tue.mpg.de/), accept the terms of license and download `.npz` files for male, female and neutral models; also download the `.npz` with the UV parameterization and put them in the `$SMPLX_DATA` folder.


### 3. GIGA setup

When CUDA is properly set, run this command (it might take a while to correctly trace dependencies for gsplat):

```bash
uv sync
```

## Training

We use [wandb](https://wandb.ai/quickstart?product=models) to log training progress - we recommed to use it too, if you want to train GIGA.

Training parameters are configured in a `.yaml` file; the dataset(s) use a separate configuration file. We provide examples in the repo.

### Datasets

In this repo, training and testing functionality is presented only for [MVHumanNet](https://github.com/GAP-LAB-CUHK-SZ/MVHumanNet), [MVHumanNet++](https://github.com/GAP-LAB-CUHK-SZ/MVHumanNet_plusplus) and [DNA-Rendering](https://dna-rendering.github.io/) datasets.

Download them first, then make sure to set correct paths to the datasets in each of the corresponding dataset configs (for example, [./configs/mvh/train_data.yaml](./configs/mvh/train_data.yaml)).

To work with other datasets, follow [DATASETS.md](./DATASETS.md) for instructions.

### Preparing static textures

To train/evaluate GIGA designed for monocular inputs, you will have to prepare static RGB textures with the following script:

```bash
python scripts/generate_static_textures.py \
    configs/mvh/texture_data.yaml \ 
    /path/to/output/textures \ # textures will be saved here
    --smplx-path $SMPLX_DATA \
    --texture-resolution 1024
```

This creates approximate textures from multi-view A-pose images (if they are available, otherwise the images are drawn for the pose specified in the dataset config). Once textures are ready, set `texture_dir` field in the dataset config to `/path/to/output/textures`.

### Training command

Starting training is simple:

```bash
# if you installed CUDA with mamba
mamba activate giga
source .venv/bin/activate
# source .venv/Scripts/activate.bat on Windows

python scripts/train.py \
    --model-config configs/mvh/giga.yaml \
    --dataset-config configs/mvh/texture_data.yaml \
    --smplx-path $SMPLX_DATA # directory with SMPLX .npz models
    # to override config parameters from CLI:
    # experiment_name=giga_alternative \
    # trainer.compile=False
```

Tips:
- Provide multiple dataset configs with commas: `--dataset-config configs/datasets/a.yaml,configs/datasets/b.yaml` if you want to train on a mixture of datasets.
- Specify `--resume-id <WANDB_ID>` to resume training logged to the `<WANDB_ID>` experiment.
- For training on SLURM clusters, check out the [`slurm_train_job.py`](./scripts/slurm_train_job.py) script. Other clusters have not been explored, feel free to adapt for your case.

## Testing

We provide checkpoints of GIGA trained on MVHumanNet and DNA-Rendering, download them with this command:

```bash
bash download_checkpoint.sh mvh # or dna
```

### Evaluation

To render virtual humans from selected cameras and evaluate metrics on them, use this:

```bash
mamba activate giga
source .venv/bin/activate

python scripts/test.py evaluate \
  --model-config configs/mvh/giga.yaml \
  --dataset-config configs/mvh/eval_data.yaml \
  --output-path /path/to/output \
  --smplx-path $SMPLX_DATA \
  --actor-id '100027' # example actor id from MVH dataset
```

Model configs specified with `--model-config` should be in the same directory with respective checkpoints. [Downloading script](#testing) will take care of this for the provided checkpoints.

### Render-only (no metrics)

If you only need to render images without computing metrics:

```bash
mamba activate giga
source .venv/bin/activate

python scripts/test.py render \
  --model-config configs/mvh/giga.yaml \
  --dataset-config configs/mvh/eval_data.yaml \
  --output-path /path/to/output \
  --smplx-path $SMPLX_DATA \
  --actor-id '100027'
```

In general, running `python scripts/test.py --help` and `python scripts/test.py evaluate --help` will help you to understand arguments better.

### Freeview rendering

This is the script to render a circular camera trajectory around the actor:

```bash
mamba activate giga
source .venv/bin/activate

python scripts/test.py freeview \
  --model-config configs/mvh/giga.yaml \
  --dataset-config configs/mvh/eval_data.yaml \
  --output-path /path/to/output \
  --smplx-path $SMPLX_DATA \
  --actor-id '100027' \
  --trajectory-mode orbit \
  --up +y
```

Notes:

- `--trajectory-mode` can be either `orbit` for a circular orbiting camera around the actor, or `interpolate` to interpolate between views from the dataset.
- Run `python scripts/test.py freeview --help` for help with other options.

## Other pytorch/CUDA versions

This repo has been also tested with `torch==2.6.0+cu124`, `torch==2.7.0`, `torch==2.7.1`. In fact, any working combination of pytorch (not older than 2.6.0) and CUDA should work. If you want to use different pytorch and CUDA, modify [pyproject.toml](./pyproject.toml) and run `uv sync --reinstall` - this exercise is left for the reader.

## Acknowledgements

* The SO3 manipulation module was borrowed from [abcamiletto](https://github.com/abcamiletto).
* Throughout this project, the Claude Sonnet family (3.5, 3.7 and 4) and Gemini 2.5 Pro provided invaluable help.

## Citation

```bibtex
@article{zubekhin2025giga,
title={GIGA: Generalizable Sparse Image-driven Gaussian Humans},
author={Zubekhin, Anton and Zhu, Heming and Gotardo, Paulo and Beeler, Thabo  and Habermann, Marc and Theobalt, Christian},
year={2025},
journal={arXiv},
eprint={2504.07144},
}
```




