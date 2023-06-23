# Bachelor's Thesis

## Overview

This repository contains the code for my bachelor's thesis with the title **Explainable AI in the Context of Autonomous Driving Using Integrated Gradients**

Most of the code in the motion-prediction folder was written by Robin Baumann for a ongoing research project. The repository can be found [here on GitLab](https://gitlab.com/Robin.Baumann/motion-prediction)

I made minor adjustments to the motion-prediction code, including but not limited to:
- Refined the device selection logic
- Fixed some bugs
- Added addional prediction method

My main work is contained in the `integrated_gradients.py` file.

## Goals

Analyze the prediction of the existing LSTM model with the integrated gradients method

## Install Guide

### Install Miniconda

Follow instructions for your OS here: https://docs.conda.io/en/latest/miniconda.html

#### Paperspace:

1. Download and install

    ```sh
    wget -O https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh
    chmod +x Miniconda3-py39_23.1.0-1-Linux-x86_64.sh
    bash Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -b -p /notebooks/miniconda
    /notebooks/miniconda/condabin/conda init bash
    ```

2. Restart terminal

### Configuration of Conda environment

```sh
conda create --name motion-prediction python=3.8 -y
conda activate motion-prediction

pip install pyyaml torch torchvision nuscenes-devkit
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
```

Because of outdated  pip version in argoverse repo download the repo locally
```sh
cd dl/
git clone https://github.com/argoai/argoverse-api.git
```
In setup.py modify the version of numpy to 1.22

```sh
pip install -e argoverse-api
pip install av2 torch-scatter tensorboard
cd ..
```

### Run project

`cd motion-prediction`

#### Preprocessing
`python preprocess.py --dataset nuscenes`

#### Training
`python train.py --dataset nuscenes --model LSTM`

#### Visualization
`python viz.py --prediction model-2s6s-LSTM-MSE-K1-nuscenes-100-2023-04-02_16-46.pth --model LSTM --dataset nuscenes --scene_idx 23`

#### Evaluate
`python evaluate.py --trained_model model-2s6s-LSTM-MSE-K1-nuscenes-100-2023-04-02_16-46.pth --dataset nuscenes --model LSTM`

#### Integrated Gradients
`python integrated_gradients.py --trained_model model-2s6s-LSTM-MSE-K1-nuscenes-100-2023-04-02_16-46.pth --dataset nuscenes --model LSTM --batch_size 20 --idx 9 --type batch`
