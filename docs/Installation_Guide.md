# Installation Guide

Anaconda is recommended for python environment management.

Create and activate a new environment for SISSA:

```bash
conda create -n sissa
conda activate sissa
```

Go to the [official website](https://pytorch.org/) of pytorch. Install pytorch based on your local environment. For example, if you are using linux, the installation command could be something like the following:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then install other necessary packages with following command:

```bash
pip install -r requirements.txt
```