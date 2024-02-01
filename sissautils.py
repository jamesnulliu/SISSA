import logging
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import os
import numpy as np
import torch
import logging
from ruamel.yaml import YAML
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import torchinfo

from models import *


def to_dataloader(
    dir: str,
    batch_size: int,
    n_workers: int,
    pin_memory: bool,
    logger_name: str,
) -> DataLoader:
    basic_config = YAML().load(open("config/basic.yml", "r"))
    window_height = basic_config["Preprocessor"]["window_height"]
    logger = logging.getLogger(logger_name)
    logger.critical("Sending data to dataloader...")
    data = np.load(
        os.path.join(dir, "data.npy")
    )  # (n_samples, n_pack, pack_dim)
    data = data[0 : int(len(data))]
    data = data[:, 0:window_height, :]
    logger.info(f"Loaded data from {dir}/data.npy, shape: {data.shape}")
    label = np.load(os.path.join(dir, "labels.npy"))
    label = label[0 : int(len(label))]
    logger.info(f"Loaded label from {dir}/labels.npy, shape: {label.shape}")
    dataset = TensorDataset(
        torch.from_numpy(data).float(), torch.from_numpy(label).long()
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=pin_memory,
    )
    logger.critical("Data sent to dataloader.")
    return dataloader


def hyperparams(model_config, model) -> dict:
    hparams = {}
    hparams["model_type"] = model_config["Model"]["type"]
    hparams["train_dataloader"] = to_dataloader(
        dir=model_config["Train"]["train_dir"],
        batch_size=model_config["Train"]["batch_size"],
        n_workers=model_config["Train"]["n_workers"],
        pin_memory=model_config["Train"]["pin_memory"],
        logger_name="Train",
    )
    hparams["val_dataloader"] = to_dataloader(
        dir=model_config["Train"]["val_dir"],
        batch_size=model_config["Train"]["batch_size"],
        n_workers=model_config["Train"]["n_workers"],
        pin_memory=model_config["Train"]["pin_memory"],
        logger_name="Train",
    )
    hparams["lr"] = model_config["Train"]["lr"]
    hparams["weight_decay"] = model_config["Train"]["weight_decay"]
    hparams["epochs"] = model_config["Train"]["epochs"]
    hparams["weights_dir"] = os.path.join(
        model_config["Train"]["weights_dir"], model_config["Model"]["name"]
    )
    hparams["train_log_dir"] = os.path.join(
        model_config["Train"]["train_log_dir"], model_config["Model"]["name"]
    )
    device = model_config["Train"]["device"]
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Cuda not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    hparams["device"] = device
    hparams["optimizer"] = torch.optim.AdamW(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
    )
    hparams["criterion"] = torch.nn.CrossEntropyLoss()
    hparams["model"] = model.to(device)

    return hparams


def update_model_config(model_config_path: str):
    basic_config = YAML().load(open("config/basic.yml", "r"))
    model_config = YAML().load(open(model_config_path, "r"))
    # Make directory to save weights during training
    weights_dir = os.path.join(
        model_config["Train"]["weights_dir"], model_config["Model"]["name"]
    )
    # Make directory to save acc, loss for every epoch
    train_log_dir = os.path.join(
        model_config["Train"]["train_log_dir"], model_config["Model"]["name"]
    )
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(train_log_dir, exist_ok=True)
    # Update model some configs based on basic config
    model_config["Params"]["n_pack"] = basic_config["Preprocessor"][
        "window_height"
    ]
    model_config["Params"]["pack_dim"] = basic_config["Preprocessor"][
        "window_width"
    ]
    model_config["Params"]["n_classes"] = basic_config["Preprocessor"][
        "n_classes"
    ]
    model_config["Train"]["train_dir"] = basic_config["Data"]["train_dir"]
    model_config["Train"]["val_dir"] = basic_config["Data"]["val_dir"]
    YAML().dump(model_config, open(model_config_path, "w"))
    return model_config


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels,
    label_names,
    title="Confusion Matrix",
    cmap="Blues",
    figsize=(10, 8),
    dpi=100,
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(
        cm,
        index=label_names,
        columns=label_names,
    )
    fig = plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(df_cm, annot=True, cmap=cmap, annot_kws={"size": 8}, fmt="d")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    return fig, cm


def model_summary(model, model_config):
    input_size = (
        model_config["Params"]["n_pack"],
        model_config["Params"]["pack_dim"],
    )
    torchinfo.summary(model=model, input_size=input_size)
