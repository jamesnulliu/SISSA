from models import *
import argparse
import sissautils
import initialize
import torch
from torch.utils.data import DataLoader
import logging
import os
from models import *
import numpy as np


def train_pattern(
    model,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    weights_dir: str,
    train_log_dir: str,
    epochs: int,
    optimizer,
    criterion,
    device,
    **kwargs,
):
    logger = logging.getLogger("Train")
    logger.critical("Start training...")

    train_losses, train_accs, val_accs = [], [], []
    for e in range(epochs):
        model.train()
        train_correct, train_total = 0, 0
        for _, (data, label) in enumerate(train_dataloader):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            train_correct += torch.sum(pred.argmax(dim=1) == label).item()
            train_total += data.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for _, (data, label) in enumerate(val_dataloader):
                data = data.to(device)
                label = label.to(device)
                pred = model(data)
                loss = criterion(pred, label)
                val_correct += torch.sum(pred.argmax(dim=1) == label).item()
                val_total += data.size(0)

        train_acc_epoch = train_correct / train_total
        val_acc_epoch = val_correct / val_total

        train_losses.append(loss.item())
        train_accs.append(train_acc_epoch)
        val_accs.append(val_acc_epoch)

        if (e + 1) % 5 == 0 and e > 69:
            save_path = os.path.join(
                weights_dir,
                f"e-{e+1}_train-acc-{train_acc_epoch:.3}_val-acc-{val_acc_epoch:.3}.pt",
            )
            torch.save(model.state_dict(), save_path)

        logger.info(
            f"Epoch: {e + 1:4} | Train acc: {train_acc_epoch:.4f}, Val acc: {val_acc_epoch:.4f}"
        )

    logger.critical("Training finished.")
    train_losses = np.array(train_losses)
    train_accs = np.array(train_accs)
    val_accs = np.array(val_accs)
    np.save(os.path.join(train_log_dir, "train_losses"), train_losses)
    np.save(os.path.join(train_log_dir, "train_accs"), train_accs)
    np.save(os.path.join(train_log_dir, "val_accs"), val_accs)
    logger.critical(f"Saved loss and accuracy to {train_log_dir}.")


if __name__ == "__main__":
    initialize.init()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_config", type=str)
    args = argparser.parse_args()
    model_config = sissautils.update_model_config(
        model_config_path=args.model_config
    )
    model = SISSA_MODELS[model_config["Model"]["type"]](
        **model_config["Params"]
    )

    hparams = sissautils.hyperparams(model_config=model_config, model=model)
    # sissautils.model_summary(model, model_config=model_config)
    train_pattern(**hparams)
