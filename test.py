from models import *
import argparse
import sissautils
import initialize
from ruamel.yaml import YAML
import numpy as np
import logging
import torch
import os
import time

def test_pattern(
    model_config_path: str,
    weights_path: str = None,
) -> tuple[np.ndarray, np.ndarray]:
    logger = logging.getLogger("Test")
    basic_config = YAML().load(open("config/basic.yml", "r"))
    model_config = YAML().load(open(model_config_path, "r"))
    # Create model
    model = SISSA_MODELS[model_config["Model"]["type"]](**model_config["Params"])
    device = model_config["Test"]["device"]
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Cuda not available.")
        logger.critical("Using CUDA.")
        device = torch.device("cuda")
    else:
        logger.critical("Using CPU.")
        device = torch.device("cpu")
    if weights_path is not None:
        model_config["Test"]["weights_path"] = weights_path
        YAML().dump(model_config, open(model_config_path, "w"))
    weights_path = model_config["Test"]["weights_path"]
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    sissautils.model_summary(model, model_config=model_config)
    # Evaluation
    model.eval()
    test_dataloader = sissautils.to_dataloader(
        model_config["Test"]["test_dir"],
        model_config["Test"]["batch_size"],
        model_config["Test"]["n_workers"],
        model_config["Test"]["pin_memory"],
        logger_name="Test",
    )
    y_true = []
    y_pred = []
    time_cost = 0
    for i, (data, label) in enumerate(test_dataloader):
        start_time = time.time()
        data = data.to(device)
        label = label.to(device)
        pred = model(data)
        pred = pred.argmax(dim=1)
        time_cost += time.time() - start_time
        y_true.append(label.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    time_avg = time_cost / len(test_dataloader)
    acc = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            acc += 1
    logger.critical(f"Test acc: {acc / len(y_true):.4f}, avg time: {time_avg:.4f} s")
    labels = [i for i in range(model_config["Params"]["n_classes"])]
    out_dir = os.path.join(model_config["Test"]["out_dir"], model_config["Model"]["name"])
    class_names = basic_config["Preprocessor"]["class_names"]
    fig, cm = sissautils.plot_confusion_matrix(y_true, y_pred, labels, class_names)
    os.makedirs(out_dir, exist_ok=True)
    fig_out_path = os.path.join(out_dir, "cm.png")
    cm_out_path = os.path.join(out_dir, "cm.npy")
    pred_out_path = os.path.join(out_dir, "pred.npy")
    true_out_path = os.path.join(out_dir,"true.npy")
    fig.savefig(fig_out_path)
    np.save(cm_out_path, cm)
    np.save(pred_out_path, y_pred)
    np.save(true_out_path, y_true)


if __name__ == "__main__":
    initialize.init()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_config", type=str)
    argparser.add_argument("--weights", type=str, default=None)
    args = argparser.parse_args()
    sissautils.update_model_config(
        model_config_path=args.model_config
    )
    test_pattern(
        model_config_path=args.model_config,
        weights_path=args.weights
    )
