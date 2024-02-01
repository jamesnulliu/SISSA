"""
Copyright 2023-2024 Shanghai University Cyber Security Laboratary

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import pandas as pd
import numpy as np
import multiprocessing as mp
import threading as td
from typing import List, Any
import ruamel.yaml
from ruamel.yaml import YAML
import os
import random

from . import utils
from .manipulator import Manipulator


class Preprocessor:
    def __init__(self):
        self.logger = logging.getLogger("Preprocessor")
        self.manipulator = Manipulator()
        self.config = YAML().load(open("./config/basic.yml", "r"))
        self.n_classes = len(self.manipulator.idx2method)
        self.config["Preprocessor"]["n_classes"] = self.n_classes
        self.raw_path = self.config["Data"]["raw_path"]
        self.processed_dir = self.config["Data"]["processed_dir"]
        self.buckets = mp.Manager().list([None] * self.n_classes)

    def worker(
        self,
        df: pd.DataFrame,
        buckets: List[np.ndarray],
        idx: int,
    ):
        data = self.manipulator.manipulate(df, idx)
        if self.config["Preprocessor"]["use_onehot"]:
            data = utils.encoder(data, "err_code", utils.err_code_s2vec)
            data = utils.encoder(data, "message_type", utils.message_type_s2vec)
            data[["time"]] = data[["time"]].apply(utils.normalize_column)
        else:
            data["err_code"] = data["err_code"].apply(
                lambda x: utils.err_code_s2i[x]
            )
            data["message_type"] = data["message_type"].apply(
                lambda x: utils.message_type_s2i[x]
            )
            data[["time", "err_code", "message_type"]] = data[
                ["time", "err_code", "message_type"]
            ].apply(utils.normalize_column)
        data["src"] = data["src"].apply(utils.ip_to_int)
        data["dst"] = data["dst"].apply(utils.ip_to_int)
        data[["src", "dst"]] = data[["src", "dst"]].apply(utils.normalize_column)
        windows = utils.split_to_windows(
            data=data.to_numpy().astype(np.float32),
            window_height=self.config["Preprocessor"]["window_height"],
            step=self.config["Preprocessor"]["step"],
        )
        if buckets[idx] is None:
            buckets[idx] = windows
        elif windows.shape[0] < 1:
            return
        else:
            try:
                buckets[idx] = np.concatenate([buckets[idx], windows], axis=0)
            except:
                self.logger.error(
                    f"Error happened when concatenating bucket {idx}, "
                    + f"bucket shape: {buckets[idx].shape}, "
                    + f"windows shape: {windows.shape}"
                )

    def preprocess(self, rd: int = 1, bias:int=3):
        self.logger.critical("Start preprocessing...")
        for i in range(rd):
            self.logger.critical(f"Processing round:{i}...")
            df = pd.read_csv(self.raw_path)
            # Drop first `i` * `bias` rows
            df = df.drop(df.index[:i * bias])
            splited_dfs = utils.split_dataframe(df, None, rows_per_part=512)
            self.logger.info(f"Split df into {len(splited_dfs)} parts.")
            random.shuffle(splited_dfs)
            futures = []
            for idx, splited_df in enumerate(splited_dfs):
                if self.config["Preprocessor"]["use_multiprocessing"]:
                    future = mp.Process(
                        target=self.worker,
                        kwargs={
                            "df": splited_df,
                            "buckets": self.buckets,
                            "idx": int(idx % self.n_classes),
                        },
                    )
                    futures.append(future)
                    future.start()
                else:
                    self.worker(splited_df, self.buckets, idx % self.n_classes)
                if (idx + 1) % self.n_classes == 0:
                    for future in futures:
                        future.join()
            min_bucket_size = min(
                [
                    bucket.shape[0]
                    for bucket in self.buckets
                    if bucket is not None
                ]
            )
            for idx, bucket in enumerate(self.buckets):
                self.logger.info(f"Bucket {idx} shape: {bucket.shape}")
                if bucket is None:
                    continue
                bucket_ids = np.arange(bucket.shape[0])
                np.random.shuffle(bucket_ids)
                self.buckets[idx] = bucket[bucket_ids[:min_bucket_size]]
                self.logger.info(
                    f"Bucket {idx} reshaped to {self.buckets[idx].shape}"
                )
        window_width = 0
        os.makedirs(self.processed_dir, exist_ok=True)
        for i, bucket in enumerate(self.buckets):
            np.save(os.path.join(self.processed_dir, f"_{i}"), bucket)
            self.logger.critical(
                f"Saved bucket {i} to {self.processed_dir}/_{i}.npy, "
                + f"shape: {bucket.shape}"
            )
            window_width = bucket.shape[2]
        # Update `window_width`
        self.config["Preprocessor"]["window_width"] = window_width
        YAML().dump(self.config, open("config/basic.yml", "w"))
        self.logger.critical("Preprocessing finished.")

    def train_val_split(self):
        self.logger.critical("Start train val split...")
        processed_dir = self.config["Data"]["processed_dir"]
        train_dir = self.config["Data"]["train_dir"]
        val_dir = self.config["Data"]["val_dir"]
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        train_data, train_labels, test_data, test_labels = [None] * 4
        for file in os.listdir(processed_dir):
            if not file.endswith(".npy"):
                continue
            # [Note] Processed file name is "_{label}.npy" by default
            label = int(file[1:-4])
            # Data shape: (num_windows, n_packets, packet_dim)
            data = np.load(os.path.join(processed_dir, file))
            np.random.shuffle(data)
            # Split data into train and test
            train_size = int(data.shape[0] * 0.8)
            if train_data is None:
                train_data = data[:train_size]
                test_data = data[train_size:]
                train_labels = np.full(train_size, label)
                test_labels = np.full(data.shape[0] - train_size, label)
            else:
                train_data = np.concatenate([train_data, data[:train_size]])
                test_data = np.concatenate([test_data, data[train_size:]])
                # Generate labels
                train_labels = np.concatenate(
                    [train_labels, np.full(train_size, label)]
                )
                test_labels = np.concatenate(
                    [test_labels, np.full(data.shape[0] - train_size, label)]
                )

        # Shuffle train and test data
        train_idx = np.arange(train_data.shape[0])
        np.random.shuffle(train_idx)
        train_data = train_data[train_idx]
        train_labels = train_labels[train_idx]
        test_idx = np.arange(test_data.shape[0])
        np.random.shuffle(test_idx)
        test_data = test_data[test_idx]
        test_labels = test_labels[test_idx]

        # Save train and test data
        np.save(os.path.join(train_dir, "data"), train_data)
        self.logger.info(
            f"Saved train data to {train_dir}/data.npy, shape: {train_data.shape}"
        )
        np.save(os.path.join(train_dir, "labels"), train_labels)
        self.logger.info(
            f"Saved train labels to {train_dir}/labels.npy, shape: {train_labels.shape}"
        )
        np.save(os.path.join(val_dir, "data"), test_data)
        self.logger.info(
            f"Saved test data to {val_dir}/data.npy, shape: {test_data.shape}"
        )
        np.save(os.path.join(val_dir, "labels"), test_labels)
        self.logger.info(
            f"Saved test labels to {val_dir}/labels.npy, shape: {test_labels.shape}"
        )
        self.logger.critical("Train test split finished.")
