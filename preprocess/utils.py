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

import pandas as pd
import numpy as np
from typing import List
import torch
import torch.nn as nn


err_code_i2s = {
    0: "E_OK",
    1: "E_NOT_OK",
    2: "E_UNKNOWN_SERVICE",
    3: "E_UNKNOWN_METHOD",
    4: "E_NOT_READY",
    5: "E_NOT_REACHABLE",
    6: "E_TIMEOUT",
    7: "E_WRONG_PROTOCOL_VERSION",
    8: "E_WRONG_INTERFACE_VERSION",
    9: "E_MALFORMED_MESSAGE",
    10: "E_WRONG_MESSAGE_TYPE",
}

err_code_s2i = {
    "E_OK": 0,
    "E_NOT_OK": 1,
    "E_UNKNOWN_SERVICE": 2,
    "E_UNKNOWN_METHOD": 3,
    "E_NOT_READY": 4,
    "E_NOT_REACHABLE": 5,
    "E_TIMEOUT": 6,
    "E_WRONG_PROTOCOL_VERSION": 7,
    "E_WRONG_INTERFACE_VERSION": 8,
    "E_MALFORMED_MESSAGE": 9,
    "E_WRONG_MESSAGE_TYPE": 10,
}

err_code_s2vec = {
    err_code_i2s[0]: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    err_code_i2s[1]: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    err_code_i2s[2]: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    err_code_i2s[3]: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    err_code_i2s[4]: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    err_code_i2s[5]: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    err_code_i2s[6]: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    err_code_i2s[7]: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    err_code_i2s[8]: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    err_code_i2s[9]: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    err_code_i2s[10]: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

message_type_i2s = {
    0: "REQUEST",
    1: "REQUEST_NO_RETURN",
    2: "NOTIFICATION",
    128: "RESPONSE",
    129: "ERROR",
}

message_type_s2i = {
    "REQUEST": 0,
    "REQUEST_NO_RETURN": 1,
    "NOTIFICATION": 2,
    "RESPONSE": 128,
    "ERROR": 129,
}


message_type_s2vec = {
    message_type_i2s[0]: [0, 0, 0, 0, 1],
    message_type_i2s[1]: [0, 0, 0, 1, 0],
    message_type_i2s[2]: [0, 0, 1, 0, 0],
    message_type_i2s[128]: [0, 1, 0, 0, 0],
    message_type_i2s[129]: [1, 0, 0, 0, 0],
}


def ip_to_int(ip: str):
    """
    Convert an IP address to an integer.
    """
    return float(ip.replace(".", ""))


def normalize_column(data: pd.Series) -> pd.Series:
    """
    Normalize a column of data.
    """
    if data.std() == 0:
        return data - data.mean()
    else:
        return (data - data.mean()) / data.std()


def encoder(df: pd.DataFrame, column: str, encoding: dict):
    if column not in df.columns:
        return df
    target_data = df[column]
    # Check nan
    if target_data.isna().any():
        raise ValueError(f"NaN found in column {column}.")
    encoded_data = []
    for elem in target_data:
        encoded_data.append(encoding[elem])
    if not isinstance(list(encoding.values())[0], list):
        raise ValueError(
            "Encoding values must be a list. Perhaps you shuold use pd.replace() instead."
        )
    encoded_names = [
        column + "_" + str(i) for i in range(len(list(encoding.values())[0]))
    ]
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_names)
    df = df.drop(columns=[column])
    df = pd.concat([df, encoded_df], axis=1)
    return df


def split_to_windows(
    data: np.ndarray,
    window_height: int = 128,
    step: int = 64,
) -> np.ndarray:
    """
    Split the raw data into windows of size `window_size` with step size `step`.

    Parameters
    ----------
    data : np.ndarray
        The raw data.
    window_height : int, optional
        The height of the window, by default 128.
    step : int, optional
        The step size, by default 64.

    Returns
    -------
    np.ndarray
        A 3D array of shape (num_windows, window_height, window_width).
    """
    windows = []
    for i in range(0, len(data), step):
        win = data[i : i + window_height]
        if len(win) == window_height:
            windows.append(win)
    windows = np.array(windows)
    if np.isnan(windows).any():
        raise ValueError("NaN found in windows.")
    return windows


def df_to_windows(
    df: pd.DataFrame,
    window_height: int = 128,
    step: int = 64,
) -> np.ndarray:
    """
    Split the raw data into windows of size `window_size` with step size `step`.

    Parameters
    ----------
    df : pd.DataFrame
        The raw data.
    window_height : int, optional
        The height of the window, by default 128.
    step : int, optional
        The step size, by default 64.

    Returns
    -------
    List[pd.DataFrame]

    """
    windows = []
    for i in range(0, df.shape[0], step):
        win = df.iloc[i : i + window_height, :]
        if win.shape[0] == window_height:
            windows.append(win)
    return windows


def split_dataframe(
    df: pd.DataFrame, n_parts: int, rows_per_part: int = None
) -> List[pd.DataFrame]:
    total_rows = df.shape[0]
    if rows_per_part is None:
        rows_per_part = total_rows // n_parts
    if n_parts is None:
        n_parts = total_rows // rows_per_part
    split_dataframes = []
    for i in range(n_parts):
        start_idx = i * rows_per_part
        end_idx = (i + 1) * rows_per_part if i < n_parts - 1 else total_rows
        split_df = df.iloc[start_idx:end_idx, :]
        split_dataframes.append(split_df)
    return split_dataframes


if __name__ == "__main__":
    data = {
        "err_code": ["E_OK", "E_NOT_READY", "E_TIMEOUT"],
        "message_type": ["REQUEST", "ERROR", "ERROR"],
    }
    df = pd.DataFrame(data)
    df = encoder(df, "err_code", err_code_s2vec)
    df = encoder(df, "message_type", message_type_s2vec)
    pass
