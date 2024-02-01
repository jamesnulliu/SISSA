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

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import sub_blocks


class SISRA_LSTM(nn.Module):
    """
    Some/IP based Safety and Security Analyzer with LSTM Backbone
    """
    def __init__(
        self,
        n_pack: int,
        pack_dim: int,
        n_classes: int,
        hidden_size: int,
        attention: bool,
    ):
        super(SISRA_LSTM, self).__init__()
        self.n_pack = n_pack
        self.pack_dim = pack_dim
        self.n_classes = n_classes
        self.attention = attention
        self.pmb = sub_blocks.PMB(
            n_pack=self.n_pack, in_dim=pack_dim, out_dim=self.n_pack
        )
        self.lstm1 = nn.LSTM(
            input_size=self.n_pack,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        if self.attention:
            self.rab = sub_blocks.RSAB(
                n_seq=n_pack, n_in=hidden_size, n_out=hidden_size
            )
        self.mlp = sub_blocks.MLP(
            n_in=n_pack * hidden_size, n_out=n_classes, n_hidden=hidden_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, n_pack, pack_dim)
        x = self.pmb(x)  # (batch_size, n_pack, n_pack)
        x, _ = self.lstm1(x)  # (batch_size, n_pack, hidden_size)
        x, _ = self.lstm2(x)  # (batch_size, n_pack, hidden_size)
        if self.attention:
            x = self.rab(x)  # (batch_size, n_pack, hidden_size)
        x = x.reshape(x.shape[0], -1)  # (batch_size, n_pack * hidden_size)
        x = self.mlp(x)  # (batch_size, n_classes)
        # Output shape: (batch_size, n_seq, n_classes)
        return x
