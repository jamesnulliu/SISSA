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


class SISSA_RNN(nn.Module):
    """
    Some/IP based Safety and Security Analyzer with RNN Backbone
    """
    def __init__(
        self,
        n_pack: int,
        pack_dim: int,
        n_classes: int,
        hidden_size: int,
        attention: bool,
    ):
        super(SISSA_RNN, self).__init__()
        self.n_pack = n_pack
        self.pack_dim = pack_dim
        self.n_classes = n_classes
        self.attention = attention
        self.hidden_size = hidden_size
        self.pmb = sub_blocks.PMB(
            n_pack=self.n_pack, in_dim=pack_dim, out_dim=self.n_pack
        )
        self.rnn1 = nn.RNN(
            input_size=self.n_pack,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.rnn2 = nn.RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        if self.attention:
            self.att_block = sub_blocks.RSAB(
                n_seq=self.n_pack, n_in=self.hidden_size, n_out=hidden_size
            )
        self.mlp = sub_blocks.MLP(
            n_in=n_pack * hidden_size, n_out=n_classes, n_hidden=hidden_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, n_pack, pack_dim)
        x = self.pmb(x)  # (batch_size, n_pack, n_pack)
        x, _ = self.rnn1(x)  # (batch_size, n_pack, hidden_size)
        x, _ = self.rnn2(x)  # (batch_size, n_pack, hidden_size)
        if self.attention:
            x = self.att_block(x)  # (batch_size, n_pack, hidden_size)
        x = x.reshape(x.shape[0], -1)  # (batch_size, n_pack * hidden_size)
        x = self.mlp(x)  # (batch_size, n_classes)
        # Output shape: (batch_size, n_classes)
        return x
