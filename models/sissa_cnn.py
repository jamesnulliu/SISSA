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
from math import floor

from . import sub_blocks


class SISSA_CNN(nn.Module):
    """
    Some/IP based Safety and Security Analyzer with CNN Backbone
    """
    def __init__(
        self,
        n_pack: int,
        pack_dim: int,
        n_classes: int,
        hidden_img_size: int,
        kernel_size: int,
        stride: int,
        padding: int,
        attention: bool,
    ):
        super(SISSA_CNN, self).__init__()
        self.n_pack = n_pack
        self.pack_dim = pack_dim
        self.n_class = n_classes
        self.hidden_img_size = hidden_img_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.attention = attention
        self.pmb = sub_blocks.PMB(n_pack, pack_dim, self.n_pack)
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
        )

        self.cnnb1_out_size = floor(
            self.conv_out_size(
                (self.hidden_img_size, self.hidden_img_size),
                kernel_size=self.kernel_size,
                stride=self.stride,
                pad=self.padding,
            )[0]
            * 0.5
        )

        self.cnnb2_out_size = floor(
            self.conv_out_size(
                (self.cnnb1_out_size, self.cnnb1_out_size),
                kernel_size=self.kernel_size,
                stride=self.stride,
                pad=self.padding,
            )[0]
            * 0.5
        )

        self.cmb = sub_blocks.CMB(
            n_channels=32,
            img_width=self.cnnb2_out_size,
            img_height=self.cnnb2_out_size,
            n_out=1,
        )

        if self.attention:
            self.att_block = sub_blocks.RSAB(
                n_seq=self.cnnb2_out_size,
                n_in=self.cnnb2_out_size,
                n_out=self.cnnb2_out_size,
            )

        self.mlp = sub_blocks.MLP(
            n_in=self.cnnb2_out_size * self.cnnb2_out_size,
            n_out=self.n_class,
            n_hidden=self.cnnb2_out_size * self.cnnb2_out_size,
        )

    @staticmethod
    def conv_out_size(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor(
            (
                (h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1)
                / stride
            )
            + 1
        )
        w = floor(
            (
                (h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1)
                / stride
            )
            + 1
        )
        return h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, n_pack, in_dim)
        x = self.pmb(x)  # (batch_size, n_pack, n_pack)
        x = x.view(
            -1, 1, self.n_pack, self.n_pack
        )  # (batch_size, 1, n_pack, n_pack)
        # Interpolate `x` from (batch_size, 1, n_pack, n_pack)
        # to (batch_size, 1, hidden_img_size, hidden_img_size)
        x = F.interpolate(
            x,
            size=(self.hidden_img_size, self.hidden_img_size),
        )  # (batch_size, 1, img_size, img_size)
        x = self.conv_block_1(
            x
        )  # (batch_size, 16, cnnb1_out_size, cnnb1_out_size)
        x = self.conv_block_2(
            x
        )  # (batch_size, 32, cnnb2_out_size, cnnb2_out_size)
        x = self.cmb(x)  # (batch_size, cnnb2_out_size, cnnb2_out_size, 1)
        x = x.squeeze(3)
        if self.attention:
            x = self.att_block(
                x
            )  # (batch_size, cnnb2_out_size, cnnb2_out_size)
        x = x.view(x.shape[0], -1)  # (batch_size, cnnb2_out_size * cnnb2_out_size) 
        x = self.mlp(x)  # (batch_size, out_dim)
        return x
