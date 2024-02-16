"""
Copyright 2023 Shanghai University Cyber Security Laboratary

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

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class PMB(nn.Module):
    """
    Packet Mapping Block
    """

    def __init__(
        self,
        n_pack: int,
        in_dim: int,
        out_dim: int,
        non_linear: Any = lambda x: x,
    ):
        super(PMB, self).__init__()
        self.n_pack = n_pack
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.non_linear = non_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (..., n_pack, in_dim)
        x = x.view(-1, self.in_dim)  # (... * n_pack, in_dim)
        x = self.non_linear(self.linear(x))  # (... * n_pack, out_dim)
        x = x.view(-1, self.n_pack, self.out_dim)  # (..., n_pack, out_dim)
        return x


class CMB(nn.Module):
    """
    Channel Mapping Block

    Breif
    -----
    This block is used to map the channels of the input image to the
    channels of the output image (by each pixel).
    """

    def __init__(
        self,
        n_channels: int,
        img_width: int,
        img_height: int,
        n_out: int,
    ):
        super(CMB, self).__init__()
        self.n_channels = n_channels
        self.n_out = n_out
        self.linear = nn.Linear(n_channels, n_out)
        self.img_width = img_width
        self.img_height = img_height

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (..., n_channels, img_width, img_height)
        x = x.view(-1, self.n_channels)  # (..., n_channels)
        x = self.linear(x)  # (..., n_out)
        x = x.view(
            -1, self.img_width, self.img_height, self.n_out
        )  # (..., img_width, img_height, n_out)
        # Output shape: (..., img_width, img_height, n_out)
        return x


class RSAB(nn.Module):
    """
    Residual Attention Block
    """

    def __init__(self, n_seq: int, n_in: int, n_out: int):
        super(RSAB, self).__init__()
        if n_in != n_out:
            raise ValueError("`n_in` must be equal to `n_out` in Residual Attention Block (RAB).")
        self.W_q = nn.Linear(n_in, n_out)
        self.W_k = nn.Linear(n_in, n_out)
        self.W_v = nn.Linear(n_in, n_out)
        self.n_seq = n_seq
        self.n_in = n_in
        self.n_out = n_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, n_seq, n_in)
        residual = x
        x = x.reshape(-1, self.n_in)  # (batch_size * n_seq, n_in)
        q = self.W_q(x)  # (batch_size * n_seq, n_out)
        k = self.W_k(x)  # (batch_size * n_seq, n_out)
        v = self.W_v(x)  # (batch_size * n_seq, n_out)
        q = q.view(-1, self.n_seq, self.n_out)  # (batch_size, n_seq, n_out)
        k = k.view(-1, self.n_seq, self.n_out)  # (batch_size, n_seq, n_out)
        v = v.view(-1, self.n_seq, self.n_out)  # (batch_size, n_seq, n_out)
        # Scaled dot-product attention
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.n_out, dtype=torch.float)
        )  # (batch_size, n_seq, n_seq)
        attn_score = torch.matmul(
            F.softmax(attn_score, dim=-1), v
        )  # (batch_size, n_seq, n_out)
        return attn_score + residual


class MLP(nn.Module):
    """
    Multi-Layer Perceptron
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, n_in)
        x = F.relu(self.fc1(x))  # (batch_size, n_hidden)
        x = self.fc2(x)  # (batch_size, n_out)
        # Output shape: (batch_size, n_out)
        return x


if __name__ == "__main__":
    mlp = MLP(n_in=16384, n_out=5, n_hidden=128)
    x = torch.randn(128, 16384)
    y = mlp(x)
    print(y.shape)
    pass
