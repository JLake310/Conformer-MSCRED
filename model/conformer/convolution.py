# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .activation import Swish, GLU


class ConformerConvModule(nn.Module):
    """
    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout

    Inputs: inputs
        inputs (batch, seq_len, num_channels, height, width): Tensor contains input sequences

    Outputs: outputs
        outputs (batch, num_channels, height, width): Tensor produces by conformer convolution module.
    """

    def __init__(
        self,
        in_channels: int,
        conv_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 1,
        dropout_p: float = 0.1,
    ) -> None:
        super(ConformerConvModule, self).__init__()
        assert (
            kernel_size - 1
        ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.layernorm = nn.LayerNorm(in_channels)
        self.conv1 = nn.Conv3d(
            conv_channels,
            conv_channels,
            (kernel_size, 1, 1),
            (1, 1, 1),
            ((kernel_size - 1) // 2, 0, 0),
        )
        self.glu = GLU(dim=2)
        self.conv2 = nn.Conv3d(
            conv_channels,
            conv_channels,
            (kernel_size, 1, 1),
            (1, 1, 1),
            ((kernel_size - 1) // 2, 0, 0),
        )
        self.batchnorm = nn.BatchNorm3d(conv_channels)
        self.swish = Swish()
        self.conv3 = nn.Conv3d(
            conv_channels,
            1,
            (kernel_size, 1, 1),
            (1, 1, 1),
            ((kernel_size - 1) // 2, 0, 0),
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor) -> Tensor:
        x: Tensor = self.layernorm(inputs)
        x = self.conv1(x)
        # x = self.glu(x)
        # print(x.size())
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.swish(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = x.squeeze(1)
        return x


class Conv2dSubampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs

    Returns: outputs, output_lengths
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(
            batch_size, subsampled_lengths, channels * sumsampled_dim
        )

        output_lengths = input_lengths >> 2
        output_lengths -= 1

        return outputs, output_lengths
