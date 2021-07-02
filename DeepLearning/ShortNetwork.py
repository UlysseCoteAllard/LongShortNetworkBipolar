# Class adapted from:
# https://github.com/TheMrGhostman/InceptionTime-Pytorch
# and
# https://github.com/okrasolar/pytorch-timeseries
import numpy as np
from typing import cast, Union, List

import torch
from torch import nn

from DeepLearning.deep_learning_utils import Conv1dSamePadding


class ShortNetwork(nn.Module):
    """A PyTorch implementation of the InceptionTime model.
    From https://arxiv.org/abs/1909.04939
    Attributes
    ----------
    num_blocks:
        The number of inception blocks to use. One inception block consists
        of 3 convolutional layers, (optionally) a bottleneck and (optionally) a residual
        connector
    in_channels:
        The number of input channels (i.e. input.shape[-1])
    number_channel_out_conv:
        The number of "hidden channels" to use. Can be a list (for each block) or an
        int, in which case the same value will be applied to each block
    bottleneck_channels:
        The number of channels to use for the bottleneck. Can be list or int. If 0, no
        bottleneck is applied
    kernel_sizes:
        The size of the kernels to use for each inception block. Within each block, each
        of the 3 convolutional layers will have kernel size
        `[kernel_size // (2 ** i) for i in range(3)]`
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, num_blocks: int, in_channels: int, number_channel_out_conv: Union[List[int], int],
                 bottleneck_channels: Union[List[int], int], kernel_sizes: Union[List[int], int],
                 use_residuals: Union[List[bool], bool, str] = 'default',
                 num_pred_classes: int = 2, activation_function=nn.SiLU
                 ) -> None:
        super().__init__()
        # for easier saving and loading
        self._input_args = {
            'num_blocks': num_blocks,
            'in_channels': in_channels,
            'out_channels': number_channel_out_conv,
            'bottleneck_channels': bottleneck_channels,
            'kernel_sizes': kernel_sizes,
            'use_residuals': use_residuals,
            'num_pred_classes': num_pred_classes
        }
        bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels,
                                                                     num_blocks))
        kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
        if use_residuals == 'default':
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]

        use_residuals = cast(List[bool], self._expand_to_blocks(
            cast(Union[bool, List[bool]], use_residuals), num_blocks)
                             )

        blocks = []
        for i in range(num_blocks):
            if i == 0:
                blocks.append(InceptionBlock(in_channels=in_channels, out_channels=number_channel_out_conv,
                                             residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
                                             kernel_size=kernel_sizes[i], activation_function=activation_function))
            else:
                blocks.append(
                    InceptionBlock(in_channels=number_channel_out_conv * 4, out_channels=number_channel_out_conv,
                                   residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
                                   kernel_size=kernel_sizes[i], activation_function=activation_function))
        self._blocks = nn.Sequential(*blocks)
        self._linear = nn.Linear(in_features=number_channel_out_conv * 4, out_features=num_pred_classes)
        print(self)
        print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, input: torch.Tensor, get_features=False) -> torch.Tensor:  # type: ignore
        x = input
        x = torch.swapaxes(x, 2, 1)
        x = self._blocks(x).mean(dim=-1)  # the mean is the global average pooling
        if get_features:
            return x
        else:
            return self._linear(x)

    @staticmethod
    def _expand_to_blocks(value: Union[int, bool, List[int], List[bool]],
                          num_blocks: int) -> Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, \
                f'Length of inputs lists must be the same as num blocks, ' \
                f'expected length {num_blocks}, got {len(value)}'
        else:
            value = [value] * num_blocks
        return value


class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(self, in_channels: int, out_channels: int,
                 residual: bool, activation_function, stride: int = 1, bottleneck_channels: int = 32,
                 kernel_size: int = 41) -> None:
        super().__init__()
        self._use_bottleneck = bottleneck_channels > 0
        if self._use_bottleneck:
            self._bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels,
                                                 kernel_size=1, bias=False)
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        start_channels = bottleneck_channels if self._use_bottleneck else in_channels

        self._conv1 = Conv1dSamePadding(in_channels=start_channels, out_channels=out_channels,
                                        kernel_size=kernel_size_s[0], stride=stride, bias=False)
        self._conv2 = Conv1dSamePadding(in_channels=start_channels, out_channels=out_channels,
                                        kernel_size=kernel_size_s[1] + 1, stride=stride, bias=False)
        self._conv3 = Conv1dSamePadding(in_channels=start_channels, out_channels=out_channels,
                                        kernel_size=kernel_size_s[2] + 1, stride=stride, bias=False)
        self._maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self._conv_from_maxpool = Conv1dSamePadding(in_channels=start_channels, out_channels=out_channels,
                                                    kernel_size=1, stride=1, bias=False)
        self._activation = activation_function()
        self._batch_norm = nn.BatchNorm1d(num_features=out_channels * 4)

        self._use_residual = residual
        if residual:
            self._residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels * 4,
                                  kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * 4),
                activation_function()
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        original_x = x
        if self._use_bottleneck:
            x = self._bottleneck(x)
        z_maxpool = self._maxpool(x)
        z1 = self._conv1(x)
        z2 = self._conv2(x)
        z3 = self._conv3(x)
        z4 = self._conv_from_maxpool(z_maxpool)
        z_concatenated = torch.cat([z1, z2, z3, z4], 1)
        z_concatenated = self._activation(self._batch_norm(z_concatenated))
        if self._use_residual:
            z_concatenated = z_concatenated + self._residual(original_x)
        return z_concatenated
