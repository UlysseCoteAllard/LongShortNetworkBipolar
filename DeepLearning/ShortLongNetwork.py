import numpy as np

import torch
from torch import nn
from typing import Union, List

from DeepLearning.ShortNetwork import ShortNetwork


class LongNetwork(nn.Module):
    def __init__(self, number_features_long: int, num_blocks_short_network: int, in_channels_short_network: int,
                 number_channel_out_conv_short_network: Union[List[int], int],
                 bottleneck_channels_short_network: Union[List[int], int],
                 kernel_sizes_short_network: Union[List[int], int],
                 use_residuals_short_network: Union[List[bool], bool, str] = 'default',
                 num_pred_classes: int = 2, activation_function=nn.SiLU
                 ) -> None:
        super().__init__()

        self._number_features_long = number_features_long

        self._short_network = ShortNetwork(num_blocks=num_blocks_short_network, in_channels=in_channels_short_network,
                                           number_channel_out_conv=number_channel_out_conv_short_network,
                                           bottleneck_channels=bottleneck_channels_short_network,
                                           kernel_sizes=kernel_sizes_short_network,
                                           use_residuals=use_residuals_short_network,
                                           num_pred_classes=num_pred_classes,
                                           activation_function=activation_function)

        self._hidden_layer = nn.Linear(number_features_long + number_channel_out_conv_short_network * 4, 8)
        self._dropout = nn.Dropout(p=0.5)
        self._batch_norm2 = nn.BatchNorm1d(8)
        self._activation = activation_function()
        self._output_layer = nn.Linear(8, num_pred_classes)

        print(self)
        print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def load_weights_short_network_and_freeze(self, state_dict):
        self._short_network.load_state_dict(state_dict=state_dict)
        for param in self._short_network.parameters():
            param.requires_grad = False
        print("Number Parameters after freeze: ", self.get_n_params())

    def forward(self, input):
        # The features are the same for every short timeindex (hence 0)
        x_features_long = input[:, 0, 0:self._number_features_long]
        x_short = input[:, :, self._number_features_long::]
        x_features_short = self._short_network(x_short, get_features=True)

        features_concatenated = self._activation(
            self._batch_norm2(self._hidden_layer(torch.cat([x_features_long, x_features_short], 1))))
        features_concatenated = self._dropout(features_concatenated)
        output = self._output_layer(features_concatenated)
        return output
