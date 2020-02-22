from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import csv
import cxr_dataset as CXR
import auc_calculation

from tiramisu.transition_down import TransitionDown
from tiramisu.dense_block import DenseBlock

from torch.nn import Module, Conv2d, BatchNorm2d, Linear, init
from torch.nn import Sequential
from typing import Optional, Sequence, Union


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FCDenseNet(Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1,
                 initial_num_features: int = 48,
                 dropout: float = 0.2,

                 down_dense_growth_rates: Union[int, Sequence[int]] = 16,
                 down_dense_bottleneck_ratios: Union[Optional[int], Sequence[Optional[int]]] = None,
                 down_dense_num_layers: Union[int, Sequence[int]] = (4, 5, 7, 10, 12),
                 down_transition_compression_factors: Union[float, Sequence[float]] = 1.0,

                 middle_dense_growth_rate: int = 16,
                 middle_dense_bottleneck: Optional[int] = None,
                 middle_dense_num_layers: int = 15,

                 up_dense_growth_rates: Union[int, Sequence[int]] = 16,
                 up_dense_bottleneck_ratios: Union[Optional[int], Sequence[Optional[int]]] = None,
                 up_dense_num_layers: Union[int, Sequence[int]] = (12, 10, 7, 5, 4)):

        super(FCDenseNet, self).__init__()

        # region Parameters handling
        self.in_channels = in_channels
        self.out_channels = out_channels

        if type(down_dense_growth_rates) == int:
            down_dense_growth_rates = (down_dense_growth_rates,) * 5
        if down_dense_bottleneck_ratios is None or type(down_dense_bottleneck_ratios) == int:
            down_dense_bottleneck_ratios = (down_dense_bottleneck_ratios,) * 5
        if type(down_dense_num_layers) == int:
            down_dense_num_layers = (down_dense_num_layers,) * 5
        if type(down_transition_compression_factors) == float:
            down_transition_compression_factors = (down_transition_compression_factors,) * 5

        if type(up_dense_growth_rates) == int:
            up_dense_growth_rates = (up_dense_growth_rates,) * 5
        if up_dense_bottleneck_ratios is None or type(up_dense_bottleneck_ratios) == int:
            up_dense_bottleneck_ratios = (up_dense_bottleneck_ratios,) * 5
        if type(up_dense_num_layers) == int:
            up_dense_num_layers = (up_dense_num_layers,) * 5
        # endregion

        # region First convolution
        self.features = Conv2d(in_channels, initial_num_features, kernel_size=3, padding=1, bias=False)
        current_channels = self.features.out_channels
        # endregion

        # region Downward path
        # Pairs of Dense Blocks with input concatenation and TransitionDown layers
        down_dense_params = [
            {
                'concat_input': True,
                'growth_rate': gr,
                'num_layers': nl,
                'dense_layer_params': {
                    'dropout': dropout,
                    'bottleneck_ratio': br
                    }
                }
            for gr, nl, br in
            zip(down_dense_growth_rates, down_dense_num_layers, down_dense_bottleneck_ratios)
            ]

        down_transition_params = [
            {
                'dropout': dropout,
                'compression': c
                } for c in down_transition_compression_factors
            ]

        # skip_connections_channels = []

        self.down_dense = Module()
        self.down_trans = Module()

        down_pairs_params = zip(down_dense_params, down_transition_params)
        for i, (dense_params, transition_params) in enumerate(down_pairs_params):
            block = DenseBlock(current_channels, **dense_params)
            current_channels = block.out_channels
            self.down_dense.add_module(f'block_{i}', block)

            # skip_connections_channels.append(block.out_channels)

            transition = TransitionDown(current_channels, **transition_params)
            current_channels = transition.out_channels
            self.down_trans.add_module(f'trans_{i}', transition)
        # endregion

        # region Middle block
        # Renamed from "bottleneck" in the paper, to avoid confusion with the Bottleneck of DenseLayers
        self.middle = DenseBlock(
            current_channels,
            middle_dense_growth_rate,
            middle_dense_num_layers,
            concat_input=True,
            dense_layer_params={
                'dropout': dropout,
                'bottleneck_ratio': middle_dense_bottleneck
                })
        current_channels = self.middle.out_channels
        # endregion
        # print('current channels:', current_channels)

        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(current_channels)
        self.final_batch_norm = nn.BatchNorm2d(current_channels)

        # Linear layer
        self.classifier = nn.Linear(current_channels, self.out_channels)

        # region Weight initialization
        for module in self.modules():
            if isinstance(module, Conv2d):
                init.kaiming_normal_(module.weight)
            elif isinstance(module, BatchNorm2d):
                module.reset_parameters()
            elif isinstance(module, Linear):
                init.xavier_uniform_(module.weight)
                init.constant_(module.bias, 0)
        # endregion

    def forward(self, x):
        res = self.features(x)

        skip_tensors = []
        for dense, trans in zip(self.down_dense.children(), self.down_trans.children()):
            res = dense(res)
            skip_tensors.append(res)
            res = trans(res)

        res = self.middle(res)
        res = self.final_batch_norm(res)
        res = torch.nn.functional.relu(res, inplace=True)
        res = torch.nn.functional.adaptive_avg_pool2d(res, (1, 1))
        res = torch.flatten(res, 1)
        res = self.classifier(res)
        return res

# model to use
class FCDenseNet103(FCDenseNet):
    def __init__(self, in_channels=3, out_channels=1000, dropout=0.0):
        super(FCDenseNet103, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            initial_num_features=48,
            dropout=dropout,

            down_dense_growth_rates=16,
            down_dense_bottleneck_ratios=None,
            down_dense_num_layers=(4, 5, 7, 10, 12),
            down_transition_compression_factors=1.0,

            middle_dense_growth_rate=16,
            middle_dense_bottleneck=None,
            middle_dense_num_layers=15,

            up_dense_growth_rates=16,
            up_dense_bottleneck_ratios=None,
            up_dense_num_layers=(12, 10, 7, 5, 4)
        )