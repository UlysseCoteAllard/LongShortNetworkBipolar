from random import shuffle
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler, Dataset


class BucketDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return torch.FloatTensor([self.inputs[index]])[0], self.targets[index]


class BucketBatchSampler(Sampler):
    # want inputs to be an array
    def __init__(self, batch_size, inputs, targets):
        # Throw an error if the number of inputs and targets don't match
        if targets is not None:
            if len(inputs) != len(targets):
                raise Exception("[BucketBatchSampler] inputs and targets have different sizes")
        # Remember batch size
        self.batch_size = batch_size
        # For each data item (it's index), keep track of combination of input and target lengths
        self.ind_n_len = []
        if targets is None:
            for i in range(0, len(inputs)):
                self.ind_n_len.append((i, (inputs[i].shape[0], 1)))
        else:
            for i in range(0, len(inputs)):
                self.ind_n_len.append((i, (inputs[i].shape[0], targets[i].shape[0])))

        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.ind_n_len)
        # Organize lengths, e.g., batch_map[(5,8)] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()  # <-- Could be a waste of performance
        # shuffle all the batches so they aren't ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2, dilation=dilation, groups=groups)
