import torch
import torch.nn as nn

class Sequential_downsample(nn.Sequential):
    def forward(self, input, stride_HW=1):
        input = self[0](input, stride_HW=stride_HW)

        for module in self[1:]:
            input = module(input)
        return input

class Custom_sequential(nn.Sequential):
    # store only one or the last(if multiple)att map per layer(instance of this class)
    def __init__(self, *args):
        super(Custom_sequential, self).__init__(*args)

    def forward(self, input, stride_HW=2, dilation_HW=[1, 1]):
        input = self[0](input, stride_HW=stride_HW, dilation_HW=[dilation_HW[0], dilation_HW[1]])

        for module in self[1:]:
            input = module(input, stride_HW=1, dilation_HW=[dilation_HW[1], dilation_HW[1]])

        return input