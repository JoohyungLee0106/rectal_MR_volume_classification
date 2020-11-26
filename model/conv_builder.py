import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_triple = _ntuple(3)

class Conv2DSimple(nn.Conv2d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel,
                 stride=1,
                 **kwargs):

        super(Conv2DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(kernel, kernel),
            stride=stride,
            bias=False)

    def forward(self, input, stride_HW=1, dilation_HW=1):
        return F.conv2d(input, self.weight, self.bias, stride_HW,
                        dilation_HW, dilation_HW, self.groups)

class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel,
                 stride_D=1):
        if type(kernel) is not tuple:
            kernel = _triple(kernel)

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel,
            stride=stride_D,
            padding=1,
            bias=False)
        self.stride_D = stride_D

    def forward(self, input, stride_HW=1, dilation_HW=1):
        return F.conv3d(input, self.weight, self.bias, (self.stride_D, stride_HW, stride_HW),
                        (1, dilation_HW, dilation_HW), (1, dilation_HW, dilation_HW), self.groups)

class Conv3DNoTemporal(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel,
                 stride_D=1,
                 padding=1):

        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, kernel, kernel),
            stride=stride_D,
            padding=(0, 1, 1),
            bias=False)
        self.stride_D = stride_D

    def forward(self, input, stride_HW=1, dilation_HW=1):
        return F.conv3d(input, self.weight, self.bias, (self.stride_D, stride_HW, stride_HW),
                        (0, dilation_HW, dilation_HW), (1, dilation_HW, dilation_HW), self.groups)


class Conv2Plus1D(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel,
                 stride_D=1):
        super(Conv2Plus1D, self).__init__()
        midplanes = (in_planes * out_planes * 3 * 3 * 3) // (in_planes * 3 * 3 + 3 * out_planes)
        self.conv1 = nn.Conv3d(in_planes, midplanes, kernel_size=(1, kernel, kernel),
                  stride=(1, stride_D, stride_D), padding=(0, 1, 1),
                  bias=False)
        self.bn1 = nn.BatchNorm3d(midplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(midplanes, out_planes, kernel_size=(kernel, 1, 1),
                  stride=(stride_D, 1, 1), padding=(1, 0, 0),
                  bias=False)

        self.stride_D = stride_D

    def forward(self, x, stride_HW=1, dilation_HW=1):
        x = self.relu1(self.bn1(F.conv3d(x, self.conv1.weight, self.conv1.bias, (1, stride_HW, stride_HW),
                                (0, dilation_HW, dilation_HW), (1, dilation_HW, dilation_HW), self.conv1.groups)))
        return F.conv3d(x, self.conv2.weight, self.conv2.bias, (self.stride_D, 1, 1), (1, 0, 0), 1, self.conv2.groups)