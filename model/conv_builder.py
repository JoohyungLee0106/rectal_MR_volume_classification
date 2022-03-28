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

class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 stride=1,
                 kernel_size=3):
        padding = (kernel_size - 1) // 2
        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(1, padding, padding),
            bias=False)


class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 stride=1,
                 kernel_size=3):
        padding = (kernel_size - 1) // 2
        midplanes = (in_planes * out_planes * 3 * 3 * 3) // (in_planes * 3 * 3 + 3 * out_planes)
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, kernel_size, kernel_size),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(kernel_size, 1, 1),
                      stride=(1, 1, 1), padding=(padding, 0, 0),
                      bias=False))


class Conv3DNoTemporal(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 stride=1,
                 kernel_size=3):
        padding = (kernel_size - 1) // 2
        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)
