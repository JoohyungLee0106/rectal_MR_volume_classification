# LAST updated: 20201109 12:30
import torch
from .conv_builder import Conv2DSimple as Conv2DSimple
from .conv_builder import Conv3DSimple as Conv3DSimple
from .conv_builder import Conv3DNoTemporal as Conv3DNoTemporal
from .conv_builder import Conv2Plus1D as Conv2Plus1D
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, conv_builder = Conv2DSimple, padding=0, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

        if conv_builder == Conv2DSimple:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        elif conv_builder == Conv3DSimple:
            self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=(3, kernel_size, kernel_size), stride=1,
                                  padding=(1, padding, padding), bias=False)
        elif conv_builder == Conv3DNoTemporal:
            self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=(1, kernel_size, kernel_size), stride=1,
                                  padding=(0, padding, padding), bias=False)
        else:
            raise ValueError("<BasicConv> Wrong conv_builder")

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, dimension = '2d'):
        '''
        pool_types=['avg', 'max']
        '''
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        if dimension == '2d':
            self.pool = [nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)]
            self.unsqueeze = self.unsqueeze_2d
        elif dimension == '3d':
            self.pool = [nn.AdaptiveAvgPool3d(1), nn.AdaptiveMaxPool3d(1)]
            self.unsqueeze = self.unsqueeze_3d

    def unsqueeze_2d(self, x):
        return x.unsqueeze(2).unsqueeze(3)

    def unsqueeze_3d(self, x):
        return x.unsqueeze(2).unsqueeze(3).unsqueeze(4)

    def forward(self, x):
        channel_att_avg = self.mlp(self.pool[0](x))
        channel_att_max = self.mlp(self.pool[1](x))
        channel_att_sum = channel_att_avg + channel_att_max

        scale = (self.unsqueeze(torch.sigmoid( channel_att_sum ))).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self, conv_builder=Conv2DSimple):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, conv_builder=conv_builder,
                                 padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale, scale

class CBAM(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, dimension = '2d', conv_builder = Conv2DSimple, no_spatial=False):
        '''
        pool_types=['avg', 'max']
        '''
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, dimension = dimension)
        self.no_spatial=no_spatial
        self.att_map = 0

        self.return_att_map = False
        if not no_spatial:
            self.SpatialGate = SpatialGate(conv_builder=conv_builder)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out, scale = self.SpatialGate(x_out)
        if self.return_att_map:
            # self.att_map = scale.clone().detach().half().squeeze(1)
            self.att_map = scale.clone().detach().float().squeeze(1)

        return x_out


class SE(nn.Module):
    def __init__(self, channel, reduction=16, dimension = '2d'):
        super(SE, self).__init__()
        self.att_map = 0
        self.return_att_map = False
        if dimension == '2d':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.unsqueeze = self.unsqueeze_2d
        if dimension == '3d':
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.unsqueeze = self.unsqueeze_3d
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def unsqueeze_2d(self, x):
        return x.unsqueeze(2).unsqueeze(3)

    def unsqueeze_3d(self, x):
        return x.unsqueeze(2).unsqueeze(3).unsqueeze(4)

    def forward(self, x, return_att_map=False):
        b = x.size()[0]
        c = x.size()[1]
        y = self.avg_pool(x).view(b, c)
        y = self.unsqueeze(self.fc(y))
        if self.return_att_map:
            # self.att_map = scale.clone().detach().half().squeeze(1)
            self.att_map = y.clone().detach().float().squeeze(1)
        return x * y.expand_as(x)



class NonLocalBlockND(nn.Module):
    return_att_map = False
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.return_att_map = False
        self.att_map = 0

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)


    def forward(self, x, stride_HW=None, dilation_HW=None):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if self.return_att_map:
            # self.att_map = f_div_C.clone().detach().half().squeeze(1)
            self.att_map = f_div_C.clone().detach().float().squeeze(1)
        return z

def NonLocalBlock2D(in_channels):
    return NonLocalBlockND(in_channels, dimension=2)

def NonLocalBlock3D(in_channels):
    return NonLocalBlockND(in_channels, dimension=3)