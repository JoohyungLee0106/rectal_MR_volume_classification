import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from .conv_builder import Conv2DSimple as Conv2DSimple
from .conv_builder import Conv3DSimple as Conv3DSimple
from .conv_builder import Conv3DNoTemporal as Conv3DNoTemporal
from .conv_builder import Conv2Plus1D as Conv2Plus1D
from .sequential import (Sequential_downsample, Custom_sequential)
from .self_attention_module import (CBAM, SE, NonLocalBlock2D, NonLocalBlock3D, NonLocalBlockND)
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['Adaptive_partial_AvgPool', 'ResNet', 'encoder', 'BasicBlock', 'Bottleneck']

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}

class Adaptive_partial_AvgPool(nn.Module):
    '''
    Input must consist of non-negative numbers
    '''
    def __init__(self, image_dim=2, threshold=0):
        super(Adaptive_partial_AvgPool, self).__init__()
        self.threshold = threshold
        self.image_dim = image_dim

    def forward(self, x, keepdims=True):
        xx=x.clone()
        xx = F.relu(xx.sign())
        # xx[xx>self.threshold] = 1.0
        xx=torch.sum(xx, list(range(2, self.image_dim+2)), keepdim=keepdims, dtype=torch.float)
        xx = xx.detach()
        x=torch.sum(x, list(range(2, self.image_dim+2)), keepdim=keepdims, dtype=torch.float)

        return torch.div(x, xx)

class Adaptive_partial_AvgPool3d(nn.Module):
    '''
    Input must consist of non-negative numbers
    '''
    def __init__(self):
        super(Adaptive_partial_AvgPool3d, self).__init__()
        # self.threshold = threshold
        # print('Adaptive_partial_AvgPool3d init !!!')
    def forward(self, x):
        # print('Adaptive_partial_AvgPool3d forward!!!')
        xx=x.clone()
        xx = F.relu(xx.sign())
        # xx.retain_grad()
        # xx[xx>self.threshold] = 1.0
        xx=torch.sum(xx, [2,3,4], keepdim=True, dtype=torch.float)
        xx = xx.detach()
        x=torch.sum(x, [2,3,4], keepdim=True, dtype=torch.float)

        return torch.div(x, xx)

# class Conv2DSimple(nn.Conv2d):
#     def __init__(self,
#                  in_planes,
#                  out_planes,
#                  kernel,
#                  stride=1,
#                  **kwargs):
#
#         super(Conv2DSimple, self).__init__(
#             in_channels=in_planes,
#             out_channels=out_planes,
#             kernel_size=(kernel, kernel),
#             stride=stride,
#             bias=False)
#
#     def forward(self, input, stride_HW=1, dilation_HW=1):
#         return F.conv2d(input, self.weight, self.bias, stride_HW,
#                         dilation_HW, dilation_HW, self.groups)

class Conv2D_downsample(nn.Conv2d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 stride=1):
        super(Conv2D_downsample, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False)

    def forward(self, input, stride_HW=1):
        return F.conv2d(input, self.weight, self.bias, stride_HW)

class Conv3D_downsample(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 stride_D=1):
        super(Conv3D_downsample, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=1,
            stride=stride_D,
            padding=0,
            bias=False)
        self.stride_D = stride_D
        # print(f'init self.stride_D: {self.stride_D}')

    def forward(self, input, stride_HW=1):
        # print(f'Downsample stride_HW: {stride_HW}')
        # print(f'weight size: {self.weight.size()}')
        # print(f'FOWARD self.stride_D: {self.stride_D}')
        return F.conv3d(input, self.weight, self.bias, (self.stride_D, stride_HW, stride_HW))

# class Conv3DSimple(nn.Conv3d):
#     def __init__(self,
#                  in_planes,
#                  out_planes,
#                  kernel,
#                  stride_D=1):
#
#         super(Conv3DSimple, self).__init__(
#             in_channels=in_planes,
#             out_channels=out_planes,
#             kernel_size=(kernel, kernel, kernel),
#             stride=stride_D,
#             padding=1,
#             bias=False)
#         self.stride_D = stride_D
#
#     def forward(self, input, stride_HW=1, dilation_HW=1):
#         return F.conv3d(input, self.weight, self.bias, (self.stride_D, stride_HW, stride_HW),
#                         (1, dilation_HW, dilation_HW), (1, dilation_HW, dilation_HW), self.groups)
#
# class Conv3DNoTemporal(nn.Conv3d):
#
#     def __init__(self,
#                  in_planes,
#                  out_planes,
#                  kernel,
#                  stride_D=1,
#                  padding=1):
#
#         super(Conv3DNoTemporal, self).__init__(
#             in_channels=in_planes,
#             out_channels=out_planes,
#             kernel_size=(1, kernel, kernel),
#             stride=stride_D,
#             padding=(0, 1, 1),
#             bias=False)
#         self.stride_D = stride_D
#
#     def forward(self, input, stride_HW=1, dilation_HW=1):
#         return F.conv3d(input, self.weight, self.bias, (self.stride_D, stride_HW, stride_HW),
#                         (0, dilation_HW, dilation_HW), (1, dilation_HW, dilation_HW), self.groups)
#
#
# class Conv2Plus1D(nn.Module):
#
#     def __init__(self,
#                  in_planes,
#                  out_planes,
#                  kernel,
#                  stride_D=1):
#         super(Conv2Plus1D, self).__init__()
#         midplanes = (in_planes * out_planes * 3 * 3 * 3) // (in_planes * 3 * 3 + 3 * out_planes)
#         self.conv1 = nn.Conv3d(in_planes, midplanes, kernel_size=(1, kernel, kernel),
#                   stride=(1, stride_D, stride_D), padding=(0, 1, 1),
#                   bias=False)
#         self.bn1 = nn.BatchNorm3d(midplanes)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv3d(midplanes, out_planes, kernel_size=(kernel, 1, 1),
#                   stride=(stride_D, 1, 1), padding=(1, 0, 0),
#                   bias=False)
#
#         self.stride_D = stride_D
#
#     def forward(self, x, stride_HW=1, dilation_HW=1):
#         x = self.relu1(self.bn1(F.conv3d(x, self.conv1.weight, self.conv1.bias, (1, stride_HW, stride_HW),
#                                 (0, dilation_HW, dilation_HW), (1, dilation_HW, dilation_HW), self.conv1.groups)))
#         return F.conv3d(x, self.conv2.weight, self.conv2.bias, (self.stride_D, 1, 1), (1, 0, 0), 1, self.conv2.groups)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, att_type = False, stride_D1=1, downsample=None):
        '''
        :param inplanes:
        :param planes:
        :param conv_builder: conv operation unit, e.g., Conv3DSimple.
        :param stride:
        :param downsample:
        '''
        super(BasicBlock, self).__init__()
        # midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)
        if conv_builder == Conv2DSimple:
            bn = nn.BatchNorm2d
            dimension = '2d'
        elif conv_builder == Conv3DSimple or conv_builder == Conv2Plus1D or conv_builder == Conv3DNoTemporal:
            bn = nn.BatchNorm3d
            dimension = '3d'
        else:
            raise ValueError("<class BasicBlock> Invalid argument: conv_builder")

        self.conv1 = conv_builder(inplanes, planes, 3, stride_D1)
        self.bn1 = bn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_builder(planes, planes, 3)
        self.bn2 = bn(planes)
        # self.results={}

        self.downsample = downsample

        if att_type == 'CBAM':
            self.att_type = CBAM(planes, dimension=dimension, conv_builder=conv_builder)
        elif att_type == 'SE':
            self.att_type = SE(planes, dimension=dimension)
        elif (att_type == 'NL') or (att_type == False):
            self.att_type = False
        else:
            raise ValueError('<BasicBLock> Wrong setting!!!')

    def forward(self, x, stride_HW=1, dilation_HW=[1,1]):
        residual = x
        # print('=====================================')
        # print(f'input: {x.size()}')
        out = self.relu(self.bn1(self.conv1(x, stride_HW= stride_HW, dilation_HW=dilation_HW[0])))
        # print(f'after first conv: {out.size()}')
        out = self.bn2(self.conv2(out, dilation_HW=dilation_HW[1]))
        # print(f'after second conv: {out.size()}')
        if x.size() != out.size():
            # print(f'Residual added!!! STRIDE: {int(x.size(3)/out.size(3))}')
            residual = self.downsample(x, stride_HW=int(x.size(3)/out.size(3)))
            # print(f'residual size: {residual.size()}')

        if self.att_type:
            out = self.att_type(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, att_type = False, stride_D1=1, downsample=None):
        '''
        :param inplanes:
        :param planes:
        :param conv_builder: conv operation unit, e.g., Conv3DSimple.
        :param stride:
        :param downsample:
        '''
        super(Bottleneck, self).__init__()
        # midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)
        if conv_builder == Conv2DSimple:
            bn = nn.BatchNorm2d
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
            self.conv3 = nn.Conv2d(planes, planes* self.expansion, kernel_size=1, stride=1, bias=False)
            dimension = '2d'
        elif conv_builder == Conv3DSimple or conv_builder == Conv2Plus1D or conv_builder == Conv3DNoTemporal:
            bn = nn.BatchNorm3d
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
            self.conv3 = nn.Conv3d(planes, planes* self.expansion, kernel_size=1, stride=1, bias=False)
            dimension = '3d'
        else:
            raise ValueError("<class Bottleneck> Invalid argument: conv_builder")

        self.bn1 = bn(planes)
        self.conv2 = conv_builder(planes, planes, 3, stride_D1)
        self.bn2 = bn(planes)
        self.bn3 = bn(planes* self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.results={}

        self.downsample = downsample

        if att_type == 'CBAM':
            self.att_type = CBAM(planes, dimension=dimension, conv_builder=conv_builder)
        elif att_type == 'SE':
            self.att_type = SE(planes, dimension=dimension)
        elif (att_type == 'NL') or (att_type == False):
            self.att_type = False
        else:
            raise ValueError('<Bottleneck> Wrong setting!!!')

    def forward(self, x, stride_HW=1, dilation_HW=[1,1]):
        residual = x
        # print('=====================================')
        # print(f'input: {x.size()}')
        out = self.relu(self.bn1(self.conv1(x)))
        # print(f'after first conv: {out.size()}')
        out = self.relu(self.bn2(self.conv2(out, stride_HW= stride_HW, dilation_HW=dilation_HW[0])))
        # print(f'after second conv: {out.size()}')
        out = self.bn3(self.conv3(out))
        if x.size() != out.size():
            # print(f'Residual added!!! STRIDE: {int(x.size(3)/out.size(3))}')
            residual = self.downsample(x, stride_HW=int(x.size(3)/out.size(3)))
            # print(f'residual size: {residual.size()}')

        if self.att_type:
            out = self.att_type(out)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self, input_channel_num=1, starting_feature_num=32, dimension = '3d'):
        if dimension == '3d':
            print(f'<stem> 3d activated !!!!')
            super(BasicStem, self).__init__(
                nn.Conv3d(input_channel_num, starting_feature_num, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                          padding=(1, 3, 3), bias=False, padding_mode='replicate'),
                nn.BatchNorm3d(starting_feature_num),
                nn.ReLU(inplace=True))
        elif dimension == '2.5d':
            print(f'<stem> 2.5d activated !!!!')
            super(BasicStem, self).__init__(
                nn.Conv3d(input_channel_num, starting_feature_num, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                          padding=(0, 3, 3), bias=False, padding_mode='replicate'),
                nn.BatchNorm3d(starting_feature_num),
                nn.ReLU(inplace=True))
        elif dimension == '2d':
            print(f'<stem> 2d activated !!!!')
            super(BasicStem, self).__init__(
                nn.Conv2d(input_channel_num, starting_feature_num, kernel_size=(7, 7), stride=(2, 2),
                          padding=(3, 3), bias=False, padding_mode='replicate'),
                nn.BatchNorm2d(starting_feature_num),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        elif dimension == '2d-att':
            print(f'<stem> 2d activated !!!!')
            super(BasicStem, self).__init__(
                nn.Conv2d(input_channel_num, starting_feature_num, kernel_size=(7, 7), stride=(2, 2),
                          padding=(3, 3), bias=False, padding_mode='replicate'),
                nn.BatchNorm2d(starting_feature_num),
                nn.ReLU(inplace=True))
        else:
            raise ValueError('<class BasicStem> Invalid argument!! : dimension')


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self, input_channel_num=1, starting_feature_num=32, dimension = '3d'):
        midplanes = (input_channel_num * starting_feature_num * 3 * 7 * 7) // (input_channel_num * 7 * 7 + 3 * starting_feature_num)
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(input_channel_num, midplanes, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, starting_feature_num, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(starting_feature_num),
            nn.ReLU(inplace=True))


class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, conv_makers=[Conv3DSimple]*4, block_num=[3, 4, 6, 3],
                 starting_feature_num=32, num_classes=1, pool2 = nn.AdaptiveAvgPool3d((1, 1, 1)),
                 zero_init_residual=False, att_type = [False]*4, if_fc = True, stride_D1=[1,2,2,2],
                 stem=None, stem_dimension = '3d', fc_ratio=8):
        """Generic resnet video generator.
        pool2 == nn.AdaptiveAvgPool3d((1, 1, 1)) or nn.AdaptiveAvgPool2d((1, 1)) or Adaptive_partial_AvgPool(image_dim=3)
        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(ResNet, self).__init__()

        assert isinstance(pool2, nn.AdaptiveAvgPool3d) or isinstance(pool2, nn.AdaptiveAvgPool2d)\
               or isinstance(pool2, Adaptive_partial_AvgPool) or isinstance(pool2, Adaptive_partial_AvgPool3d)

        print('ResNet starting feature number: '+str(starting_feature_num))
        self.stem = stem(starting_feature_num=starting_feature_num, dimension = stem_dimension)
        # self.return_att_map_off()

        self.pool2 = pool2
        self.att_type = att_type
        self.conv_makers = conv_makers
        self.return_att_map = False
        self.att_loc = []
        for att in att_type:
            if (att == 'CBAM') or (att == 'SE'):
                self.att_loc.append(-1)
            elif att == 'NL':
                self.att_loc.append(-2)
            elif att == False:
                self.att_loc.append(0)

        self.inplanes = starting_feature_num

        # self.layer1 = self._make_layer(block, conv_makers[0], starting_feature_num, block_num[0], stride_D1=stride_D1[0], att_type=att_type[0], layer_num=1),
        # self.layer2 = self._make_layer(block, conv_makers[1], starting_feature_num*2, block_num[1], stride_D1=stride_D1[1], att_type=att_type[1], layer_num=2),
        # self.layer3 = self._make_layer(block, conv_makers[2], starting_feature_num*4, block_num[2], stride_D1=stride_D1[2], att_type=att_type[2], layer_num=3),
        # self.layer4 = self._make_layer(block, conv_makers[3], starting_feature_num*8, block_num[3], stride_D1=stride_D1[3], att_type=att_type[3], layer_num=4),
        self.layers = nn.ModuleList([
            self._make_layer(block, conv_makers[0], starting_feature_num, block_num[0], stride_D1=stride_D1[0], att_type=att_type[0], layer_num=1),
            self._make_layer(block, conv_makers[1], starting_feature_num*2, block_num[1], stride_D1=stride_D1[1], att_type=att_type[1], layer_num=2),
            self._make_layer(block, conv_makers[2], starting_feature_num*4, block_num[2], stride_D1=stride_D1[2], att_type=att_type[2], layer_num=3),
            self._make_layer(block, conv_makers[3], starting_feature_num*8, block_num[3], stride_D1=stride_D1[3], att_type=att_type[3], layer_num=4),
        ])

        if if_fc:
            self.fc = nn.Linear(fc_ratio * starting_feature_num * block.expansion, num_classes)
        # print(f'fc: {self.fc}')
        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    # def if_return_att_map(self, if_return_att_map):
    #     self.return_att_map = if_return_att_map
    #     assert type(if_return_att_map) == bool
    #     # for i in range(len(self.att_type)):
    #     if self.att_type[0] == 'CBAM':
    #         self.layer1[self.att_loc[0]].att_type.return_att_map = if_return_att_map
    #     if self.att_type[1] == 'CBAM':
    #         self.layer2[self.att_loc[1]].att_type.return_att_map = if_return_att_map
    #     elif self.att_type[1] == 'NL':
    #         self.layer2[-2].return_att_map = if_return_att_map
    #     if self.att_type[2] == 'CBAM':
    #         self.layer3[self.att_loc[2]].att_type.return_att_map = if_return_att_map
    #     elif self.att_type[2] == 'NL':
    #         self.layer3[-2].return_att_map = if_return_att_map
    #     if self.att_type[3] == 'CBAM':
    #         self.layer4[self.att_loc[3]].att_type.return_att_map = if_return_att_map

    # def att_collector(self):
    #     att_map = {}
    #     if self.att_type[0]:
    #         att_map['layer1'] = self.layer1.att_map
    #     if self.att_type[1]:
    #         att_map['layer2'] = self.layer2.att_map
    #     if self.att_type[2]:
    #         att_map['layer3'] = self.layer3.att_map
    #     if self.att_type[3]:
    #         att_map['layer4'] = self.layer4.att_map
    #     return att_map

    # def att_collector(self):
    #     att_map = {}
    #     for i in range(len(self.att_type)):
    #         if self.att_type[i]:
    #             att_map['layer'+str(i+1)] = self.layers[i].att_map
    #     return att_map

    def if_return_att_map(self, if_return_att_map):
        assert type(if_return_att_map) == bool
        self.return_att_map = if_return_att_map
        for i in range(len(self.att_type)):
            if (self.att_type[i] == 'CBAM') or (self.att_type[i] == 'SE'):
                self.layers[i][self.att_loc[i]].att_type.return_att_map = if_return_att_map
            elif self.att_type[i] == 'NL':
                # print(self.layers[i])
                self.layers[i][self.att_loc[i]].return_att_map = if_return_att_map
            # else:
            #     print('see?')
            #     raise ValueError('<class ResNet> if_return_att_map')

    def att_collector(self):
        att_map = {}
        for i in range(len(self.att_type)):
            if (self.att_type[i] == 'CBAM') or (self.att_type[i] == 'SE'):
                att_map['layer'+str(i+1)] = self.layers[i][self.att_loc[i]].att_type.att_map
            elif self.att_type[i] == 'NL':
                att_map['layer'+str(i+1)] = self.layers[i][self.att_loc[i]].att_map
        return att_map

    def clear_att_map(self):
        for i in range(len(self.att_type)):
            if (self.att_type[i] == 'CBAM') or (self.att_type[i] == 'SE'):
                self.layers[i][self.att_loc[i]].att_type.att_map = 0
            elif self.att_type[i] == 'NL':
                self.layers[i][self.att_loc[i]].att_map = 0
            # if self.att_type[i]:
            #     self.layer[i].att_map = 0

    def forward(self, x):
        x = self.stem(x)

        x = self.layers[0](x, stride_HW=1, dilation_HW=[1, 1])
        x = self.layers[1](x, stride_HW=2, dilation_HW=[1, 1])
        x = self.layers[2](x, stride_HW=2, dilation_HW=[1, 1])
        x = self.layers[3](x, stride_HW=2, dilation_HW=[1, 1])

        x = self.pool2(x)

        x = x.flatten(1)
        x = self.fc(x)
        x = x.view(x.size(0))
        out = {'category': x}
        if self.return_att_map:
            out.update(self.att_collector())
        return out


    def _make_layer(self, block, conv_builder, planes, block_num, stride_D1=1, att_type=False, layer_num=0):
        downsample = None

        if conv_builder == Conv2DSimple:
            conv_ds = Conv2D_downsample
            bn = nn.BatchNorm2d
        elif conv_builder == Conv3DSimple or conv_builder == Conv2Plus1D or conv_builder == Conv3DNoTemporal:
            conv_ds = Conv3D_downsample
            bn = nn.BatchNorm3d
        else:
            raise ValueError('<class> ResNet: Invalid argument: conv_makers')

        if stride_D1 != 1 or self.inplanes != planes * block.expansion:
            # print(f'IF ACTIVATED!!! stride_D1: {stride_D1}')
            # ds_stride = conv_builder.get_downsample_stride(stride)isinstance(pool2, nn.AdaptiveAvgPool3d)
            downsample = Sequential_downsample(
                conv_ds(self.inplanes, planes * block.expansion, stride_D1),
                bn(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, att_type, stride_D1, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, block_num):
            layers.append(block(self.inplanes, planes, conv_builder, att_type))

        if att_type == 'NL':
            # if 3d
            if isinstance(self.pool2, nn.AdaptiveAvgPool3d) or isinstance(self.pool2, Adaptive_partial_AvgPool3d):
                layers.insert(-1, NonLocalBlock3D(planes))
            # if 2d
            elif isinstance(self.pool2, nn.AdaptiveAvgPool2d) or isinstance(self.pool2, Adaptive_partial_AvgPool):
                layers.insert(-1, NonLocalBlock2D(planes))
        elif (att_type == 'CBAM') or (att_type == 'SE'):
            layers[-1].att_type.return_att_map = True

        return Custom_sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

        for key in self.state_dict():
            if key.split(".")[-1] == 'bias':
                self.state_dict()[key][...] = 0


def _resnet(arch, pretrained=False, progress=True, **kwargs):
    model = ResNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)


        state_dict['stem.0.weight'] = torch.sum(state_dict["stem.0.weight"], dim=1, keepdim=True)
        # state_dict["layer1.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer1.0.downsample.0.weight")
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        # state_dict["layer2.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer2.0.downsample.0.weight")
        # state_dict["layer3.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer3.0.downsample.0.weight")
        # state_dict["layer4.0.downsample.0.conv_downsample.weight"] = state_dict.pop("layer4.0.downsample.0.weight")

        model.load_state_dict(state_dict, strict=False)
    return model

def encoder(resnet_type = 34, encoder_dim_type = '2d', att_type = [False]*4, stride_D1=[1, 2, 2, 2],
            starting_feature_num=32, num_classes=1):
    assert resnet_type == 18 or resnet_type == 34 or resnet_type == 50
    for att in att_type:
        assert att == False or att == 'CBAM' or att == 'SE' or att == 'NL'
    assert encoder_dim_type == '2d' or encoder_dim_type == 'r3d' or encoder_dim_type == '2plus1d' or encoder_dim_type == 'rmc6' or\
           encoder_dim_type == 'rmc2' or encoder_dim_type == 'rmc3' or encoder_dim_type == 'rmc4' or encoder_dim_type == 'rmc5' or \
           encoder_dim_type == 'mc2' or encoder_dim_type == 'mc3' or encoder_dim_type == 'mc4' or encoder_dim_type == 'mc5'

    if resnet_type == 18:
        block = BasicBlock
        block_num = [2, 2, 2, 2]
    elif resnet_type == 34:
        block = BasicBlock
        block_num = [3, 4, 6, 3]
    elif resnet_type == 50:
        block = Bottleneck
        block_num = [3, 4, 6, 3]
    else:
        raise ValueError("<encoder> Wrong 'resnet_type' !!!")

    if encoder_dim_type == '2d':
        stem = BasicStem
        stem_dimension = '2d'
        conv_makers = [Conv2DSimple] * 4
        pool2 = nn.AdaptiveAvgPool2d((1, 1))
    elif encoder_dim_type == 'r3d':
        stem = BasicStem
        stem_dimension = '3d'
        conv_makers = [Conv3DSimple] * 4
        pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
    elif encoder_dim_type == 'rmc2':
        stem = BasicStem
        stem_dimension = '2.5d'
        conv_makers = [Conv3DSimple] * 4
        pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
    elif encoder_dim_type == 'rmc3':
        stem = BasicStem
        stem_dimension = '2.5d'
        conv_makers = [Conv3DNoTemporal] + [Conv3DSimple] * 3
        pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
    elif encoder_dim_type == 'rmc4':
        stem = BasicStem
        stem_dimension = '2.5d'
        conv_makers = [Conv3DNoTemporal]*2 + [Conv3DSimple] * 2
        pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
    elif encoder_dim_type == 'rmc5':
        stem = BasicStem
        stem_dimension = '2.5d'
        conv_makers = [Conv3DNoTemporal]*3 + [Conv3DSimple]
        pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
    elif encoder_dim_type == 'rmc6':
        stem = BasicStem
        stem_dimension = '2.5d'
        conv_makers = [Conv3DNoTemporal]*4
        pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
    elif encoder_dim_type == 'mc2':
        stem = BasicStem
        stem_dimension = '3d'
        conv_makers = [Conv3DNoTemporal] * 4
        pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
    elif encoder_dim_type == 'mc3':
        stem = BasicStem
        stem_dimension = '3d'
        conv_makers = [Conv3DSimple] + [Conv3DNoTemporal] * 3
        pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
    elif encoder_dim_type == 'mc4':
        stem = BasicStem
        stem_dimension = '3d'
        conv_makers = [Conv3DSimple] * 2 + [Conv3DNoTemporal] * 2
        pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
    elif encoder_dim_type == 'mc5':
        stem = BasicStem
        stem_dimension = '3d'
        conv_makers = [Conv3DSimple] * 3 + [Conv3DNoTemporal]
        pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
    elif encoder_dim_type == '2plus1d':
        stem = R2Plus1dStem
        stem_dimension = '3d'
        conv_makers = [Conv2Plus1D] * 4
        pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        raise ValueError("<encoder> Wrong 'encoder_dim_type' !!!")


    return ResNet(block=block,
                  conv_makers=conv_makers,
                  block_num=block_num,
                  starting_feature_num=starting_feature_num,
                  pool2=pool2,
                  att_type=att_type,
                  stride_D1=stride_D1,
                  stem=stem,
                  stem_dimension=stem_dimension,
                  num_classes=num_classes)


#
# #
#
# model = resnet_2d(block=BasicBlock)
# # print(f'model dict: {a.state_dict().keys()}')
#
# x = torch.randn(2, 1, 256,256)
# y=model(x)
# print(y['category'].size())