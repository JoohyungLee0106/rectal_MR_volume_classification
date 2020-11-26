import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .encoders import BasicBlock as BasicBlock
from .encoders import Bottleneck as Bottleneck
from .conv_builder import Conv2DSimple as Conv2DSimple
from .conv_builder import Conv3DSimple as Conv3DSimple
from .conv_builder import Conv3DNoTemporal as Conv3DNoTemporal
from .conv_builder import Conv2Plus1D as Conv2Plus1D


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class fcn_block(nn.Module):
    def __init__(self, feature_dim_skip, feature_dim_toptobot, conv_builder = Conv3DSimple, feature_dim_out = 64,
                 scale_factor_depth = 2, if_deconv=False):
        super(fcn_block, self).__init__()
        # assert if_deconv
        if scale_factor_depth == 1:
            deconv_kernel_size_3d = (1,4,4)
        elif scale_factor_depth == 2:
            deconv_kernel_size_3d = (4,4,4)
        else:
            raise ValueError("<fcn_block> Invalid deconv_kernel_size !!!")
        if conv_builder == Conv2DSimple:
            self.conv_skip = nn.Conv2d(feature_dim_skip, feature_dim_out, 1)
            if if_deconv:
                self.upsample = nn.ConvTranspose2d(feature_dim_out, feature_dim_out, 4, stride=2, padding=1, bias=False)
            else:
                self.upsample = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners =True)
        else:
            self.conv_skip = nn.Conv3d(feature_dim_skip, feature_dim_out, 1)
            if if_deconv:
                self.upsample = nn.ConvTranspose3d(feature_dim_out, feature_dim_out, deconv_kernel_size_3d, stride=(int(scale_factor_depth),2,2), padding=((scale_factor_depth-1),1,1), bias=False)
            else:
                self.upsample = nn.Upsample(scale_factor=(float(scale_factor_depth),2.0,2.0), mode='trilinear', align_corners =True)
        # self.conv_toptobot = conv_builder(feature_dim_toptobot, feature_dim_out, 3, 1)

    def forward(self, skip, top_to_bot):
        skip = self.conv_skip(skip)
        # print(f'<unet> skip: {skip.size()}')
        # print(f'<unet> top_to_bot 1: {top_to_bot.size()}')
        top_to_bot = self.upsample(top_to_bot)
        # print(f'<unet> top_to_bot 2: {top_to_bot.size()}')
        return skip + top_to_bot

class unet_block(nn.Module):
    def __init__(self, feature_dim_skip, feature_dim_toptobot, conv_builder = Conv3DSimple, feature_dim_out = 128,
                 scale_factor_depth = 2, if_deconv=False):
        super(unet_block, self).__init__()
        # assert int(feature_dim_toptobot/2) == feature_dim_skip
        assert feature_dim_out == feature_dim_skip
        # assert if_deconv
        if scale_factor_depth == 1:
            deconv_kernel_size_3d = (1,4,4)
        elif scale_factor_depth == 2:
            deconv_kernel_size_3d = (4,4,4)
        else:
            raise ValueError("<unet_block> Invalid deconv_kernel_size !!!")
        # print(f'conv_builder: {conv_builder}')
        if conv_builder == Conv2DSimple:
            print(f'conv_builder = 2d')
            if if_deconv:
                self.upsample = nn.ConvTranspose2d(feature_dim_toptobot, feature_dim_skip, 4, stride=2, padding=1, bias=False)
                feature_dim_conv_in = feature_dim_skip * 2
            else:
                self.upsample = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners =True)
                feature_dim_conv_in = feature_dim_toptobot + feature_dim_skip
            bn_builder = nn.BatchNorm2d
        else:
            print(f'conv_builder = 3d')
            if if_deconv:
                self.upsample = nn.ConvTranspose3d(feature_dim_toptobot, feature_dim_skip, deconv_kernel_size_3d, stride=(int(scale_factor_depth),2,2), padding=((scale_factor_depth-1),1,1), bias=False)
                feature_dim_conv_in = feature_dim_skip*2
            else:
                self.upsample = nn.Upsample(scale_factor=(float(scale_factor_depth),2.0,2.0), mode='trilinear', align_corners =True)
                feature_dim_conv_in = feature_dim_toptobot + feature_dim_skip
            bn_builder = nn.BatchNorm3d

        self.conv_block = nn.Sequential(conv_builder(feature_dim_conv_in, feature_dim_skip, 3),
                                        bn_builder(feature_dim_skip), nn.ReLU(),
                                        conv_builder(feature_dim_skip, feature_dim_skip, 3),
                                        bn_builder(feature_dim_skip), nn.ReLU()
                                        )

    def forward(self, skip, top_to_bot):
        # print(f'<unet> skip: {skip.size()}')
        # print(f'<unet> top_to_bot 1: {top_to_bot.size()}')
        top_to_bot = self.upsample(top_to_bot)
        # print(f'<unet> top_to_bot 2: {top_to_bot.size()}')
        x = torch.cat((skip, top_to_bot), dim=1)

        return self.conv_block(x)


class decoder(nn.Module):
    def __init__(self, decoder_type='unet', conv_builder = [Conv3DSimple]*3, scale_factor_depth = [1,2,2], if_deconv=True, starting_feature_num=32):
        super(decoder, self).__init__()
        # feature_dim_in = [32,32,64,128]
        feature_dim_in = [starting_feature_num, starting_feature_num, starting_feature_num*2, starting_feature_num*4]
        if decoder_type == 'unet':
            feature_dim_out = [starting_feature_num,starting_feature_num,starting_feature_num*2]
            block = unet_block
        elif decoder_type == 'fcn':
            feature_dim_out = [starting_feature_num*2]*3
            block = fcn_block
        else:
            raise ValueError('<decoder> Wrong argument: type')
        #
        if Conv2DSimple in conv_builder:
            self.layer0_1 = block(feature_dim_skip=feature_dim_in[0], feature_dim_toptobot=feature_dim_in[1],
                                  conv_builder=conv_builder[0], feature_dim_out=feature_dim_out[0],
                                  scale_factor_depth=scale_factor_depth[0], if_deconv=if_deconv)

        self.layer1_2 = block(feature_dim_skip=feature_dim_in[1], feature_dim_toptobot=feature_dim_in[2],
                              conv_builder=conv_builder[1], feature_dim_out=feature_dim_out[1],
                              scale_factor_depth=scale_factor_depth[1], if_deconv=if_deconv)

        self.layer2_3 = block(feature_dim_skip=feature_dim_in[2], feature_dim_toptobot=feature_dim_in[3],
                              conv_builder=conv_builder[2], feature_dim_out=feature_dim_out[2],
                              scale_factor_depth=scale_factor_depth[2], if_deconv=if_deconv)

        self._initialize_weights()

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

        for key in self.state_dict():
            if key.split(".")[-1] == 'bias':
                self.state_dict()[key][...] = 0

