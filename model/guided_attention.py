import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .encoders import encoder
from .decoders import decoder
from .conv_builder import Conv2DSimple as Conv2DSimple
from .conv_builder import Conv3DSimple as Conv3DSimple
from .conv_builder import Conv3DNoTemporal as Conv3DNoTemporal
from .conv_builder import Conv2Plus1D as Conv2Plus1D
from .sequential import (Sequential_downsample, Custom_sequential)
from .self_attention_module import (CBAM, SE, NonLocalBlock2D, NonLocalBlock3D)
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# encoder:
# resnet_type = 34, encoder_dim_type = '2d', att_type = [False]*4, stride_D1=[1, 2, 2, 2], starting_feature_num=32
# decoder:
# decoder_type='unet', conv_builder = [Conv3DSimple]*3, scale_factor_depth = [1,2,2], if_deconv=True
class Attention(nn.Module):
    def __init__(self, resnet_type=34, encoder_dim_type='2d', att_type=[False]*4, stride_D1=[1, 2, 2, 2],
                 decoder_type='unet', if_deconv=True, starting_feature_num = 32):
        super(Attention, self).__init__()
        # assert if_deconv
        assert decoder_type == 'fcn' or decoder_type == 'unet'
        assert not(stride_D1==[1, 2, 2, 2] and if_deconv)

        self.return_att_map = False
        self.att_type = att_type
        self.decoder_type = decoder_type
        self.encoder_dim_type = encoder_dim_type
        self.att_loc = []
        for att in att_type:
            if (att == 'CBAM') or (att == 'SE'):
                self.att_loc.append(-1)
            elif att == 'NL':
                self.att_loc.append(-2)
            elif att == False:
                self.att_loc.append(0)

        if decoder_type == 'fcn':
            last_feature = 64
        elif decoder_type == 'unet':
            last_feature = 32

        if encoder_dim_type == '2d':
            if decoder_type == 'fcn':
                self.conv_l3 = nn.Conv2d(128, last_feature, 1)
            if if_deconv:
                self.upsample_last = nn.ConvTranspose2d(last_feature, last_feature, 8, stride=4, padding=2, bias=False)
            else:
                self.upsample_last = nn.Upsample(scale_factor=4.0, mode='bilinear', align_corners =True)
            self.last_conv = nn.Conv2d(last_feature, 2, 1)
        else:
            if decoder_type == 'fcn':
                self.conv_l3 = nn.Conv3d(128, last_feature, 1)
            if if_deconv:
                self.upsample_last = nn.ConvTranspose3d(last_feature, last_feature, (1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False)
            else:
                self.upsample_last = nn.Upsample(scale_factor=(1.0,2.0,2.0), mode='trilinear', align_corners =True)
            self.last_conv = nn.Conv3d(last_feature, 2, 1)

        self._initialize_weights()

        self.encoder = encoder(resnet_type=resnet_type, encoder_dim_type=encoder_dim_type, att_type=att_type,
                               stride_D1=stride_D1, starting_feature_num=starting_feature_num)
        self.decoder = decoder(decoder_type=decoder_type, conv_builder=self.encoder.conv_makers[:-1], scale_factor_depth=stride_D1[:-1],
                               if_deconv=if_deconv)

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

        if isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
            assert m.kernel_size[0] == m.kernel_size[1]
            initial_weight = self.get_upsampling_weight(
                m.in_channels, m.out_channels, m.kernel_size[0])
            m.weight.data.copy_(initial_weight)

    def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
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

    def if_return_att_map(self, if_return_att_map):
        assert type(if_return_att_map) == bool
        self.return_att_map = if_return_att_map
        for i in range(len(self.att_type)):
            if (self.att_type[i] == 'CBAM') or (self.att_type[i] == 'SE'):
                self.encoder.layers[i][self.att_loc[i]].att_type.return_att_map = if_return_att_map
            elif self.att_type[i] == 'NL':
                self.encoder.layers[i][self.att_loc[i]].return_att_map = if_return_att_map

    def att_collector(self):
        att_map = {}
        for i in range(len(self.att_type)):
            if (self.att_type[i] == 'CBAM') or (self.att_type[i] == 'SE'):
                att_map['layer' + str(i + 1)] = self.encoder.layers[i][self.att_loc[i]].att_type.att_map
            elif self.att_type[i] == 'NL':
                att_map['layer' + str(i + 1)] = self.encoder.layers[i][self.att_loc[i]].att_map
        return att_map

    def clear_att_map(self):
        for i in range(len(self.att_type)):
            if (self.att_type[i] == 'CBAM') or (self.att_type[i] == 'SE'):
                self.encoder.layers[i][self.att_loc[i]].att_type.att_map = 0
            elif self.att_type[i] == 'NL':
                self.encoder.layers[i][self.att_loc[i]].att_map = 0

    def forward(self, x):
        encoder0 = self.encoder.stem(x)
        encoder1 = self.encoder.layers[0](encoder0, stride_HW=1, dilation_HW=[1, 1])
        encoder2 = self.encoder.layers[1](encoder1, stride_HW=2, dilation_HW=[1, 1])
        encoder3 = self.encoder.layers[2](encoder2, stride_HW=2, dilation_HW=[1, 1])
        cl = self.encoder.layers[3](encoder3, stride_HW=2, dilation_HW=[1, 1])
        if self.decoder_type == 'fcn':
            encoder3 = self.conv_l3(encoder3)
        decoder2 = self.decoder.layer2_3(encoder2, encoder3)
        decoder1 = self.decoder.layer1_2(encoder1, decoder2)
        seg = self.upsample_last(decoder1)
        seg = self.last_conv(seg)

        cl = self.encoder.pool2(cl)
        cl = cl.flatten(1)
        cl = self.encoder.fc(cl)
        cl = cl.view(x.size(0))

        out = {'category': cl , 'rectum': seg[:,0:1,:], 'cancer': seg[:,1:2,:]}

        if self.return_att_map:
            out.update(self.att_collector())

        return out

class Attention_custom_3d(nn.Module):
    def __init__(self, resnet_type=34, encoder_dim_type='2d', att_type=[False]*4, stride_D1=[1, 2, 2, 2]):
        super(Attention_custom_3d, self).__init__()

        self.return_att_map = False
        self.att_type = att_type

        if encoder_dim_type == '2d':
            self.conv_rectum = nn.Conv2d(64, 1, 1)
            self.conv_cancer = nn.Conv2d(128, 1, 1)
        else:
            self.conv_rectum = nn.Conv3d(64, 1, 1)
            self.conv_cancer = nn.Conv3d(128, 1, 1)
        self.sigmoid_attention = nn.Sigmoid()
        self.att_loc = []
        for att in att_type:
            if (att == 'CBAM') or (att == 'SE'):
                self.att_loc.append(-1)
            elif att == 'NL':
                self.att_loc.append(-2)
            elif att == False:
                self.att_loc.append(0)

        self._initialize_weights()

        self.encoder = encoder(resnet_type=resnet_type, encoder_dim_type=encoder_dim_type, att_type=att_type,
                               stride_D1=stride_D1)


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

        if isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
            assert m.kernel_size[0] == m.kernel_size[1]
            initial_weight = self.get_upsampling_weight(
                m.in_channels, m.out_channels, m.kernel_size[0])
            m.weight.data.copy_(initial_weight)


    def if_return_att_map(self, if_return_att_map):
        assert type(if_return_att_map) == bool
        self.return_att_map = if_return_att_map
        for i in range(len(self.att_type)):
            if (self.att_type[i] == 'CBAM') or (self.att_type[i] == 'SE'):
                self.encoder.layers[i][self.att_loc[i]].att_type.return_att_map = if_return_att_map
            elif self.att_type[i] == 'NL':
                self.encoder.layers[i][self.att_loc[i]].return_att_map = if_return_att_map

    def att_collector(self):
        att_map = {}
        for i in range(len(self.att_type)):
            if (self.att_type[i] == 'CBAM') or (self.att_type[i] == 'SE'):
                att_map['layer'+str(i+1)] = self.encoder.layers[i][self.att_loc[i]].att_type.att_map
            elif self.att_type[i] == 'NL':
                att_map['layer'+str(i+1)] = self.encoder.layers[i][self.att_loc[i]].att_map
        return att_map

    def clear_att_map(self):
        for i in range(len(self.att_type)):
            if (self.att_type[i] == 'CBAM') or (self.att_type[i] == 'SE'):
                self.encoder.layers[i][self.att_loc[i]].att_type.att_map = 0
            elif self.att_type[i] == 'NL':
                self.encoder.layers[i][self.att_loc[i]].att_map = 0

    def forward(self, x):
        encoder0 = self.encoder.stem(x)
        encoder1 = self.encoder.layers[0](encoder0, stride_HW=1, dilation_HW=[1, 1])
        encoder2 = self.encoder.layers[1](encoder1, stride_HW=2, dilation_HW=[1, 1])
        rectum = self.conv_rectum(encoder2)
        rectum_attention = (self.sigmoid_attention(rectum))
        encoder2 = torch.mul(encoder2, rectum_attention)

        cancer = self.conv_cancer(self.encoder.layers[2](encoder2, stride_HW=1, dilation_HW=[1, 2]))
        encoder3 = self.encoder.layers[2](encoder2, stride_HW=2, dilation_HW=[1, 1])

        cl = self.encoder.layers[3](encoder3, stride_HW=2, dilation_HW=[1, 1])
        cl = self.encoder.pool2(cl)
        cl = cl.flatten(1)
        cl = self.encoder.fc(cl)
        cl = cl.view(x.size(0))

        out = {'category': cl , 'rectum': rectum, 'cancer': cancer}

        if self.return_att_map:
            out.update(self.att_collector())

        return out


class Attention_custom_2d(nn.Module):
    def __init__(self, resnet_type=34, encoder_dim_type='2d', att_type=[False]*4, stride_D1=[1, 2, 2, 2]):
        super(Attention_custom_2d, self).__init__()

        self.return_att_map = False
        self.att_type = att_type

        if encoder_dim_type == '2d':
            self.conv_rectum = nn.Conv2d(64, 1, 1)
            self.conv_cancer = nn.Conv2d(128, 1, 1)
        else:
            self.conv_rectum = nn.Conv3d(64, 1, 1)
            self.conv_cancer = nn.Conv3d(128, 1, 1)
        self.sigmoid_attention = nn.Sigmoid()
        self.att_loc = []
        for att in att_type:
            if (att == 'CBAM') or (att == 'SE'):
                self.att_loc.append(-1)
            elif att == 'NL':
                self.att_loc.append(-2)
            elif att == False:
                self.att_loc.append(0)

        self._initialize_weights()

        self.encoder = encoder(resnet_type=resnet_type, encoder_dim_type=encoder_dim_type, att_type=att_type,
                               stride_D1=stride_D1)


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

        if isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
            assert m.kernel_size[0] == m.kernel_size[1]
            initial_weight = self.get_upsampling_weight(
                m.in_channels, m.out_channels, m.kernel_size[0])
            m.weight.data.copy_(initial_weight)

    def if_return_att_map(self, if_return_att_map):
        assert type(if_return_att_map) == bool
        self.return_att_map = if_return_att_map
        for i in range(len(self.att_type)):
            if (self.att_type[i] == 'CBAM') or (self.att_type[i] == 'SE'):
                self.encoder.layers[i][self.att_loc[i]].att_type.return_att_map = if_return_att_map
            elif self.att_type[i] == 'NL':
                self.encoder.layers[i][self.att_loc[i]].return_att_map = if_return_att_map

    def att_collector(self):
        att_map = {}
        for i in range(len(self.att_type)):
            if (self.att_type[i] == 'CBAM') or (self.att_type[i] == 'SE'):
                att_map['layer'+str(i+1)] = self.encoder.layers[i][self.att_loc[i]].att_type.att_map
            elif self.att_type[i] == 'NL':
                att_map['layer'+str(i+1)] = self.encoder.layers[i][self.att_loc[i]].att_map
        return att_map

    def clear_att_map(self):
        for i in range(len(self.att_type)):
            if (self.att_type[i] == 'CBAM') or (self.att_type[i] == 'SE'):
                self.encoder.layers[i][self.att_loc[i]].att_type.att_map = 0
            elif self.att_type[i] == 'NL':
                self.encoder.layers[i][self.att_loc[i]].att_map = 0

    def forward(self, x):
        encoder0 = self.encoder.stem(x)
        encoder1 = self.encoder.layers[0](encoder0, stride_HW=1, dilation_HW=[1, 1])
        seg = self.encoder.layers[1](encoder1, stride_HW=1, dilation_HW=[1, 2])
        rectum = self.conv_rectum(seg)
        rectum_attention = self.sigmoid_attention(F.interpolate(rectum, [int(rectum.size(2)/2), int(rectum.size(3)/2)],
                                                      mode='bicubic', align_corners=True))

        cancer = self.conv_cancer(self.encoder.layers[2](seg, stride_HW=1, dilation_HW=[2, 4]))


        encoder2 = self.encoder.layers[1](encoder1, stride_HW=2, dilation_HW=[1, 1])
        encoder2 = torch.mul(encoder2, rectum_attention)
        encoder3 = self.encoder.layers[2](encoder2, stride_HW=2, dilation_HW=[1, 1])
        cl = self.encoder.layers[3](encoder3, stride_HW=2, dilation_HW=[1, 1])
        cl = self.encoder.pool2(cl)
        cl = cl.flatten(1)
        cl = self.encoder.fc(cl)
        cl = cl.view(x.size(0))

        out = {'category': cl , 'rectum': rectum, 'cancer': cancer}

        if self.return_att_map:
            out.update(self.att_collector())

        return out
