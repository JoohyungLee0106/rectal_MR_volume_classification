import torch.nn as nn
import torch
import torch.nn.functional as F
from .conv_builder import Conv3DSimple as Conv3DSimple
from .conv_builder import Conv3DNoTemporal as Conv3DNoTemporal
from .conv_builder import Conv2Plus1D as Conv2Plus1D
from .self_attention_module import (CBAM, SE, NonLocalBlock2D, NonLocalBlock3D, NonLocalBlockND)


model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, att_type = False):

        super(BasicBlock, self).__init__()

        self.cbr1 = nn.Sequential(
            conv_builder(inplanes, planes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.cbr2 = nn.Sequential(
            conv_builder(planes, planes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if att_type == 'CBAM':
            self.att_type = CBAM(planes, conv_builder=conv_builder)
        elif att_type == 'SE':
            self.att_type = SE(planes)
        elif (att_type == 'NL') or (att_type == False):
            self.att_type = False
        else:
            raise ValueError('<BasicBLock> Wrong setting!!!')

    def forward(self, x):
        residual = x

        out = self.cbr1(x)
        out = self.cbr2(out)

        if self.downsample is not None:
            # print(f'input: {residual.size()}')
            residual = self.downsample(x)
            # print(f'residual: {residual.size()}')
            # print(f'out: {out.size()}')

        if self.att_type:
            out = self.att_type(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, att_type = False):

        super(Bottleneck, self).__init__()

        # 1x1x1
        self.cbr1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.cbr2 = nn.Sequential(
            conv_builder(planes, planes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.cbr3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if att_type == 'CBAM':
            self.att_type = CBAM(planes, conv_builder=conv_builder)
        elif att_type == 'SE':
            self.att_type = SE(planes)
        elif (att_type == 'NL') or (att_type == False):
            self.att_type = False
        else:
            raise ValueError('<BasicBLock> Wrong setting!!!')

    def forward(self, x):
        residual = x

        out = self.cbr1(x)
        out = self.cbr2(out)
        out = self.cbr3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.att_type:
            out = self.att_type(out)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self, image_channel=1, starting_feature_num=32, dimension='3d'):
        if dimension == '3d':
            print(f'<stem> 3d activated !!!!')
            super(BasicStem, self).__init__(
                nn.Conv3d(image_channel, starting_feature_num, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                          padding=(1, 3, 3), bias=False),
                nn.BatchNorm3d(starting_feature_num),
                nn.ReLU(inplace=True))

        elif dimension == '2.5d':
            print(f'<stem> 2.5d activated !!!!')
            super(BasicStem, self).__init__(
                nn.Conv3d(image_channel, starting_feature_num, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                          padding=(0, 3, 3), bias=False),
                nn.BatchNorm3d(starting_feature_num),
                nn.ReLU(inplace=True))


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self, image_channel=1, starting_feature_num=32,  dimension = '3d'):
        midplanes = (image_channel * starting_feature_num * 3 * 3 * 3) // (image_channel * 3 * 3 + 3 * starting_feature_num)
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(1, midplanes, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, starting_feature_num, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(starting_feature_num),
            nn.ReLU(inplace=True))


class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, stem, block_num=[3, 4, 6, 3], image_channel=1, stem_dimension='3d',
                 num_classes=1, starting_feature_num=32, att_type = [False]*4, global_pool = nn.AdaptiveAvgPool3d((1, 1, 1)),
                 zero_init_residual=False, if_framewise=False, loss = None, loss_param={}, margin_triplet = 256.0):
        """Generic resnet video generator.
        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = starting_feature_num
        # self.register_parameter(name='loss', param=loss['category'])
        # self.loss = loss(**loss_param)
        self.margin_triplet = margin_triplet

        self.stem = stem(image_channel, starting_feature_num, dimension=stem_dimension)
        self.att_loc = []
        self.return_att_map = False
        self.att_type = att_type
        self.self_att_cls = False

        if att_type[1] == 'CBAM':
            self.self_att_cls = CBAM
        elif att_type[1] == 'SE':
            self.self_att_cls = SE
        elif att_type[1] == 'NL':
            self.self_att_cls = NonLocalBlockND

        for att in att_type:
            if (att == 'CBAM') or (att == 'SE'):
                self.att_loc.append(-1)
            elif att == 'NL':
                self.att_loc.append(-2)
            elif att == False:
                self.att_loc.append(0)

        self.layers = nn.ModuleList([
            self._make_layer(block, conv_makers[0], starting_feature_num, block_num[0], stride=1, att_type=att_type[0]),
            self._make_layer(block, conv_makers[1], starting_feature_num * 2, block_num[1], stride=2, att_type=att_type[1]),
            self._make_layer(block, conv_makers[2], starting_feature_num * 4, block_num[2], stride=2, att_type=att_type[2]),
            self._make_layer(block, conv_makers[3], starting_feature_num * 8, block_num[3], stride=2, att_type=att_type[3])
            ])


        # self.layer1 = self._make_layer(block, conv_makers[0], starting_feature_num, block_num[0], stride=1, att_type=att_type[0])
        # self.layer2 = self._make_layer(block, conv_makers[1], starting_feature_num*2, block_num[1], stride=2, att_type=att_type[1])
        # self.layer3 = self._make_layer(block, conv_makers[2], starting_feature_num*4, block_num[2], stride=2, att_type=att_type[2])
        # self.layer4 = self._make_layer(block, conv_makers[3], starting_feature_num*8, block_num[3], stride=2, att_type=att_type[3])

        self.global_pool = global_pool
        if if_framewise:
            self.flatten_idx = 2
        else:
            self.flatten_idx = 1

        self.fc_bin = nn.Linear( (starting_feature_num * 8 * block.expansion)**2, num_classes)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)


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

    # def register_device(self):
    #     self.device_num = self.fc.weight.get_device()
    #
    # def to_device(self, data):
    #     data['image'] = data['image'].cuda(self.device_num, non_blocking=True)
    #     data['category'] = data['category'].cuda(self.device_num, non_blocking=True)
    #     data['repr_num'] = data['repr_num'].cuda(self.device_num, non_blocking=True)


    def forward(self, data, if_train=False):
        n = data["repr_num"].size(0)
        d = data["repr_num"].size(1)
        # data['repr_num'] = data['repr_num'].cuda(self.fc.weight.get_device())
        r =int(data["image"].size(0))
        rt3 = int(data["image"].size(0)/2)
        rt2 = r-rt3
        # print(f'shape: {data["image"].size()}')

        # rt2 = int((torch.sum(data["repr_num"]) - torch.sum(data["repr_num"] * data["category"].view(n,1).expand(n,d))).item())
        # x = (data['image'] - torch.mean(data['image'], dim=(2,3,4), keepdim=True))/torch.std(data['image'], dim=(2,3,4), keepdim=True)
        x = self.stem(data['image'])

        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)

        # N-C-D
        x = self.global_pool(x).flatten(2)
        # N-C^2
        x = (torch.tril(torch.bmm(x, x.permute(0, 2, 1)))).flatten(1) / 7.0
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)
        x = F.normalize(x, 2, 1)


        # temp_idx = torch.nonzero(data['repr_num'].flatten(0)).squeeze(1)
        # print(f'repr(N,D) {torch.sum(data["repr_num"])}:\n{data["repr_num"]}')
        # print(f'temp_idx(R) {temp_idx.size()}:\n{temp_idx}')
        # temp_idx = torch.nonzero(data['repr_num'].view(data['repr_num'].size(0), data['repr_num'].size(1), 1)
        #                          .expand(data['repr_num'].size(0), data['repr_num'].size(1), x.size(1)).cuda(x.get_device()), as_tuple=True)

        # print(f'category before conversion:\n{data["category"]}')
        # data['category'] = (data['category'].unsqueeze(1).repeat(1, d).flatten(0))[temp_idx]
        # print(f'category after conversion:\n{data["category"]}')
        # print(f'x before flatten {x.size()}(N,C,D):\n{x.permute(0,2,1)[:, :, 3]}')

        # print(f'x before flatten(N,D,C) {x.permute(0,2,1).size()}:\n{x.permute(0,2,1)[:,:,:3]}')
        # (R, C)
        # x = torch.index_select(x.permute(1, 0, 2).flatten(1), dim=1, index = temp_idx).permute(1,0)
        # torch.gather(x.permute(1, 0, 2).flatten(1), dim=1, index=temp_idx)

        # print(f'x after flatten {x.size()}(R,C):\n{x[:,:3]}')

###################
        logit = self.fc_bin(x).squeeze(1)

        out = {'category': logit}

        if self.return_att_map:
            out.update(self.att_collector())
        if if_train:
            x2 = torch.sum(torch.pow(x, 2), dim=1, keepdim=True).expand(r, r)
            # print(f'x2:\n{x2}')
            # print(f'square:\n{x2 + x2.permute(1, 0)}')
            # print(f'negative:\n{(2.0 * torch.mm(x, x.permute(1, 0)))}')
            # (R, R)
            # distmat = torch.sqrt(x2 + x2.permute(1, 0) - (2.0 * torch.mm(x, x.permute(1, 0))) + 1e-07) + 1e-07
            distmat = x2 + x2.permute(1, 0) - (2.0 * torch.mm(x, x.permute(1, 0)))
            distmat[range(r), range(r)] = -999.0
            # distmat = torch.sqrt(x2 + x2.permute(1, 0) - (2.0 * torch.mm(x, x.permute(1, 0))) )
            # distmat = x2 + x2.permute(1,0) - (2.0 * torch.mm(x, x.permute(1,0)))
            # (check) distmat must be symmetric
            # print(f'distmat:\n{distmat}')

            # print(f'Nan location:\n{torch.nonzero(distmat != distmat)}')

            # print(f'LOSS TRIPLET:\n{torch.cat([F.relu(torch.amax(distmat[:rt2, :rt2], dim=1) - torch.amin(distmat[:rt2, rt2:], dim=1) + self.margin_triplet), F.relu(-torch.amin(distmat[rt2:, :rt2], dim=1) + torch.amax(distmat[rt2:, rt2:], dim=1) + self.margin_triplet)], dim=0)}')

            loss_triplet = torch.sum(F.relu(torch.amax(distmat[:rt2, :rt2], dim=1) - torch.amin(distmat[:rt2, rt2:], dim=1) + self.margin_triplet)) + \
                        torch.sum(F.relu(-torch.amin(distmat[rt2:, :rt2], dim=1) + torch.amax(distmat[rt2:, rt2:], dim=1) + self.margin_triplet))

            loss_focal = F.binary_cross_entropy_with_logits(logit, data['category'], reduction='none')
            # print(f'LOSS FOCAL:\n{((1.0 - torch.exp(-loss_focal)) ** 2.0) * loss_focal}')
            loss_focal = torch.sum(
                ((1.0 - torch.exp(-loss_focal)) ** 2.0) * loss_focal
            )


            out.update({'loss': (loss_triplet/self.margin_triplet) + loss_focal})
            # out.update({'loss': loss_focal})
            # out.update({'loss': self.loss['category'](x, data['category'])})
            # self.loss['category'].update_center(x, data['repr_num'], feat, data['category'])

        # data['category'] = (data['category'].unsqueeze(1).repeat(1, data['repr_num'].size(1)))[temp_idx]

        return out, data['category']

    def _make_layer(self, block, conv_builder, planes, blocks, stride, att_type):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample, att_type = att_type))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder, att_type = att_type))

        if att_type == 'NL':
            layers.insert(-1, NonLocalBlock3D(planes))
        elif (att_type == 'CBAM') or (att_type == 'SE'):
            layers[-1].att_type.return_att_map = True

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif (isinstance(m, nn.BatchNorm3d)) or (isinstance(m, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.self_att_cls:
            for obj in self.self_att_cls.obj:
                obj._initialize_weights()

        nn.init.normal_(self.fc_bin.weight, 0, 0.01)
        nn.init.constant_(self.fc_bin.bias, 0)



def encoder(resnet_type = 34, encoder_dim_type = '2d', att_type = [False]*4, stride_D1=[1, 2, 2, 2],
            starting_feature_num=32, num_classes=1, if_framewise=False, loss = None, loss_param={}):
    assert resnet_type == 18 or resnet_type == 34 or resnet_type == 50
    for att in att_type:
        assert att == False or att == 'CBAM' or att == 'SE' or att == 'NL'
    assert encoder_dim_type == 'r3d' or encoder_dim_type == '2plus1d' or encoder_dim_type == 'r2d' or\
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

    if if_framewise:
        z_out = 7
    else:
        z_out = 1

    # if encoder_dim_type == '2d':
    #     stem = BasicStem
    #     stem_dimension = '2d'
    #     conv_makers = [Conv2DSimple] * 4
    #     pool2 = nn.AdaptiveAvgPool2d((1, 1))
    if encoder_dim_type == 'r2d':
        stem = BasicStem
        stem_dimension = '2.5d'
        conv_makers = [Conv3DNoTemporal]*4
        # global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        global_pool = nn.AdaptiveAvgPool3d((z_out, 1, 1))
    elif encoder_dim_type == 'r3d':
        stem = BasicStem
        stem_dimension = '3d'
        conv_makers = [Conv3DSimple] * 4
        # global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        global_pool = nn.AdaptiveAvgPool3d((z_out, 1, 1))
    elif encoder_dim_type == 'rmc2':
        stem = BasicStem
        stem_dimension = '2.5d'
        conv_makers = [Conv3DSimple] * 4
        # global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        global_pool = nn.AdaptiveAvgPool3d((z_out, 1, 1))
    elif encoder_dim_type == 'rmc3':
        stem = BasicStem
        stem_dimension = '2.5d'
        conv_makers = [Conv3DNoTemporal] + [Conv3DSimple] * 3
        # global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        global_pool = nn.AdaptiveAvgPool3d((z_out, 1, 1))
    elif encoder_dim_type == 'rmc4':
        stem = BasicStem
        stem_dimension = '2.5d'
        conv_makers = [Conv3DNoTemporal]*2 + [Conv3DSimple] * 2
        # global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        global_pool = nn.AdaptiveAvgPool3d((z_out, 1, 1))
    elif encoder_dim_type == 'rmc5':
        stem = BasicStem
        stem_dimension = '2.5d'
        conv_makers = [Conv3DNoTemporal]*3 + [Conv3DSimple]
        # global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        global_pool = nn.AdaptiveAvgPool3d((z_out, 1, 1))
    elif encoder_dim_type == 'mc2':
        stem = BasicStem
        stem_dimension = '3d'
        conv_makers = [Conv3DNoTemporal] * 4
        # global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        global_pool = nn.AdaptiveAvgPool3d((z_out, 1, 1))
    elif encoder_dim_type == 'mc3':
        stem = BasicStem
        stem_dimension = '3d'
        conv_makers = [Conv3DSimple] + [Conv3DNoTemporal] * 3
        # global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        global_pool = nn.AdaptiveAvgPool3d((z_out, 1, 1))
    elif encoder_dim_type == 'mc4':
        stem = BasicStem
        stem_dimension = '3d'
        conv_makers = [Conv3DSimple] * 2 + [Conv3DNoTemporal] * 2
        # global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        global_pool = nn.AdaptiveAvgPool3d((z_out, 1, 1))

    elif encoder_dim_type == 'mc5':
        stem = BasicStem
        stem_dimension = '3d'
        conv_makers = [Conv3DSimple] * 3 + [Conv3DNoTemporal]
        # global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        global_pool = nn.AdaptiveAvgPool3d((z_out, 1, 1))
    elif encoder_dim_type == '2plus1d':
        stem = R2Plus1dStem
        stem_dimension = '3d'
        conv_makers = [Conv2Plus1D] * 4
        # global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        global_pool = nn.AdaptiveAvgPool3d((z_out, 1, 1))
    else:
        raise ValueError("<encoder> Wrong 'encoder_dim_type' !!!")


    return VideoResNet(block=block,
                  conv_makers=conv_makers,
                  starting_feature_num=starting_feature_num,
                  global_pool=global_pool,
                  att_type=att_type,
                  stem_dimension=stem_dimension,
                  block_num=block_num,
                  stem=stem,
                  if_framewise=if_framewise,
                  num_classes=num_classes,
                  loss = loss,
                  loss_param=loss_param)