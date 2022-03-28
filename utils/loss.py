import torch
import torch.nn as nn
import random
import math
from torchvision import transforms, utils
import torch.nn.functional as F
# import torch.nn.SmoothL1Loss as SmoothL1Loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma=gamma

    def forward(self, logit, label):

        # print(f'<LOSS - 1> logit: {logit.size()}, label: {label.size()}')
        # logit = logit.view(logit.size(0), -1)  # N,C,H,W => N, H*W (C=1)
        # label = label.view(label.size(0), -1)  # N,H,W => N, H*W

        BCE = F.binary_cross_entropy_with_logits(logit, label, reduction='none')
        loss = ((1.0 - torch.exp(-BCE)) ** self.gamma) * BCE
        # print(f'<LOSS - 2> logit: {logit.size()}, label: {label.size()}, loss: {loss.size()}')
        # return loss.sum(dim=1, keepdim=False)
        return loss


class FocalLoss_CenterLoss(nn.Module):
    def __init__(self, gamma=2, cl_alpha = 0.5, cl_lambda = 0.003, feat_dim = (256, 2), backprop_center = False):
        super(FocalLoss_CenterLoss, self).__init__()
        self.fl_gamma = gamma
        self.cl_alpha = cl_alpha
        self.cl_lambda = cl_lambda
        self.backprop_center = backprop_center
        self.num_classes = feat_dim[1]
        self.class_idx = torch.arange(self.num_classes, dtype=torch.float32, device="cuda").view(1,1,1,self.num_classes)
        # if backprop_center:
        #     # self.register_parameter(name='center', param=torch.nn.Parameter(torch.zeros(feat_dim, requires_grad=True, device="cuda")))
        #     self.center = torch.nn.Parameter(torch.zeros(feat_dim, requires_grad=True))
        #     print(f'backprop_center TRUE')
        # else:
        #     self.center = torch.zeros(feat_dim, requires_grad=False)
        #     print(f'backprop_center FALSE')

        self.center = torch.nn.Parameter(torch.zeros(feat_dim))
        if backprop_center:
            self.center.requires_grad_(True)
        else:
            self.center.requires_grad_(False)

        # self.center = nn.Parameter(torch.nn.Parameter(torch.zeros(feat_dim)))
        if not backprop_center:
            self.center.requires_grad = False

        self.feat_dim = feat_dim
        self.mse = nn.MSELoss(reduction='none')
        # self.center = self.center.

    def forward(self, logit, repr, feat, label):
        """
        1) loss_fl : <N, D>
        2) loss_cl : <N, D>
        :param logit: <N, D>
        :param repr: <N, D>
        :param feat: <N, D, C>
        :param label: <N>
        :return:
        """
        N = label.size(0) # Batch-size
        D = logit.size(1) # Depth: 7 (z-axis)
        # C = self.feat_dim[0]

        # label: <N, D>
        label = label.view(N, 1).expand(N,D)

        BCE = F.binary_cross_entropy_with_logits(logit, label, reduction='none')
        # loss_fl: <N, D>
        loss_fl = ((1.0 - torch.exp(-BCE)) ** self.fl_gamma) * BCE

        # (N, D, C)
        # feat: N-D-C, self.center: C, class
        # =>
        # feat: N-D-Class, self.center: C, class
        loss_cl = torch.pow(feat, 2).sum(dim=2, keepdim=True).expand(N, D, self.num_classes) + \
                  torch.pow(self.center, 2).sum(dim=0, keepdim=True).view(1, 1, self.num_classes).expand(N, D, self.num_classes)

        # N-D-C * C-Class => N-D-Class
        loss_cl = loss_cl - (2.0 * torch.matmul(feat, self.center) )
        mask_label = label.view(N, D, 1)
        # N-D-1
        loss_cl = self.cl_lambda * torch.gather(loss_cl, dim=2, index=mask_label.long()).view(N, D)
        # print(f'device> lossfl: {loss_fl.get_device()}, losscl: {loss_cl.get_device()}, repr: {repr.get_device()}')
        loss = torch.sum((loss_fl + loss_cl) * repr, dim=1, keepdim=False)
        # print(f'loss fl: {loss_fl}, loss cl: {loss_cl}, repr: {repr}')

        return loss

    def update_center(self, logit, repr, feat, label):
        """
        1) self.center: <C, Class>
        diff: <N, D, C>
        :param logit: <N, D>
        :param repr: <N, D>
        :param feat: <N, D, C>
        :param label: <N>
        :return:
        """
        # batch_size = label.size(0)
        with torch.no_grad():
            N = label.size(0)  # Batch-size
            D = logit.size(1)  # Depth: 7 (z-axis)
            C = self.feat_dim[0]
            # _, counts = torch.unique(label, return_counts=True)
            # print(f'hihiih> label: {label.size()}')
            # print(f'hihiih> repr: {repr.size()}')
            counts = torch.tensor([(N*D)-torch.sum(repr*label.view(N,1).expand(N,D)), torch.sum(repr*label.view(N,1).expand(N,D))], device = feat.get_device()).type(torch.float32)

            # N-D-C-self.num_classes
            # Unlike forward where we choose one from classes, we need to use all class-vectors
            feat = feat.view(N,D,C,1).expand(N,D,C,self.num_classes)
            centers = self.center.view(1, 1, C, self.num_classes).expand(N,D,C,self.num_classes)
            diff = centers - feat

            mask = label.view(N,1,1,1).expand(N,D,C,self.num_classes)
            mask = torch.eq(mask, self.class_idx.expand(N,D,C,self.num_classes)) * repr.view(N,D,1,1).expand(N,D,C,self.num_classes)

            self.center = self.center - self.cl_alpha * (torch.sum(diff*mask, dim=(0,1), keepdim=False)) / (1.0 + counts.view(1,self.num_classes))



