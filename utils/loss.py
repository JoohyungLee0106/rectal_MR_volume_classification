import torch
import torch.nn as nn
from torchvision import transforms, utils
import torch.nn.functional as F

# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
# 나중에 average top-k loss 해보자

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, if_custom_mean=False, if_logit=True):
        super(FocalLoss, self).__init__()
        self.gamma=gamma
        # self.division_factor = 1.0
        # if if_logit:
        #     self.forward = self.forward_logit
        # else:
        #     if if_custom_mean:
        #         self.forward = self.forward_custom_mean
        #     else:
        #         self.forward = self.forward_normal

    # def set_division_factor(self, x):
    #     self.division_factor = float(x)

    # def forward_logit(self, logit, label):
    def forward(self, logit, label):
        logit = logit.view(logit.size(0), -1)  # N,C,H,W => N, H*W (C=1)
        label = label.view(label.size(0), -1)  # N,H,W => N, H*W

        BCE = F.binary_cross_entropy_with_logits(logit, label, reduction='none')
        loss = ((1.0 - torch.exp(-BCE)) ** self.gamma) * BCE
        return loss.sum(dim=1, keepdim=False)

class FocalLoss_2categories_onlyrepr(nn.Module):
    def __init__(self, gamma=2, if_custom_mean=False, if_logit=True):
        super(FocalLoss_2categories_onlyrepr, self).__init__()
        self.gamma=gamma

    def forward(self, logit, label, repr_num):
        logit = logit.view(logit.size(0), -1)  # N,C,H,W => N, H*W (C=1)
        # label = label.view(label.size(0), -1)  # N,H,W => N, H*W

        label = label.unsqueeze(1).repeat(1, 7)

        BCE = F.binary_cross_entropy_with_logits(logit, label, reduction='none')
        focal_loss = ((1.0 - torch.exp(-BCE)) ** self.gamma) * BCE

        # print(f'logit: {logit}')
        # # print(f'logit[:,repr_num]: {logit[:,repr_num]}')
        # print(f'repr_num: {repr_num}, type: {type(repr_num)}')
        # print(f'label: {logit}')
        # loss = 0

        loss = torch.zeros(focal_loss.size(0), device=focal_loss.get_device())

        i = 0

        for j in repr_num:
            # print(f'reprenum : {repr_num}')
            # print(f'reprenum type: {type(repr_num)}')
            # print(f'reprenum 0 type: {type(repr_num[0])}')
            loss[i] = torch.sum(focal_loss[i, j])
            i += 1

        # i=0
        # for j in repr_num:
        #     weight[i, j] = 1
        #     i+=1

        # for i in range(logit.size(0)):
        #     for j in repr_num[i]:
        #         weight[i,j]=1

        # loss = torch.sum( focal_loss * weight, dim = 1, keepdim=False)

        # for i in range(logit.size(0)):
        #     for j in repr_num[i]:
        #         loss = loss + focal_loss[i,j]

        return loss


class FocalLoss_2categories_negativet3(nn.Module):
    def __init__(self, gamma=2, if_custom_mean=False, if_logit=True):
        super(FocalLoss_2categories_negativet3, self).__init__()
        self.gamma=gamma

    def forward(self, logit, label, repr_num):
        logit = logit.view(logit.size(0), -1)  # N,C,H,W => N, H*W (C=1)
        # label = label.view(label.size(0), -1)  # N,H,W => N, H*W

        label = label.unsqueeze(1).repeat(1, 7)

        BCE = F.binary_cross_entropy_with_logits(logit, label, reduction='none')
        focal_loss = ((1.0 - torch.exp(-BCE)) ** self.gamma) * BCE

        BCE_neg = F.binary_cross_entropy_with_logits(-logit, (1.0-label), reduction='none')
        focal_loss_negative = ((1.0 - torch.exp(-BCE_neg)) ** self.gamma) * BCE_neg

        loss = torch.zeros(focal_loss.size(0), device=focal_loss.get_device())

        i = 0

        for j in repr_num:
            not_repr = list(range(7))
            for nr in j:
                not_repr.remove(nr)
            loss[i] = torch.sum(focal_loss[i, j])
            if label[i, 0] == 0:
                loss[i] += torch.sum(focal_loss_negative[i, not_repr])


            i += 1


        return loss


class FocalLoss_2categories_allt2(nn.Module):
    def __init__(self, gamma=2, if_custom_mean=False, if_logit=True):
        super(FocalLoss_2categories_allt2, self).__init__()
        self.gamma = gamma

    def forward(self, logit, label, repr_num):
        logit = logit.view(logit.size(0), -1)  # N,C,H,W => N, H*W (C=1)
        # label = label.view(label.size(0), -1)  # N,H,W => N, H*W

        label = label.unsqueeze(1).repeat(1, 7)

        BCE = F.binary_cross_entropy_with_logits(logit, label, reduction='none')
        focal_loss = ((1.0 - torch.exp(-BCE)) ** self.gamma) * BCE


        # weight = torch.zeros(focal_loss.size())

        loss = torch.zeros(focal_loss.size(0), device=focal_loss.get_device())

        i=0

        for j in repr_num:
            # print(f'reprenum : {repr_num}')
            # print(f'reprenum type: {type(repr_num)}')
            # print(f'reprenum 0 type: {type(repr_num[0])}')
            if label[i,0] == 0:
                # weight[i, :] = 1
                loss[i] = torch.sum(focal_loss[i, :])
            else:
                loss[i] = torch.sum(focal_loss[i,j])
            i+=1

        # loss = torch.sum(focal_loss * weight, dim=1, keepdim=False)

        # for i in range(logit.size(0)):
        #     if label[i,0] == 1:
        #         for j in range(weight.size(1)):
        #                 if j not in repr_num[i]:
        #                     weight[i,j] = 0
        #
        # for i in range(logit.size(0)):
        #     if label[i,0] == 1:
        #         for j in repr_num[i]:
        #             loss += focal_loss[i,j]
        #     else:
        #         loss += torch.sum(focal_loss[i,:])
        return loss

class FocalLoss_rankingloss_prob_label(nn.Module):
    def __init__(self, gamma=2, if_custom_mean=False, if_logit=True):
        super(FocalLoss_rankingloss_prob_label, self).__init__()
        self.gamma = gamma

    def forward(self, logit, label, repr_num):
        logit = logit.view(logit.size(0), -1)  # N,C,H,W => N, H*W (C=1)
        prob = torch.sigmoid(logit)

        # prob_rs_min = ((torch.min(prob, dim=1)[0]).unsqueeze(1).expand_as(logit)).detach()

        label = label.unsqueeze(1).repeat(1, 7)

        BCE = F.binary_cross_entropy_with_logits(logit, label, reduction='none')
        focal_loss = ((1.0 - torch.exp(-BCE)) ** self.gamma) * BCE

        # BCE_ranking = F.binary_cross_entropy(prob, prob_min, reduction='none')

        loss = torch.zeros(focal_loss.size(0), device=focal_loss.get_device())

        i=0

        for rs in repr_num:
            list7 = list(range(7))
            nrs = [nr for nr in list7 if nr not in rs]
            # print(f'rs:{rs}, nrs: {nrs}')
            # print(f' weight size: {(F.relu((prob[i, nrs] - torch.min(prob[i,rs])).sign())).size()}')
            # print(f' BCE_ranking size: {(BCE_ranking[i, nrs]).size()}')
            BCE_zero = F.binary_cross_entropy(prob[i, nrs], torch.zeros(len(nrs), device=focal_loss.get_device()).detach(), reduction='none')
            focal_loss_zero = ((1.0 - torch.exp(-BCE_zero)) ** self.gamma) * BCE_zero

            loss[i] = torch.sum(focal_loss[i, rs]) + torch.sum(F.relu((prob[i, nrs] - torch.min(prob[i,rs])).sign()) * focal_loss_zero)

            i += 1

        return loss

class FocalLoss_rankingloss_prob_relative(nn.Module):
    def __init__(self, gamma=2, if_custom_mean=False, if_logit=True):
        super(FocalLoss_rankingloss_prob_relative, self).__init__()
        self.gamma = gamma

    def forward(self, logit, label, repr_num):
        logit = logit.view(logit.size(0), -1)  # N,C,H,W => N, H*W (C=1)
        prob = torch.sigmoid(logit)
        # prob_min = ((torch.min(prob, dim=1)[0]).unsqueeze(1).expand_as(logit)).detach()

        # prob_rs_min = ((torch.min(prob, dim=1)[0]).unsqueeze(1).expand_as(logit)).detach()
        # prob_min.requires_grad = False
        # label = label.view(label.size(0), -1)  # N,H,W => N, H*W

        label = label.unsqueeze(1).repeat(1, 7)

        BCE = F.binary_cross_entropy_with_logits(logit, label, reduction='none')
        focal_loss = ((1.0 - torch.exp(-BCE)) ** self.gamma) * BCE

        # BCE_ranking = F.binary_cross_entropy(prob, prob_min, reduction='none')

        loss = torch.zeros(focal_loss.size(0), device=focal_loss.get_device())

        i=0

        for rs in repr_num:
            list7 = list(range(7))
            nrs = [nr for nr in list7 if nr not in rs]
            # print(f'rs:{rs}, nrs: {nrs}')
            # print(f' weight size: {(F.relu((prob[i, nrs] - torch.min(prob[i,rs])).sign())).size()}')
            # print(f' BCE_ranking size: {(BCE_ranking[i, nrs]).size()}')
            BCE_relative = F.binary_cross_entropy(prob[i, nrs], (torch.min(prob[i, rs])).expand(len(nrs)).detach(), reduction='none')
            focal_loss_relative = ((1.0 - torch.exp(-BCE_relative)) ** self.gamma) * BCE_relative

            loss[i] = torch.sum(focal_loss[i, rs]) + torch.sum(F.relu((prob[i, nrs] - torch.min(prob[i,rs])).sign()) * focal_loss_relative)

            i += 1

        return loss

class FocalLoss_rankingloss_logit(nn.Module):
    def __init__(self, gamma=2, if_custom_mean=False, if_logit=True):
        super(FocalLoss_rankingloss_logit, self).__init__()
        self.gamma = gamma

    def forward(self, logit, label, repr_num):
        logit = logit.view(logit.size(0), -1)  # N,C,H,W => N, H*W (C=1)
        # prob = torch.sigmoid(logit)
        # label = label.view(label.size(0), -1)  # N,H,W => N, H*W

        label = label.unsqueeze(1).repeat(1, 7)

        BCE = F.binary_cross_entropy_with_logits(logit, label, reduction='none')
        focal_loss = ((1.0 - torch.exp(-BCE)) ** self.gamma) * BCE

        loss = torch.zeros(focal_loss.size(0), device=focal_loss.get_device())

        i=0

        for rs in repr_num:
            list7 = list(range(7))
            nrs = [nr for nr in list7 if nr not in rs]

            loss[i] = torch.sum(focal_loss[i, rs]) + torch.sum(torch.relu(logit[i, nrs] - torch.min(logit[i,rs])))

            i+=1

        return loss


class HingeLoss(nn.Module):
    def __init__(self, gamma=2, if_custom_mean=False, if_logit=True):
        super(HingeLoss, self).__init__()
        self.gamma=gamma
        # self.division_factor = 1.0
        # if if_logit:
        #     self.forward = self.forward_logit
        # else:
        #     if if_custom_mean:
        #         self.forward = self.forward_custom_mean
        #     else:
        #         self.forward = self.forward_normal

    # def set_division_factor(self, x):
    #     self.division_factor = float(x)

    # def forward_logit(self, logit, label):

    def forward(self, logit, label, repr_num):
        logit = logit.view(logit.size(0), -1)  # N,C,H,W => N, H*W (C=1)
        label = label.view(label.size(0), -1)  # N,H,W => N, H*W

        label = label.unsqueeze(1).repeat(1, 7)

        # BCE = F.binary_cross_entropy_with_logits(logit, label, reduction='none')
        # focal_loss = ((1.0 - torch.exp(-BCE)) ** self.gamma) * BCE

        # print(f'logit: {logit}')
        # # print(f'logit[:,repr_num]: {logit[:,repr_num]}')
        # print(f'repr_num: {repr_num}, type: {type(repr_num)}')
        # print(f'label: {logit}')
        loss = 0

        for i in range(logit.size(0)):
            if label[i,0] == 1:
                for j in range(7):
                    if j in repr_num[i]:
                        loss += torch.abs(logit[i,j] - 1.0)
                    else:
                        loss += F.relu(logit[i,j] - 1.0)
            else:
                for j in range(7):
                    if j in repr_num[i]:
                        loss += torch.abs(logit[i, j])
                    else:
                        loss += F.relu(logit[i, j])
        return loss

class FocalMaxLoss(nn.Module):
    def __init__(self, gamma=2, if_custom_mean=False, if_logit=True):
        super(FocalMaxLoss, self).__init__()
        self.gamma = gamma
        # self.division_factor = 1.0
        # if if_logit:
        #     self.forward = self.forward_logit
        # else:
        #     if if_custom_mean:
        #         self.forward = self.forward_custom_mean
        #     else:
        #         self.forward = self.forward_normal

    # def set_division_factor(self, x):
    #     self.division_factor = float(x)

    # def forward_logit(self, logit, label):
    def forward(self, logit, label):
        # logit = logit.view(logit.size(0), -1)  # N,C,H,W => N, H*W (C=1)
        # label = label.view(label.size(0), -1)  # N,H,W => N, H*W
        # print(f'<FocalMaxLoss> logit shape: {logit.size()}, label shape: {label.size()}')
        label = label.unsqueeze(1).repeat(1,7)
        weight = F.softmax(logit*self.gamma, dim=1)
        # print(f'<FocalMaxLoss> label shape: {label.size()}, weight shape: {logit.size()}')

        BCE = F.binary_cross_entropy_with_logits(logit, label, reduction='none')
        # loss = ((1.0 - torch.exp(-BCE)) ** self.gamma) * BCE
        # print(f'<FocalMaxLoss> BCE shape: {BCE.size()}, loss shape: {loss.size()}')

        # print('===============================================')
        return (weight*BCE).mean(dim=1, keepdim=False)

    # def forward_normal(self, pred, label):
    #     '''
    #     :param logit: NCHW or NCDHW, C=1
    #     :param label: NHW or NDHW
    #     :return:
    #     '''
    #     pred = pred.view(pred.size(0), -1)  # N,C,H,W => N, H*W (C=1)
    #     label = label.view(label.size(0), -1)  # N,H,W => N, H*W
    #
    #     BCE = F.binary_cross_entropy(pred, label, reduction='none')
    #     loss= ((1 - torch.exp(-BCE)) ** self.gamma) * BCE
    #
    #     return loss.mean(dim=1, keepdim=False)
    #     # return loss.sum(dim=1, keepdim=False)
    #
    # def forward_custom_mean(self, pred, label):
    #     '''
    #     :param logit: NHW or NDHW, C=1
    #     :param label: NHW or NDHW
    #     :return:
    #     '''
    #
    #     pred = pred.view(pred.size(0), -1)  # N,C,H,W => N, H*W (C=1)
    #     label = label.view(label.size(0), -1)  # N,H,W => N, H*W
    #
    #     BCE = F.binary_cross_entropy(pred, label, reduction='none')
    #     loss= ((1 - torch.exp(-BCE)) ** self.gamma) * BCE
    #
    #     # return loss.mean(dim=1, keepdim=False)
    #     return (loss.sum(dim=1, keepdim=False)/self.division_factor)


class DiceLoss(nn.Module):
    def __init__(self, weight=1.0, if_square = False):
        super(DiceLoss, self).__init__()
        self.weight = float(weight)
        self.sigmoid = nn.Sigmoid()
        if if_square:
            self.magnitude = self.get_magnitude_l2
        else:
            self.magnitude = self.get_magnitude_l1

    def get_magnitude_l1(self, x):
        return torch.sum(x, dim=1, keepdim=False)

    def get_magnitude_l2(self, x):
        return torch.sum(x * x, dim=1, keepdim=False)

    def forward(self, logit, label):
        '''

        :param pred: NHW or NDHW
        :param label: NHW or NDHW
        :return:
        '''
        pred = self.sigmoid(logit)
        pred = pred.view(pred.size(0), -1)  # NHW or NDHW => N * ~
        label = label.view(label.size(0), -1)  # NHW or NDHW => N * ~

        numerator = 2.0 * torch.sum( torch.mul(pred, label), dim=1, keepdim=False)
        denominator = self.magnitude(pred) + self.magnitude(label)

        # print(f'numerator size: {numerator.size()}')
        # print(f'denominator size: {denominator.size()}')

        return -self.weight * torch.div(numerator+0.00000001, denominator+0.00000001)


class DiceLoss_background(nn.Module):
    def __init__(self, weight=1.0, if_square = False):
        super(DiceLoss_background, self).__init__()
        self.weight = float(weight)
        self.sigmoid = nn.Sigmoid()
        if if_square:
            self.magnitude = self.get_magnitude_l2
        else:
            self.magnitude = self.get_magnitude_l1

    def get_magnitude_l1(self, x):
        return torch.sum(x, dim=1, keepdim=False)

    def get_magnitude_l2(self, x):
        return torch.sum(x * x, dim=1, keepdim=False)

    def forward(self, logit, label):
        '''

        :param pred: NHW or NDHW
        :param label: NHW or NDHW
        :return:
        '''
        pred = self.sigmoid(logit)
        pred_fg = pred.view(pred.size(0), -1)  # NHW or NDHW => N * ~
        label_fg = label.view(label.size(0), -1)  # NHW or NDHW => N * ~
        pred_bg = 1.0 - pred_fg
        label_bg = 1.0 - label_fg

        numerator_fg = 2.0 * torch.sum( torch.mul(pred_fg, label_fg), dim=1, keepdim=False)
        denominator_fg = self.magnitude(pred_fg) + self.magnitude(label_fg)

        numerator_bg = 2.0 * torch.sum( torch.mul(pred_bg, label_bg), dim=1, keepdim=False)
        denominator_bg = self.magnitude(pred_bg) + self.magnitude(label_bg)

        return -self.weight * ( 0.5*torch.div(numerator_fg+0.00000001, denominator_fg+0.00000001) +
                                0.5*torch.div(numerator_bg+0.00000001, denominator_bg+0.00000001) )