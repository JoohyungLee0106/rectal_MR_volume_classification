# LAST updated: 20201120 12:25
import configparser
from configparser import ExtendedInterpolation
import pandas as pd
import argparse
import copy
import os
import random
import shutil
import time
import statistics
import warnings
import numpy as np
import math
# from .utils import loss as losses
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import utils.loss as losses
from sklearn import metrics
from utils.augmentation import augmentation
import utils.data_provider as custom_datasets
# model = models.__dict__[args.arch](pretrained=True)

from utils.augmentation import augmentation

# import model.guided_attention as models
import model.guided_attention_slice_att as models
# import model.self_attention_module as self_attention

# import model.guided_attention as models
import matplotlib.pyplot as plt


# is model restoration (after patience) better than conventional method?
# network 랑 network_attention 합치거나 dict 를 합치거나

def parse_int_tuple(input):
    return tuple(int(k.strip()) for k in input[1:-1].split(','))


config = configparser.ConfigParser(converters={'tuple': parse_int_tuple}, interpolation=ExtendedInterpolation())
config.read('config_rectumcrop_crc.ini')
node_list = list(config['node'].keys())

parser = argparse.ArgumentParser(description='PyTorch Image Classification Training')
parser.add_argument('-n', '--node', default='',
                    choices=node_list,
                    help='model architecture: ' +
                         ' | '.join(node_list) +
                         ' (default: kaist_desktop)')
# temp!!!
parser.add_argument('-w', '--workers', default=8, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('-g', '--gpu', nargs="+", default=0, type=int,
                    help='(List of) GPU id(s) to use.')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('-l', '--loss', nargs="+", default='',
#                     choices=['category', 'rectum', 'cancer'],
#                     help='Loss computation from category/rectum/cancer  (default: category)')
args = parser.parse_args()

# temp!
# kaist_server, kaist_desktop, ncc_jhl, ncc_oh, ncc_or, ncc_server, ncc_rtx, ncc_image, ncc_titan2
args.node = 'kaist_server'
args.gpu = [0,1,2]
GPU_ID = "0,1,2"
# args.node = 'kaist_desktop'
# args.gpu = [0,1]
# GPU_ID = "0,1"
# config[args.node]['output_device'] = '1'

args.workers = 24
# resnet_2d, r3d, mc3, rmc3
# config['loss']['category'] = 'FocalLoss_rankingloss_prob_relative'
config['loss']['category'] = 'FocalLoss_rankingloss_prob_label'
# config['by_exp']['dimension'] = '2D'
config['by_exp']['dimension'] = '3D'
# config['by_exp']['network'] = 'encoder'
# config['by_exp']['network'] = 'Attention'
config['by_exp']['network'] = 'Attention_custom'

config['by_exp']['if_half_precision'] = 'False'
config['default']['batch_size'] = '8'
# config['by_exp']['save_group_name'] = 'cvpr2021_rc'

# config['loss_arg']['category'] = '2.0'
config['loss_arg']['rectum'] = '2.0'
config['loss_arg']['cancer'] = '1.0'
config['default']['end_patience'] = '40'

if config['by_exp']['network'] == 'Attention_custom':
    if config['by_exp']['dimension'] == '2D':
        config['by_exp']['network'] = 'Attention_custom_2d'
        config['by_exp']['mask_size'] = '(64,64)'
    elif config['by_exp']['dimension'] == '3D':
        config['by_exp']['network'] = 'Attention_custom_3d'
        config['by_exp']['mask_size'] = '(7,64,64)'
else:
    if config['by_exp']['dimension'] == '2D':
        config['by_exp']['mask_size'] = '(256,256)'
    elif config['by_exp']['dimension'] == '3D':
        config['by_exp']['mask_size'] = '(7,256,256)'

args.resume = False
args.train = True
# 0 if do not want to perform the test
args.test_aug = 10
args.test_noaug = 1

# args.att_dtype = torch.half
args.att_dtype = torch.float

# 'keys': ['category', 'rectum']
# NETWORK_PARAM = {'cc_layer': 3, 'att_layer': 2, 'pool2': nn.AdaptiveAvgPool3d((1, 1, 1)), 'data_type':torch.float, 'block': models.BasicBlock,
#                  'starting_feature_num': 32, 'layers' : [3,4,6,3], 'backbone': models.r3d}
# NETWORK_PARAM = {'cc_layer': 3, 'att_layer': 3, 'pool2': nn.AdaptiveAvgPool3d((1, 1, 1)), 'data_type':torch.float, 'block': models.BasicBlock,
#                  'starting_feature_num': 32, 'layers' : [3,4,6,3], 'keys': ['category', 'cancer'], 'backbone': models.r3d}
# NETWORK_PARAM = {'cc_layer': 3, 'att_layer': 2, 'pool2': nn.AdaptiveAvgPool2d((1, 1)), 'data_type':torch.float, 'block': models.BasicBlock,
#                  'starting_feature_num': 32, 'layers' : [3,4,6,3], 'backbone': models.resnet_2d}
# Bottleneck , BasicBlock , 'att_type': 'SE', 'CBAM', None , 'if_nl': [False]*3 , [True,True,False]

NETWORK_PARAM = {'resnet_type': 18, 'encoder_dim_type': 'rmc4', 'att_type': ['SE']*4, 'stride_D1': [1, 1, 1, 1]}


# if config['by_exp']['dimension'] == '2D':
#     NETWORK_PARAM = {'resnet_type': 18, 'encoder_dim_type': '2d', 'att_type': ['CBAM']*4, 'stride_D1': [1,1,1,1]}
# elif config['by_exp']['dimension'] == '3D':
#     NETWORK_PARAM = {'resnet_type': 18, 'encoder_dim_type': 'rmc6', 'att_type': [False]*4, 'stride_D1': [1,1,1,1]}

if config['by_exp']['network'] == 'Attention':
    DECODER_PARAM = {'decoder_type': 'fcn', 'if_deconv': True}
    DECODER = DECODER_PARAM['decoder_type']
elif 'Attention_custom' in config['by_exp']['network']:
    DECODER_PARAM = {}
    DECODER = 'custom'

NETWORK_PARAM.update(DECODER_PARAM)


config['by_exp']['save_folder_name'] = 'fin_framewise'+'_loss_'+str(config.get('loss', 'category')) + '_'+str(args.node)

dir_results = os.path.join(config.get(args.node, 'result_save_directory'), config.get('by_exp', 'save_folder_name'))


#######################################################################################################################3

try:
    os.mkdir(config.get(args.node, 'result_save_directory'))
except:
    pass
try:
    os.mkdir(os.path.join(config.get(args.node, 'result_save_directory'), config.get('by_exp', 'save_folder_name')))
except:
    pass


def main(fold, performance_metric_tr, performance_metric_val, performance_metric_stat):
    print('MAIN!!!')
    best_loss = 99999
    best_epoch = -1
    # global best_loss
    dir_results_fold = os.path.join(dir_results, 'fold_' + str(fold))
    try:
        os.mkdir(dir_results_fold)
    except:
        pass

    ####################################################

    reset_seed()

    # if config.getboolean('by_exp', 'if_half_precision'):
    #     model = (models.__dict__[config.get('by_exp', 'network')](**NETWORK_PARAM)).half()
    #     for layer in model.modules():
    #         if isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm3d)):
    #             layer.float()
    # else:
    #     model = models.__dict__[config.get('by_exp', 'network')](**NETWORK_PARAM)
    model = models.__dict__[config.get('by_exp', 'network')](**NETWORK_PARAM)

    optimizer = torch.optim.SGD(model.parameters(), config.getfloat('default', 'lr_init'),
                                momentum=config.getfloat('default', 'momentum'),
                                weight_decay=config.getfloat('default', 'weight_decay'))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=config.getfloat('default', 'scheduler_factor'),
                                                           patience=config.getfloat('default', 'scheduler_patience'))

    # restore_and_save_model_init(args, model, fold, dir_results)
    # restore_and_save_optimizer_init(optimizer, fold, dir_results)
    # print('start')
    # print(model.state_dict().keys())
    # print('end')
    if args.train:
        restore_and_save_model_init(args, model, optimizer, fold, dir_results)
    # print(model.backbone.stem[0].weight.size())
    if len(args.gpu) == 1:
        warnings.warn('You have chosen a specific GPU. This will disable data parallelism.')
        print("Use GPU: {} for training".format(args.gpu[0]))

        torch.cuda.set_device(int(args.gpu[0]))
        # network params are required. it can be stored in config if stored in json
        model = model.cuda()
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
        # network params are required. it can be stored in config if stored in json

        model = torch.nn.DataParallel(model, device_ids=args.gpu,
                                      output_device=config.getint(args.node, 'output_device')).cuda()

    criterion = {}
    # unnecessary scatter/gather
    for key in config['loss'].keys():
        criterion[key] = (losses.__dict__[config.get('loss', key)])(config.getfloat('loss_arg', key)).cuda()


    train_dataset = custom_datasets.CustomDataset(config.get(args.node, 'csv_tr') + str(fold) + '.csv',
                                                  dimension=config.get('by_exp', 'dimension'),
                                                  if_with_mask=config.getboolean('by_exp', 'if_with_mask'),
                                                  mask_size=config['by_exp'].gettuple('mask_size'),
                                                  image_size=config.getint('default', 'image_size'),
                                                  if_replace_path=config.getboolean('default', 'if_replace_path'),
                                                  if_half_precision=config.getboolean('by_exp', 'if_half_precision'),
                                                  path_to_be_replaced=config.get('default', 'path_to_be_replaced'),
                                                  path_new=config.get(args.node, 'path_new'),
                                                  transform=augmentation(config.getint('default', 'image_size')))
    # , transform = augmentation(config.getint('default', 'image_size'))
    # val_dataset = custom_datasets.CustomDataset(config.get(args.node, 'csv_val')+ str(fold) + '.csv', dimension=config.get('by_exp', 'dimension'),
    #                              if_with_mask=config.getboolean('by_exp', 'if_with_mask'), mask_size=config['by_exp'].gettuple('mask_size'), image_size=config.getint('default', 'image_size'),
    #                             if_replace_path=config.getboolean('default', 'if_replace_path'), if_half_precision = config.getboolean('by_exp', 'if_half_precision'),
    #                             path_to_be_replaced=config.get('default', 'path_to_be_replaced'),
    #                             path_new=config.get(args.node, 'path_new'))

    test_dataset_aug = custom_datasets.CustomDataset(config.get(args.node, 'csv_val') + str(fold) + '.csv',
                                                       dimension=config.get('by_exp', 'dimension'),
                                                       if_with_mask=config.getboolean('by_exp', 'if_with_mask'),
                                                       mask_size=config['by_exp'].gettuple('mask_size'),
                                                       image_size=config.getint('default', 'image_size'),
                                                       if_replace_path=config.getboolean('default', 'if_replace_path'),
                                                       if_half_precision=config.getboolean('by_exp',
                                                                                           'if_half_precision'),
                                                       path_to_be_replaced=config.get('default', 'path_to_be_replaced'),
                                                       path_new=config.get(args.node, 'path_new'),
                                                       transform=augmentation(config.getint('default', 'image_size'),
                                                                              False))

    test_dataset_noaug = custom_datasets.CustomDataset(config.get(args.node, 'csv_val') + str(fold) + '.csv',
                                                 dimension=config.get('by_exp', 'dimension'),
                                                 if_with_mask=config.getboolean('by_exp', 'if_with_mask'),
                                                 mask_size=config['by_exp'].gettuple('mask_size'),
                                                 image_size=config.getint('default', 'image_size'),
                                                 if_replace_path=config.getboolean('default', 'if_replace_path'),
                                                 if_half_precision=config.getboolean('by_exp', 'if_half_precision'),
                                                 path_to_be_replaced=config.get('default', 'path_to_be_replaced'),
                                                 path_new=config.get(args.node, 'path_new'),
                                                 transform=augmentation(config.getint('default', 'image_size'), True),
                                                 if_repr_num = True)

    if config.getboolean('default', 'oversampling'):
        train_sampler = set_sampler(fold)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.getint('default', 'batch_size'), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=config.getint('default', 'batch_size'), shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    test_loader_aug = torch.utils.data.DataLoader(
        test_dataset_aug, batch_size=config.getint('default', 'batch_size'), shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader_noaug = torch.utils.data.DataLoader(
        test_dataset_noaug, batch_size=config.getint('default', 'batch_size'), shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # optionally resume from a checkpoint
    if args.resume:
        best_loss = restore_model(args, fold, model, optimizer)
    else:
        args.start_epoch = 1

    if args.train:
        reset_seed()
        # TEMP
        # t1=time.time()
        for epoch in range(args.start_epoch, config.getint('default', 'max_epoch')):
            # print(f'time per epoch: {time.time()-t1}')
            # t1 = time.time()
        # for epoch in range(3):

            # train for one epoch
            loss_all = train(train_loader, model, criterion, optimizer, performance_metric_tr, args)

            for obj in AverageMeter.instances_tr:
                obj.store()
                obj.reset_epoch()
                obj.save(dir_results_fold)

            # remember best acc@1 and save checkpoint
            # print(f'current key: {key_metric.avg[-1]}, so far the best: {best_loss}')
            is_best = loss_all < best_loss
            best_loss = min(loss_all, best_loss)

            if is_best:
                best_epoch = epoch
                print(f'Best model at epoch {best_epoch}')
                for obj in AverageMeter.instances_tr:
                    obj.store_best_state()
                save_checkpoint({
                    'epoch': epoch + 1,
                    'network': config.get('by_exp', 'network'),
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'best_epoch': best_epoch,
                    'optimizer': optimizer.state_dict(),
                }, dir_results_fold)
            lr = optimizer.param_groups[-1]['lr']
            scheduler.step(loss_all)
            if lr != optimizer.param_groups[-1]['lr']:
                print(f'Scheduler activated!!! : old ({lr}), new ({optimizer.param_groups[-1]["lr"]})')
                if len(args.gpu) == 1:
                    loc = 'cuda:{}'.format(args.gpu[0])
                    checkpoint = torch.load(dir_results_fold + '/model_best.pth.tar', map_location=loc)
                else:
                    checkpoint = torch.load(dir_results_fold + '/model_best.pth.tar')
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                optimizer.param_groups[-1]['lr'] = lr * config.getfloat('default', 'scheduler_factor')

            if (optimizer.param_groups[-1]['lr'] < float(config.get('default', 'lr_min'))) or \
                    ((epoch - best_epoch) > config.getint('default', 'end_patience')):
                break
            # print(f'TIME PER ONE EPOCH: {time.time() - t1}')
        for obj in AverageMeter.instances_tr:
            obj.reset_epoch()
            obj.update_best()
            obj.save(dir_results_fold)
            obj.reset_avg()

    if args.test_aug:

        # optionally resume from a checkpoint
        _ = restore_model(args, fold, model, optimizer)
        reset_seed()
        for epoch in range(args.test_aug):
            print(f'<Evaluation> Fold: {fold}, Epoch {epoch + 1} started !!!')
            validate(test_loader_aug, model, performance_metric_stat, args, dir_results_fold)

        for obj in PerformanceMetrics.instances_test:
            # print(f'obj id: {obj.id}')
            # obj.store()
            # obj.store_best_state()
            obj.update_last(repetition=args.test_aug,aug_type = 'aug')
            obj.reset_epoch()

    if args.test_noaug:

        # optionally resume from a checkpoint
        _ = restore_model(args, fold, model, optimizer)

        # for epoch in range(args.test_noaug):
        #     print(f'<Evaluation> Fold: {fold}, Epoch {epoch + 1} started !!!')
        reset_seed()

        if (fold == 1) and config.getboolean('by_exp', 'if_with_mask'):
            validate_with_att_map(test_loader_noaug, model, performance_metric_stat, args, dir_results_fold)
        else:
            validate(test_loader_noaug, model, performance_metric_stat, args, dir_results_fold)

        if NETWORK_PARAM['att_type'] != [False] * 4:
            if len(args.gpu) == 1:
                model.if_return_att_map(False)
            else:
                model.module.if_return_att_map(False)

        for obj in PerformanceMetrics.instances_test:
            print(f'obj id: {obj.id}')
            # obj.store()
            # obj.store_best_state()
            obj.update_last(repetition=args.test_noaug, aug_type = 'noaug')
            obj.reset_epoch()



    # print(f'TIME PER 10 EPOCH: {time.time() - t1}')


# end main
def reset_seed():
    if config.get('default', 'seed') is not None:
        random.seed(config.getint('default', 'seed'))
        np.random.seed(config.getint('default', 'seed'))
        torch.cuda.manual_seed_all(config.getint('default', 'seed'))
        torch.manual_seed(config.getint('default', 'seed'))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        cudnn.benchmark = True

def set_sampler(fold):
    csv_file = pd.read_csv(config.get(args.node, 'csv_tr') + str(fold) + '.csv')
    stage = csv_file['t-stage']

    label, counts = np.unique(stage, return_counts=True)
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    weights = weights.double()

    temp = stage.to_numpy(dtype='double')
    temp2 = stage.to_numpy(dtype='double')
    ww = weights.numpy()

    temp2[np.where(temp == label[0])] = ww[0]
    temp2[np.where(temp == label[1])] = ww[1]

    sample_weights = torch.from_numpy(temp2)
    # print(f'sample_weights: {sample_weights}')
    return torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))


def train(train_loader, model, criterion, optimizer, performance_metric, args):
    # switch to train mode
    loss_all = 0
    cnt = 0

    model.train()

    for i, data in enumerate(train_loader):
        repr_num_list = []
        for repr_num in data['repr_num']:
            repr_num= repr_num.replace('[', '')
            repr_num=repr_num.replace(']', '')
            repr_num_list.append([int(s) for s in repr_num.split(', ')])

        if len(args.gpu) == 1:
            data['image'] = data['image'].cuda(args.gpu[0], non_blocking=True)
            for key in config['loss'].keys():
                data[key] = data[key].cuda(args.gpu[0], non_blocking=True)

        data['image'] = data['image'].cuda(non_blocking=True)

        # classification output: logit
        # segmentation output: logit
        output = model(data['image'])
        loss = 0

        for key in config['loss'].keys():
            if key == 'category':
                output['category'] = output['category'].cuda(non_blocking=True)
                data['category'] = data['category'].cuda(non_blocking=True)

                loss += criterion['category'](output["category"], data["category"], repr_num_list)
                for metric in performance_metric['category']:
                    if metric.metric == 'repr_num':
                        metric.update(output['category'], repr_num_list)
                    else:
                        metric.update(output['category'], data['category'])
            else:
                output[key] = output[key].cuda(non_blocking=True)
                data[key] = data[key].cuda(non_blocking=True)
                loss += criterion[key](output[key], data[key])
                for metric in performance_metric[key]:
                    metric.update(output[key], data[key])

        loss_all += torch.sum(loss)
        cnt += loss.size(0)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        (torch.mean(loss)).backward()
        optimizer.step()
    return loss_all / float(cnt)


# key_metric
def validate(val_loader, model, performance_metric, args, dir_results_fold):

    model.eval()
    sample_size = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            repr_num_list = []
            for repr_num in data['repr_num']:
                repr_num= repr_num.replace('[', '')
                repr_num=repr_num.replace(']', '')
                repr_num_list.append([int(s) for s in repr_num.split(', ')])
            sample_size += data["image"].size(0)
            if len(args.gpu) == 1:
                data['image'] = data['image'].cuda(args.gpu[0], non_blocking=True)
                for key in config['loss'].keys():
                    data[key] = data[key].cuda(args.gpu[0], non_blocking=True)

            data['image'] = data['image'].cuda(non_blocking=True)
            # classification output: logit
            # segmentation output: logit
            output = model(data['image'])
            for key in config['loss'].keys():
                # output[key] = output[key].to(torch.device("cpu"))
                # data[key] = data[key].to(torch.device("cpu"))
                output[key] = output[key].cuda(non_blocking=True)
                data[key] = data[key].cuda(non_blocking=True)
                if key == 'category':
                    for metric in performance_metric[key]:
                        metric.update(output[key], data[key], repr_num_list)
                else:
                    for metric in performance_metric[key]:
                        metric.update(output[key], data[key])


            torch.save(torch.sigmoid(output['category']),
                       os.path.join(dir_results_fold, 'cls_probs_' + str(sample_size) + '.pth.tar'))

def validate_with_att_map(val_loader, model, performance_metric, args, dir_results_fold):
    if NETWORK_PARAM['att_type'] != [False] * 4:
        if len(args.gpu) == 1:
            model.if_return_att_map(True)
        else:
            model.module.if_return_att_map(True)
    att_map = {'patient_id': []}
    att_map.update(
        {'repr_pred': torch.tensor([], dtype=args.att_dtype, requires_grad=False, device=torch.device('cpu:0'))}
    )

    if config['by_exp']['dimension'] == '3D':
        # att_map.update({'repr_num': torch.tensor([], dtype=torch.float, requires_grad=False, device=torch.device('cpu:0'))})
        att_map.update({'repr_num': []})

    if (NETWORK_PARAM['att_type'][1] == 'CBAM') or (NETWORK_PARAM['att_type'][1] == 'SE'):
        att_map.update(
            {'layer1': torch.tensor([], dtype=args.att_dtype, requires_grad=False, device=torch.device('cpu:0')),
           'layer2': torch.tensor([], dtype=args.att_dtype, requires_grad=False, device=torch.device('cpu:0')),
           'layer3': torch.tensor([], dtype=args.att_dtype, requires_grad=False, device=torch.device('cpu:0')),
           'layer4': torch.tensor([], dtype=args.att_dtype, requires_grad=False, device=torch.device('cpu:0'))}
            )
    elif NETWORK_PARAM['att_type'][1] == 'NL':
        for i in range(len(NETWORK_PARAM['att_type'])):
            if NETWORK_PARAM['att_type'][i]:
                att_map.update({'layer'+str(i+1) : torch.tensor([], dtype=args.att_dtype, requires_grad=False, device=torch.device('cpu:0'))})

    if config.getboolean('by_exp', 'if_with_mask'):
        att_map.update(
            {'rectum': torch.tensor([], dtype=args.att_dtype, requires_grad=False, device=torch.device('cpu:0')),
             'cancer': torch.tensor([], dtype=args.att_dtype, requires_grad=False, device=torch.device('cpu:0'))}
        )
    att_map_clean = copy.deepcopy(att_map)
    # switch to evaluate mode
    model.eval()
    sample_size = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            repr_num_list = []
            for repr_num in data['repr_num']:
                repr_num= repr_num.replace('[', '')
                repr_num=repr_num.replace(']', '')
                repr_num_list.append([int(s) for s in repr_num.split(', ')])
            sample_size += data["image"].size(0)
            # print(f'i: {i}')
            if len(args.gpu) == 1:
                data['image'] = data['image'].cuda(args.gpu[0], non_blocking=True)
                for key in config['loss'].keys():
                    data[key] = data[key].cuda(args.gpu[0], non_blocking=True)

            data['image'] = data['image'].cuda(non_blocking=True)

            output = model(data['image'])

            for key in config['loss'].keys():
                # output[key] = output[key].to(torch.device("cpu"))
                # data[key] = data[key].to(torch.device("cpu"))
                output[key] = output[key].cuda(non_blocking=True)
                data[key] = data[key].cuda(non_blocking=True)
                if key == 'category':
                    for metric in performance_metric[key]:
                        metric.update(output[key], data[key], repr_num_list)
                else:
                    for metric in performance_metric[key]:
                        metric.update(output[key], data[key])

            if config.getboolean('by_exp', 'if_with_mask'):
                att_map['rectum'] = torch.cat((att_map['rectum'], torch.sigmoid(output['rectum'].clone()).cpu()), dim=0)
                att_map['cancer'] = torch.cat((att_map['cancer'], torch.sigmoid(output['cancer'].clone()).cpu()), dim=0)

            # print(f'repr_num: {data["repr_num"]}')
            if config['by_exp']['dimension'] == '3D':
                att_map['repr_num'].extend(data['repr_num'])
                torch.cat((att_map['repr_pred'], torch.sigmoid(output['category']).clone().cpu()), dim=0)

            att_map['patient_id'].extend(data['id'])

            if (NETWORK_PARAM['att_type'][1] == 'CBAM' or NETWORK_PARAM['att_type'][1] == 'SE' or NETWORK_PARAM['att_type'][1] == 'NL'):
                for key in att_map.keys():
                    if 'layer' in key:
                        att_map[key] = torch.cat((att_map[key], output[key].clone().cpu()), dim=0)

                if len(args.gpu) > 1:
                    model.module.clear_att_map()
                else:
                    model.clear_att_map()

            torch.save(att_map, os.path.join(dir_results_fold, 'att_map_'+str(sample_size)+'.pth.tar'))
            torch.save(torch.sigmoid(output['category']), os.path.join(dir_results_fold, 'cls_probs_' + str(sample_size) + '.pth.tar'))
            att_map = copy.deepcopy(att_map_clean)


def save_checkpoint(state, filename):
    torch.save(state, os.path.join(filename, 'model_best.pth.tar'))


def restore_model(args, fold, model, optimizer):
    # optionally resume from a checkpoint
    print('resume!!!')
    load_path = os.path.join(dir_results, 'fold_' + str(fold), 'model_best.pth.tar')
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(dir_results))
        if len(args.gpu) > 1:
            checkpoint = torch.load(load_path)
        elif len(args.gpu) == 1:
            loc = 'cuda:{}'.format(args.gpu[0])
            checkpoint = torch.load(load_path, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_epoch = checkpoint['best_epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(dir_results))
        raise ValueError("<restore_model_init>")

    return best_loss

def restore_and_save_model_init(args, model, optimizer, fold, dir_results):
    print('init!!!')
    if fold == -1:
        if (NETWORK_PARAM['att_type'] == [False] * 4):
            __strict = True
        else:
            __strict = False

        if config.getboolean('by_exp', 'if_with_mask'):
            __strict = False
            load_path = 'model_init_' + str(NETWORK_PARAM['resnet_type']) + '_' + str(NETWORK_PARAM['encoder_dim_type']) + '_' + str(NETWORK_PARAM['att_type'][1]) +'_custom.pth.tar'
        else:
            load_path = 'model_init_'+str(NETWORK_PARAM['resnet_type'])+'_'+str(NETWORK_PARAM['encoder_dim_type'])+'.pth.tar'

        if os.path.isfile(load_path):
            print(f'=> loading checkpoint {load_path}, strict:{__strict}')
            if len(args.gpu) > 1:
                checkpoint = torch.load(load_path)
            elif len(args.gpu) == 1:
                loc = 'cuda:{}'.format(args.gpu[0])
                checkpoint = torch.load(load_path, map_location=loc)
            if config.getboolean('by_exp', 'if_with_mask'):
                __keys = list(checkpoint['state_dict'].keys())
                for key in __keys:
                    checkpoint['state_dict']['encoder.'+key] = checkpoint['state_dict'].pop(key)
            # args.start_epoch = checkpoint['epoch']
            # best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'], strict=__strict)
            if (NETWORK_PARAM['att_type'] == [False] * 4) and (not config.getboolean('by_exp', 'if_with_mask')):
                optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found")
            raise ValueError("<restore_model_init>")
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(dir_results, 'model_init.pth.tar'))
    else:
        if len(args.gpu) > 1:
            # checkpoint = torch.load(os.path.join(dir_results, 'model_init.pth.tar'))
            checkpoint = torch.load('model_init.pth.tar')
        elif len(args.gpu) == 1:
            loc = 'cuda:{}'.format(args.gpu[0])
            # checkpoint = torch.load(os.path.join(dir_results, 'model_init.pth.tar'), map_location=loc)
            checkpoint = torch.load('model_init.pth.tar', map_location=loc)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])


def adjust_learning_rate(optimizer, lr_old):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = lr_old * (0.1 ** (1.0 / 100.0))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PerformanceMetrics(object):
    instances_tr = []
    instances_val = []
    instances_test = []

    performance_tr = {}
    performance_val = {}
    performance_test = {}

    def __init__(self, data_type, metric):
        assert isinstance(data_type, str)
        assert isinstance(metric, str)

        self.data_type = data_type
        self.metric = metric
        self.id = data_type + '_' + metric
        self.best = 0

        if self.data_type == 'tr':
            PerformanceMetrics.instances_tr.append(self)
        elif self.data_type == 'val':
            PerformanceMetrics.instances_val.append(self)
        elif self.data_type == 'test':
            PerformanceMetrics.instances_test.append(self)

    def update_last(self, metric, value):
        if self.data_type == 'tr':
            if metric in PerformanceMetrics.performance_tr.keys():
                PerformanceMetrics.performance_tr[metric].append(value)
            else:
                PerformanceMetrics.performance_tr[metric] = [value]
        elif self.data_type == 'val':
            if metric in PerformanceMetrics.performance_val.keys():
                PerformanceMetrics.performance_val[metric].append(value)
            else:
                PerformanceMetrics.performance_val[metric] = [value]
        elif self.data_type == 'test':
            if metric in PerformanceMetrics.performance_test.keys():
                PerformanceMetrics.performance_test[metric].append(value)
            else:
                PerformanceMetrics.performance_test[metric] = [value]

    def store_best_state(self):
        pass

    def store(self):
        pass


class AverageMeter(PerformanceMetrics):
    """Computes and stores the average and current value"""

    def __init__(self, data_type, metric):
        super(AverageMeter, self).__init__(data_type, metric)

        self.reset_epoch()
        self.avg = []
        self.best = 0

        self.metric = metric

        if metric == 'accuracy':
            self.update = self.update_accuracy
        elif metric == 'sensitivity':
            self.update = self.update_sensitivity
        elif metric == 'specificity':
            self.update = self.update_specificity
        elif metric[:4] == 'dice':
            self.update = self.update_dice
        elif metric == 'repr_num':
            self.update = self.update_repr_num

    def reset_epoch(self):
        self.sum = 0
        self.count = 0

    def reset_avg(self):
        self.avg = []

    def store_best_state(self):
        self.best = self.avg[-1]

    def store(self):
        # Temp!!
        # if self.count == 0:
        #     self.avg.append(-1)
        # else:
        self.avg.append(self.sum / float(self.count))

    def update_best(self):
        super(AverageMeter, self).update_last(self.metric, self.best)
        self.avg.append(self.best)

    def update_last(self, repetition = 1, aug_type=''):
        super(AverageMeter, self).update_last(self.metric+'_'+aug_type, self.sum/float(self.count))

    def update_accuracy(self, logit_tensor, target_tensor):
        logit_tensor = torch.max(logit_tensor, dim=1)[0]
        pred_tensor = F.relu((logit_tensor).sign())
        val_list = (pred_tensor == target_tensor).tolist()
        self.sum += sum(val_list)
        self.count += len(val_list)

        # print(f'<{self.id}> output: {logit_tensor.tolist()}, pred: {pred_tensor.tolist()}, target: {target_tensor.tolist()} => {val_list}')
        # print(f'sum: {sum(val_list)}')

    def update_sensitivity(self, logit_tensor, target_tensor):
        logit_tensor = torch.max(logit_tensor, dim=1)[0]
        pred_tensor = F.relu(logit_tensor.sign())
        val_list = (pred_tensor * target_tensor).tolist()
        self.sum += sum(val_list)
        self.count += sum(target_tensor.tolist())
        # print(f'<{self.id}> output: {logit_tensor.tolist()}, pred: {pred_tensor.tolist()}, target: {target_tensor.tolist()} => {val_list}')

    def update_specificity(self, logit_tensor, target_tensor):
        logit_tensor = torch.max(logit_tensor, dim=1)[0]
        pred_tensor = F.relu(logit_tensor.sign())
        val_list = ((1. - pred_tensor) * (1. - target_tensor)).tolist()
        self.sum += sum(val_list)
        self.count += (len(val_list) - sum(target_tensor.tolist()))
        # print(f'<{self.id}> logit: {logit_tensor.tolist()}, pred: {pred_tensor.tolist()}, target: {target_tensor.tolist()} => {val_list}')

    def update_repr_num(self, logit, target_repr_num):
        pred = torch.argmax(logit, dim = 1, keepdim=False)
        list_comp = [1 if pred[i] in target_repr_num[i] else 0 for i in range(logit.size(0))]
        self.sum += sum(list_comp)
        self.count += len(list_comp)

    def update_dice(self, logit_tensor, target_tensor):
        pred_tensor = F.relu(logit_tensor.sign())
        pred_tensor = pred_tensor.view(pred_tensor.size(0), -1)  # NHW or NDHW => N * ~
        target_tensor = target_tensor.view(target_tensor.size(0), -1)  # NHW or NDHW => N * ~

        intersection = torch.sum(pred_tensor * target_tensor, dim=1, keepdim=False) + 0.00000001
        denominator = torch.sum(pred_tensor + target_tensor, dim=1, keepdim=False) + 0.00000001
        val_list = (intersection * 2.0 / denominator).tolist()

        self.sum += sum(val_list)
        self.count += len(val_list)

    def save(self, save_dir):
        with open(os.path.join(save_dir, self.id + '.txt'), "w") as output:
            output.write(str(self.avg))


class StatisticalAnalysis(PerformanceMetrics):
    """Computes and stores various values ONLY after finish training"""

    def __init__(self, data_type, metric, args):
        super(StatisticalAnalysis, self).__init__(data_type, metric)
        self.args = args
        self.reset_epoch()

    def reset_epoch(self):
        if len(args.gpu) == 1:
            self.logit_tensor = torch.tensor([], requires_grad=False).cuda(self.args.gpu[0])
            self.target_tensor = torch.tensor([], requires_grad=False, dtype=torch.double).cuda(self.args.gpu[0])
            # self.repr_num_pred = torch.tensor([], requires_grad=False)
            # self.repr_num_target = []
        else:
            self.logit_tensor = torch.tensor([], requires_grad=False).cuda()
            self.target_tensor = torch.tensor([], requires_grad=False, dtype=torch.double).cuda()
            # self.repr_num_pred = torch.tensor([], requires_grad=False)
        self.repr_num_target =[]

    def update(self, logit_tensor, target_tensor, target_list):
        # logit_tensor = torch.max(logit_tensor, dim=1)[0]
        self.logit_tensor = torch.cat((self.logit_tensor, logit_tensor.detach().clone()), dim=0)
        self.target_tensor = torch.cat((self.target_tensor, target_tensor.detach().clone()), dim=0)
        self.repr_num_target.extend(list(target_list))


    def check(self):
        if torch.prod(torch.max(self.target_tensor, dim=0, keepdim=False)[0] ==
                      torch.min(self.target_tensor, dim=0, keepdim=False)[0]) == 0:
            raise ValueError('Target tensor does NOT equal')
        else:
            print(f'Checked well !!!')

    def update_last(self, repetition = 1, aug_type=''):
        # print(f'size all: {self.logit_tensor.size(0)}, repetition: {repetition}')
        self.logit_tensor = self.logit_tensor.view(repetition, int(self.logit_tensor.size(0)/repetition), -1)
        self.target_tensor = self.target_tensor.view(repetition, -1)

        self.check()

        repr_num_pred = list((torch.argmax(torch.mean(self.logit_tensor, dim=0, keepdim=False), dim=1)).tolist())
        self.logit_tensor = torch.max(torch.mean(self.logit_tensor, dim=0, keepdim=False), dim=1)[0]
        self.target_tensor = self.target_tensor[0, :]

        self.repr_num_target = list(self.repr_num_target[:int(len(self.repr_num_target)/float(repetition))])
        repr_num_acc_list = [1 if repr_num_pred[i] in self.repr_num_target[i] else 0 for i in range(len(repr_num_pred))]
        repr_num_acc_all = sum(repr_num_acc_list) / float(len(repr_num_acc_list))

        repr_num_acc_t2_list = (torch.tensor(repr_num_acc_list, device=self.target_tensor.get_device()) * (1-self.target_tensor)).tolist()
        repr_num_acc_t2 = sum(repr_num_acc_t2_list) / float(torch.sum(1-self.target_tensor))
        repr_num_acc_t3_list = (torch.tensor(repr_num_acc_list, device=self.target_tensor.get_device()) *  self.target_tensor).tolist()
        repr_num_acc_t3 = sum(repr_num_acc_t3_list) / float(torch.sum(self.target_tensor))

        list_prob = list((torch.sigmoid(self.logit_tensor)).tolist())
        list_pred = list((F.relu(self.logit_tensor.sign())).tolist())
        list_target = list(self.target_tensor.tolist())
        value_auc = metrics.roc_auc_score(list(list_target), list(list_prob), average=None)
        # list_fpr, list_tpr, _ = metrics.roc_curve(list(list_target), list(list_prob))
        # roc_auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(list(list_target), list(list_pred)).ravel()
        # print(f'tn: {tn}, tp: {tp}, fn: {fn}, fp: {fp}')
        value_mcc = ((tp * tn) - (fp * fn)) / math.sqrt(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

        value_recall = tp / (tp + fn)
        value_precision = tp / (tp + fp)
        value_f1 = 2.0 * value_precision * value_recall / (value_precision + value_recall)
        value_PPV = tp / (tp + fp)
        value_NPV = tn / (tn + fn)
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)


        super(StatisticalAnalysis, self).update_last('tn_'+aug_type, tn)
        super(StatisticalAnalysis, self).update_last('tp_'+aug_type, tp)
        super(StatisticalAnalysis, self).update_last('fn_'+aug_type, fn)
        super(StatisticalAnalysis, self).update_last('fp_'+aug_type, fp)
        super(StatisticalAnalysis, self).update_last('auc_'+aug_type, value_auc)
        super(StatisticalAnalysis, self).update_last('precision_'+aug_type, value_precision)
        super(StatisticalAnalysis, self).update_last('recall_'+aug_type, value_recall)
        super(StatisticalAnalysis, self).update_last('f1_'+aug_type, value_f1)
        super(StatisticalAnalysis, self).update_last('PPV_'+aug_type, value_PPV)
        super(StatisticalAnalysis, self).update_last('NPV_'+aug_type, value_NPV)
        super(StatisticalAnalysis, self).update_last('mcc_'+aug_type, value_mcc)
        super(StatisticalAnalysis, self).update_last('accuracy_'+aug_type, accuracy)
        super(StatisticalAnalysis, self).update_last('sensitivity_'+aug_type, sensitivity)
        super(StatisticalAnalysis, self).update_last('specificity_'+aug_type, specificity)
        super(StatisticalAnalysis, self).update_last('repr_num_acc_all_'+aug_type, repr_num_acc_all)
        super(StatisticalAnalysis, self).update_last('repr_num_acc_t2_'+aug_type, repr_num_acc_t2)
        super(StatisticalAnalysis, self).update_last('repr_num_acc_t3_'+aug_type, repr_num_acc_t3)


if __name__ == '__main__':
    time_start = time.time()
    summary = {}
    summary_test = {}
    summary_setting = {}

    if config.getboolean('by_exp', 'if_with_mask'):
        summary['mask_size'] = config.get('by_exp', 'mask_size')
    summary = {'metric': ['tr mean', 'tr stddev', 'val mean', 'val stddev']}

    performance_metric_tr = {}
    performance_metric_val = {}
    performance_metric_stat = {}

    if 'rectum' in config['loss'].keys():
        if args.test_aug or args.test_noaug:
            performance_metric_stat['rectum'] = [AverageMeter('test', 'dice_rectum')]
        if args.train:
            performance_metric_tr['rectum'] = [AverageMeter('tr', 'dice_rectum')]
            # performance_metric_val['rectum'] = [AverageMeter( 'val', 'dice_rectum')]

    if 'cancer' in config['loss'].keys():
        if args.test_aug or args.test_noaug:
            performance_metric_stat['cancer'] = [AverageMeter('test', 'dice_cancer')]
        if args.train:
            performance_metric_tr['cancer'] = [AverageMeter('tr', 'dice_cancer')]
            # performance_metric_val['cancer'] = [AverageMeter( 'val', 'dice_cancer')]

    if 'category' in config['loss'].keys():
        if args.test_aug or args.test_noaug:
            performance_metric_stat['category'] = [StatisticalAnalysis('test', 'stats', args)]

        if args.train:
            performance_metric_tr['category'] = [AverageMeter('tr', 'accuracy'),
                                                 AverageMeter('tr', 'sensitivity'),
                                                 AverageMeter('tr', 'specificity'),
                                                 AverageMeter('tr', 'repr_num')]
            # performance_metric_val['category'] = [AverageMeter('val', 'accuracy'),
            #                                       AverageMeter('val', 'sensitivity'),
            #                                       AverageMeter('val', 'specificity')]


    for i in config['default'].gettuple('fold_num'):
    # for i in [5,6,7,8]:
        print('FOLD: ' + str(i))
        main(i, performance_metric_tr, performance_metric_val, performance_metric_stat)

        for key, value in PerformanceMetrics.performance_tr.items():
            if ('accuracy' in key) or ('sensitivity' in key) or ('specificity' in key):
                summary_test[key] = str(round(100.0 * statistics.mean(value), 1)) + ' ' + u"\u00B1" + ' ' + \
                                    str(round(100.0 * statistics.pstdev(value), 1))
            elif ('dice' in key):
                summary_test[key] = str(round(100.0 * statistics.mean(value), 1)) + ' ' + u"\u00B1" + ' ' + \
                                    str(round(100.0 * statistics.pstdev(value), 1))
            else:
                summary_test[key] = [statistics.mean(value), statistics.pstdev(value)]
        # for key, value in PerformanceMetrics.performance_val.items():
        #     summary[key].extend([statistics.mean(value), statistics.pstdev(value)])
        for key, value in PerformanceMetrics.performance_test.items():
            if ('auc' in key) or ('mcc' in key):
                summary_test[key] = str(round(statistics.mean(value), 3)) + ' ' + u"\u00B1" + ' ' + str(round(statistics.pstdev(value), 3))
            elif ('accuracy' in key) or ('sensitivity' in key) or ('specificity' in key):
                summary_test[key] = str(round(100.0 * statistics.mean(value), 1)) + ' ' + u"\u00B1" + ' ' + \
                                    str(round(100.0 * statistics.pstdev(value), 1))
            elif ('dice' in key):
                summary_test[key] = str(round(100.0 * statistics.mean(value), 1)) + ' ' + u"\u00B1" + ' ' + \
                                    str(round(100.0 * statistics.pstdev(value), 1))
            else:
                summary_test[key] = [statistics.mean(value), statistics.pstdev(value)]

            if args.test_aug or args.test_noaug:
                with open(os.path.join(dir_results, 'all_fold_test_aug.txt'), "w") as output:
                    output.write(str(PerformanceMetrics.performance_test))
                with open(os.path.join(dir_results, 'summary_test_aug.txt'), "w") as output:
                    output.write(str(summary_test))
            if args.train:
                with open(os.path.join(dir_results, 'all_fold_tr.txt'), "w") as output:
                    output.write(str(PerformanceMetrics.performance_tr))
                with open(os.path.join(dir_results, 'summary.txt'), "w") as output:
                    output.write(str(summary))


    summary_setting['node'] = str(args.node)
    summary_setting['gpu'] = str(args.gpu)
    summary_setting['num_workers'] = str(args.workers)
    summary_setting['dimension'] = str(config['by_exp']['dimension'])
    summary_setting['if_half_precision'] = str(config['by_exp']['if_half_precision'])
    summary_setting['batch_size'] = str(config['default']['batch_size'])
    summary_setting['network'] = str(config['by_exp']['network'])
    summary_setting['if_resume'] = str(args.resume)
    summary_setting['if_train'] = str(args.train)
    summary_setting['num_test_aug'] = str(args.test_aug)
    summary_setting['num_test_noaug'] = str(args.test_noaug)
    summary_setting['NETWORK_PARAM'] = str(NETWORK_PARAM)
    summary_setting['overall_time(sec)'] = str(int(time.time() - time_start))

    with open(os.path.join(dir_results, 'summary_setting.txt'), "w") as output:
        output.write(str(summary_setting))
