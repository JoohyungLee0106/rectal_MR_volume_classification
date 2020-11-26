import configparser
from configparser import ExtendedInterpolation

# dimension only for the network parameter

# argparse:
# node_id, workers (num_workers), gpu, evaluate

# ddp:
# multiprocessing_distributed, world_size, rank, dist_url, dist_backend

# network_params:
# dimension, backbone, att_layer, stride_D1

# by_exp
# mask_size, if_with_mask, network (arch), save_folder_name, seed

# loss_params
# gamma, if_logit

# by_node
# csv_tr, csv_val, result_save_directory, path_new,

# default
# max_epoch (epoch), fold_size, batch_size, if_replace_path, path_to_be_replaced, image_size, if_augment, fold_num,
# resume, lr_init, momentum, weight_decay, print_freq, resume, dist_backend
config = configparser.ConfigParser()

#todo
# config['ddp'] = {
#     'multiprocessing_distributed': '10',
#     'world_size': '16',
#     'rank' : 'True',
#     # Need to adjust this
#     'dist_url': 'tcp://224.66.41.62:23456',
#     'dist_backend' : 'nccl'
# }

# ini file stores all items in string..
# config['network_params'] = {
#     'dimension': '3D',
##     resnet_2d, r3d, mc2, mc3, mc4, mc5, rmc2, rmc3, rmc4, rmc5, rmc6, r2plus1d,
    # 'backbone': 'r3d',
    ## need to fix 'network_attention: 2 -> 3
    # 'att_layer': 3,
    # 'stride_D1': [1, 2, 2, 2]
# }

config['by_exp'] = {
    # 'mask_size': (7, 64, 64),
    'mask_size': (7, 256, 256),
    'if_with_mask': False,
    # 'if_with_mask': True,
    'if_half_precision': False,
    # ResNet, resnet_2d, r3d, mc2, mc3, mc4, mc5, rmc2, rmc3, rmc4, rmc5, rmc6, r2plus1d, ResNet_attention
    'network': 'r3d',
    'save_group_name' : 'cvpr2021_rc',
    # 'data_folder_name' : 'size_280',
    'data_folder_name' : 'cropped_margin_10pctg_270',
    'dimension': '3D',
    'weight_category': 1.0,
    'weight_rectum': 1.0,
    'weight_cancer': 1.0,
}
# To control everything
config['loss'] = {
    # 'gamma': 2,
    # 'if_logit': True,
    'category': 'FocalLoss',
    # 'rectum': 'DiceLoss',
    # 'cancer': 'DiceLoss'
}

config['loss_arg'] ={
    'category': 2.0,
    # 'rectum': 1.0,
    # 'cancer': 1.0
}

config['node'] = {
    'kaist_server': '',
    'kaist_desktop': '',
    'ncc_jhl': '',
    'ncc_oh': '',
    'ncc_or': '',
    'ncc_server': '',
    'ncc_rtx': '',
    'ncc_image': ''
}

config['default'] = {
    'max_epoch': 500,
    'batch_size': 8,
    'if_replace_path' : True,
    'path_to_be_replaced': 'D:/Rectum_exp/bmp',
    'image_size': 256,
    'if_augment': True,
    'fold_num': [1,2,3,4,5,6,7,8,9,10],
    # 'resume': '',
    # ImageNet default (lr): 0.1
    'lr_init' : 0.01,
    'momentum': 0.9,
    # ImageNet default (weight decay): 1e-4
    'weight_decay': 1e-2,
    'print_epoch_freq': 10,
    'oversampling': True,
    'seed': 1234,
    'lr_min': 1e-6,
    'scheduler_factor': 0.3,
    'scheduler_patience': 10,
    'end_patience': 40,
    # 'node': ('kaist_server', 'kaist_desktop', 'ncc_jhl', 'ncc_oh', 'ncc_or', 'ncc_server')
}
# with open('./config.ini', 'w') as f:
#     config.write(f)

######################################################################################################################
# config = configparser.ConfigParser()

# # core = 12
# gpu for monitor = None
# RAM = 128 gb
# PyTorch version: 1.6.0
config['kaist_server'] = {
    'csv_tr': '/home/joohyung/sdb/KAIST_NCC/data/data_path_3d_new/${by_exp:dimension}_training_fold_',
    'csv_val': '/home/joohyung/sdb/KAIST_NCC/data/data_path_3d_new/${by_exp:dimension}_validation_fold_',
    'result_save_directory': '/home/joohyung/sdb/KAIST_NCC/results/${by_exp:save_group_name}',
    'path_new' : '/home/joohyung/sdb/KAIST_NCC/data/${by_exp:data_folder_name}',
    'output_device': '1'
}
# # core = 12
# gpu for monitor = 1
# RAM = 31.3 gb
# PyTorch version: 1.5.1
config['kaist_desktop'] = {
    'csv_tr': '/home/joohyung/KAIST_NCC/data/data_path_3d_new/${by_exp:dimension}_training_fold_',
    'csv_val': '/home/joohyung/KAIST_NCC/data/data_path_3d_new/${by_exp:dimension}_validation_fold_',
    'result_save_directory': '/home/joohyung/KAIST_NCC/results/${by_exp:save_group_name}',
    'path_new' : '/home/joohyung/KAIST_NCC/data/${by_exp:data_folder_name}',
    'output_device': '0'
}
# # core = 12
# RAM = 16 gb
# PyTorch version: ??
config['ncc_jhl'] = {
    'csv_tr': 'C:/Rectum_exp/data_path_3d_new/${by_exp:dimension}_training_fold_',
    'csv_val': 'C:/Rectum_exp/data_path_3d_new/${by_exp:dimension}_validation_fold_',
    'result_save_directory': 'C:/Rectum_exp/results/${by_exp:save_group_name}',
    'path_new' : 'C:/Rectum_exp/${by_exp:data_folder_name}'
}
# # core = 14
config['ncc_oh'] = {
    'csv_tr': 'C:/Rectum_exp/data_path_3d_new/${by_exp:dimension}_training_fold_',
    'csv_val': 'C:/Rectum_exp/data_path_3d_new/${by_exp:dimension}_validation_fold_',
    'result_save_directory': 'C:/Rectum_exp/results/${by_exp:save_group_name}',
    'path_new' : 'C:/Rectum_exp/${by_exp:data_folder_name}',
    'output_device': '0'
}

config['ncc_or'] = {
    'csv_tr': 'C:/Rectum_exp/data_path_3d_new/${by_exp:dimension}_training_fold_',
    'csv_val': 'C:/Rectum_exp/data_path_3d_new/${by_exp:dimension}_validation_fold_',
    'result_save_directory': 'C:/Rectum_exp/${by_exp:save_group_name}',
    'path_new' : 'C:/Rectum_exp/${by_exp:data_folder_name}'
}
# # core = 12
# gpu for monitor = 1
# PyTorch version: 1.6.0
config['ncc_server'] = {
    'csv_tr': '/mnt/Rectum_exp/data_path_3d_new/${by_exp:dimension}_training_fold_',
    'csv_val': '/mnt/Rectum_exp/data_path_3d_new/${by_exp:dimension}_validation_fold_',
    'result_save_directory': '/mnt/Rectum_exp/${by_exp:save_group_name}',
    'path_new' : '/mnt/Rectum_exp/${by_exp:data_folder_name}',
    'output_device': '0'
}

# # core =
# gpu for monitor =
# PyTorch version:
config['ncc_rtx'] = {
    'csv_tr': '/home/jieun/code_rectal cancer/data_path_3d_new/${by_exp:dimension}_training_fold_',
    'csv_val': '/home/jieun/code_rectal cancer/data_path_3d_new/${by_exp:dimension}_validation_fold_',
    'result_save_directory': '/home/jieun/results_rectal cancer/${by_exp:save_group_name}',
    'path_new' : '/home/jieun/code_rectal cancer/${by_exp:data_folder_name}',
    'output_device': '0'
}

# # core = 28 (wow)
# gpu for monitor =
# PyTorch version: 1.6
config['ncc_image'] = {
    'csv_tr': '/home/server1/jieun/data_path_3d_new/${by_exp:dimension}_training_fold_',
    'csv_val': '/home/server1/jieun/data_path_3d_new/${by_exp:dimension}_validation_fold_',
    'result_save_directory': '/home/server1/jieun/${by_exp:save_group_name}',
    'path_new' : '/home/server1/jieun/${by_exp:data_folder_name}',
    'output_device': '1'
}

# # core =
# gpu for monitor =
# PyTorch version:
config['ncc_titan2'] = {
    'csv_tr': '/Desktop/jieun_cvpr/rectal_cancer/data_path_3d_new/${by_exp:dimension}_training_fold_',
    'csv_val': '/Desktop/jieun_cvpr/rectal_cancer/data_path_3d_new/${by_exp:dimension}_validation_fold_',
    'result_save_directory': '/Desktop/jieun_cvpr/rectal_cancer/${by_exp:save_group_name}',
    'path_new' : '/Desktop/jieun_cvpr/rectal_cancer/${by_exp:data_folder_name}',
    'output_device': '0'
}



with open('./config_rectumcrop_c.ini', 'w') as f:
      config.write(f)
