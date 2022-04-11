import torch

import sys
import torch.nn as nn
#import albumentations as A
import numpy as np
#from albumentations.pytorch import ToTensorV2


class Configuration:    
    def __init__(self):
        self.dataset = 'coco_stuff'
        self.method = 'fcn_sefmentor'
# TODO: figure out how to make this greyscale
        self.data = {
            "image_tool": "cv2",\
            "input_mode": "BGR", 
            "num_classes": 171,\
            "label_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 
                            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
                            40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
                            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 
                            78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 
                            97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 
                            113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 
                            129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 
                            145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 
                            161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 
                            177, 178, 179, 180, 181, 182],\
            "reduce_zero_label": True,\
            "data_dir": "~/DataSet/pascal_context",\
            "workers": 8
        }

        self.train = {
                "batch_size": 16,\
                "data_transformer": {
                    "size_mode": "fix_size",\
                    "input_size": [520, 520],\
                    "align_method": "only_pad",\
                    "pad_mode": "random"
                }
                
            }
        
        self.val = {
            "batch_size": 8,\
            "mode": "ss_test",\
            "data_transformer": {
                "size_mode": "diverse_size",\
                "align_method": "only_pad",\
                "pad_mode": "pad_right_down"
            }
        }
       
        self.test = {
            "batch_size": 8,\
            "mode": "ss_test",\
            "data_transformer": {
                "size_mode": "diverse_size",\
                "align_method": "only_pad",\
                "pad_mode": "pad_right_down"
      }
            }
        
        self.train_trans = {
            "trans_seq": ["random_hflip", "resize", "random_resize", "random_crop", "random_brightness"],\
            "random_brightness": {
                "ratio": 1.0,\
                "shift_value": 10
            },
            "resize": {
                "min_side_length": 520
            },
            "random_hflip": {
                "ratio": 0.5,\
                "swap_pair": []
            },
            "random_resize": {
                "ratio": 1.0,\
                "method": "random",\
                "scale_range": [0.5, 2.0],\
                "aspect_range": [0.9, 1.1]
            },
            "random_rotate": {
                "ratio": 1.0,\
                "rotate_degree": 10
            },
            "random_crop":{
                "ratio": 1.0,\
                "crop_size": [520, 520],\
                "method": "random",\
                "allow_outside_center": False
            }
        }
        
        self.val_trans = {
        "trans_seq": ["resize"],\
        "resize": {
            "min_side_length": 520
        },
        "random_crop": {
            "ratio": 1.0,\
            "crop_size": [520, 520],
            "method": "center",\
            "allow_outside_center": False
      }
        }

        self.normalize = {
            "div_value": 255.0,\
            "mean_value": [0.485, 0.456, 0.406],\
            "mean": [0.485, 0.456, 0.406],\
            "std": [0.229, 0.224, 0.225]
        }
        
        self.checkpoints = {
            "checkpoints_name": "fs_aspocnet_coco_stuff_seg",\
            "checkpoints_dir": "./checkpoints/coco_stuff",\
            "save_iters": 1000
        }

        self.network = {
            "backbone": "deepbase_resnet101_dilated8",\
            "model_name": "asp_ocnet",\
            "bn_type": "torchsyncbn",\
            # Added bn_momentum manually
            "bn_momentum": 0.1,\
            "stride": 8,\
            "factors": [[8, 8]],\
            "loss_weights": {
                "aux_loss": 0.4,\
                "seg_loss": 1.0
            }
        }

        self.logging = {
            "logfile_level": "info",\
            "stdout_level": "info",\
            "log_file": "./log/ade20k/fs_aspocnet_pascal_context_seg.log",\
            "log_format": "%(asctime)s %(levelname)-7s %(message)s",\
            "rewrite": True
        }

        self.lr = {
            "base_lr": 0.001,\
            "metric": "iters",\
            "lr_policy": "lambda_poly",\
            "warm": {
                "warm_iters": 1500,\
                "warm_ratio": 1e-6,\
                "freeze_backbone": False
            },
            "step": {
                "gamma": 0.5,\
                "step_size": 100
      }
        }

        self.solver = {
            "display_iter": 10,\
            "test_interval": 5000,\
            "max_iters": 30000
        }

        self.optim = {
            "optim_method": "sgd",\
            "adam": {
                "betas": [0.9, 0.999],\
                "eps": 1e-08,\
                "weight_decay": 0.0001
            },
            "adamw": {
                "betas": [0.9, 0.999],\
                "eps": 1e-08,\
                "weight_decay": 0.01
            },
            "sgd":{
                "weight_decay": 0.0001,\
                "momentum": 0.9,\
                "nesterov": False
      }
        }

        self.loss = {
            "loss_type": "fs_auxce_loss",\
            "params": {
                "ce_reduction": "elementwise_mean",\
                "ce_ignore_index": -1,\
                "ohem_minkeep": 100000,\
                "ohem_thresh": 0.9
            }
        }





    #     self.transform = \
    #     A.Compose([
    #     A.RandomGamma(always_apply=False, p = 0.5,gamma_limit=(10,300)),
    #     A.ShiftScaleRotate(always_apply = False, p = 0.5,shift_limit=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-180,180), interpolation=0, border_mode=0, value=(0, 0, 0)),
    #     A.Blur(always_apply=False, blur_limit=(3, 10), p=0.2),
    #     A.Flip(always_apply=False, p=0.5),
    #     A.ElasticTransform(always_apply=False, p=0.85, alpha=0.5, sigma=150, alpha_affine=50.0, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False),
    #     A.InvertImg(always_apply=False, p=0.5),
    #     A.CoarseDropout(always_apply = False, p = 0.25, min_holes = 1, max_holes = 100, min_height = 25, max_height=25),
    #     A.MultiplicativeNoise(always_apply=False, p=0.25, multiplier=(0.1, 2), per_channel=True, elementwise=True)
    # ], p=0.85)