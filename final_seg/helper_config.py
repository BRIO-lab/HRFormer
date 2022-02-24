import torch
import sys
import torch.nn as nn
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

class Configuration:    
    def __init__(self):
        self.etl = {
                    'raw_data_file': "data/IEEE_data.csv", \
                    'processed_path': "data",\
                    'val_size':  0.2,\
                    'test_size': 0.01,\
                    'random_state': np.random.randint(1,50)
                    }
        self.data_constants = \
            {
                'IMAGE_HEIGHT': 1024,\
                'IMAGE_WIDTH' : 1024,\
                'MODEL_TYPE'  : 'fem',\
                
                'STORE_DATA_RAM' : True,\
                'IMAGES_GPU_DATA_TYPE' : torch.FloatTensor,\
                'LABELS_GPU_DATA_TYPE' : torch.FloatTensor, \
                'MAX_EPOCHS': 350,\
                'VALIDATION_BATCH_SIZE': 4,\
                
                'IMAGE_THRESHOLD': 0,\
                'NUM_PRINT_IMAGE': 2,\
                'ALPHA_IMG' : 0.3,\
                'IMG_CHANNELS' : 1,\
                
                'MODEL_NAME' : "FEM_ALL_DATA_ORS_21-12-03",\
                'IMAGE_DIRECTORY': "/blue/banks/ajensen123/JTML/JTML_ALL_DATA_ONE_FOLDER/",\
                'LOAD_FROM_CHECKPOINT' : False,\

                
        }
        
        self.data_loader_parameters = \
            {
                'BATCH_SIZE': 10,\
                'SHUFFLE': True,\
                'NUM_WORKERS': 0,\
                'PIN_MEMORY': False, \
                'SUBSET_IMAGES': False
            }
        
        self.train = \
            {
                "loss_fn" : nn.MSELoss()
            }

        self.test = \
            {
                'CUSTOM_TEST_SET': True, \
                'TEST_SET_NAME': "/ISTA_TEST.csv"
            }
        self.transform = \
        A.Compose([
        A.RandomGamma(always_apply=False, p = 0.5,gamma_limit=(10,300)),
        A.ShiftScaleRotate(always_apply = False, p = 0.5,shift_limit=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-180,180), interpolation=0, border_mode=0, value=(0, 0, 0)),
        A.Blur(always_apply=False, blur_limit=(3, 10), p=0.2),
        A.Flip(always_apply=False, p=0.5),
        A.ElasticTransform(always_apply=False, p=0.85, alpha=0.5, sigma=150, alpha_affine=50.0, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False),
        A.InvertImg(always_apply=False, p=0.5),
        A.CoarseDropout(always_apply = False, p = 0.25, min_holes = 1, max_holes = 100, min_height = 25, max_height=25),
        A.MultiplicativeNoise(always_apply=False, p=0.25, multiplier=(0.1, 2), per_channel=True, elementwise=True)
    ], p=0.85)