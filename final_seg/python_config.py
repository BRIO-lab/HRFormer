import torch
import sys
import torch.nn as nn
#import albumentations as A
import numpy as np
#from albumentations.pytorch import ToTensorV2

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
                'MODEL_TYPE'  : 'tib',\
                
                'STORE_DATA_RAM' : True,\
                'IMAGES_GPU_DATA_TYPE' : torch.FloatTensor,\
                'LABELS_GPU_DATA_TYPE' : torch.FloatTensor, \
                'MAX_EPOCHS': 350,\
                'VALIDATION_BATCH_SIZE': 4,\
                
                'IMAGE_THRESHOLD': 0,\
                'NUM_PRINT_IMAGE': 2,\
                'ALPHA_IMG' : 0.3,\
                'IMG_CHANNELS' : 1,\
                
                'MODEL_NAME' : "TIB_ALL_DATA_21-12-13",\
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

        self.loss = \
            {\
        "loss_type": "fs_auxce_loss",
        "params": {
        "ce_reduction": "elementwise_mean",
        "ce_ignore_index": -1,
        "ohem_minkeep": 100000,
        "ohem_thresh": 0.9
            }
        }
