import torch
import torch.nn as nn
import torch.utils.data as data
import os
import pickle

from model import Model
from const import IMAGES_FOLDER, ANNOTATION_FILE, MAIN_FOLDER
from dataset import MSCOCO

def conf_training(resuming=False, input_type=0, *args):
    """Function that initiates the configuration of the model depending if a last model
       is loaded or if it's the beginning of a new model"""
    
    #Data
    trainset = MSCOCO(IMAGES_FOLDER, ANNOTATION_FILE, train=True, input_type=input_type)
    evalset = MSCOCO(IMAGES_FOLDER, ANNOTATION_FILE, evalu=True, input_type=input_type)

    # Loss
    criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    
    # Number of epochs
    epochs = 10

    # Batch sizes
    batch_size_train = 25
    batch_size_val = 25
    
    if not resuming:
        # Model
        net = Model(input_type=input_type)
        # net = Model(input_type=input_type).cuda()

        # Optimizer
        optimizer = torch.optim.Adam(net.parameters())
        
        #First epoch
        current_epoch = -1
    
    else:
        #Load the last saved model with its configurations
        checkpoint = torch.load(os.path.join(MAIN_FOLDER,"model_"+args[0]))
        
        #Model
        net = Model(input_type=input_type)
        # net = Model(input_type=input_type).cuda()
        net.load_state_dict(checkpoint['state_dict'])
        
        #Current_epoch
        current_epoch = checkpoint['epoch']
        
        #Optimizer
        optimizer = torch.optim.Adam(net.parameters())
    
    #Data loaders
    trainloader = torch.utils.data.DataLoader(trainset,
                                         batch_size=batch_size_train,
                                         shuffle=True,
                                         num_workers=4
                                        )

    evaloader = torch.utils.data.DataLoader(evalset,
                                         batch_size=batch_size_val,
                                         shuffle=True,
                                         num_workers=4
                                        )
    
    evalset_length = len(evalset)
    
    return epochs, trainloader, evaloader, optimizer, net, current_epoch, criterion, evalset_length, evalset