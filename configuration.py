import torch
import torch.nn as nn
import torch.utils.data as data
import os
import pickle

from model import Model
from const import IMAGES_FOLDER, ANNOTATION_FILE, MAIN_FOLDER
from dataset import MSCOCO

def conf_training(resuming=False, *args):
    """Function that initiates the configuration of the model depending if a last model
       is loaded or if it's the beginning of a new model"""
    
    #Data
    trainset = MSCOCO(IMAGES_FOLDER, ANNOTATION_FILE, train=True)
    evalset = MSCOCO(IMAGES_FOLDER, ANNOTATION_FILE, evalu=True)

    # Loss
    criterion = nn.MSELoss()
    
    # Number of epochs
    epochs = 10

    # Batch sizes
    batch_size_train = 1
    batch_size_val = 1
    
    if not resuming:
        # Model
        net = Model()

        # Optimizer
        optimizer = torch.optim.Adam(net.parameters())
        
        #First epoch
        current_epoch = 0
        
        #Loss train and val setup
        loss_train = []
        loss_val = []
    
    else:
        #Load the last saved model with its configurations
        checkpoint = torch.load(os.path.join(MAIN_FOLDER,"model_"+args[0])) #Take the last file - Later the best saved model
        
        #Model
        net = Model()
        net.load_state_dict(checkpoint['state_dict'])
        
        #Current_epoch
        current_epoch = checkpoint['epoch']
        
        #Optimizer
        #optimizer = load_state_dict(checkpoint['optimizer'])
        optimizer = torch.optim.Adam(net.parameters())
        
        #Loss train and val history
        lossObject = open(os.path.join(MAIN_FOLDER,"loss_" + args[0]),'rb')  
        loss_dic = pickle.load(lossObject)
        loss_train = loss_dic['loss_train']
        loss_val = loss_dic['loss_val']
    
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
    
    return epochs, trainloader, evaloader, optimizer, net, current_epoch, loss_train, loss_val, criterion, evalset_length, evalset