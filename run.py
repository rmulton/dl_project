import matplotlib.pyplot as plt
from torch.autograd.variable import Variable
import pickle
import os
import torch
import torch.nn as nn
import torch.utils.data as data


from model import plotKeypointsOverOutputModel, Model
from const import IMAGES_FOLDER, IMAGES_FOLDER_TEST, ANNOTATION_FILE_TEST, MAIN_FOLDER
from dataset import MSCOCO
from configuration import conf_training

def training(epochs, trainloader, evaloader, optimizer, net, current_epoch, criterion, evalset_length, evalset):
    plt.ion()
    # if current_epoch == -1:
    #     #If not resuming a model, creating the loss file
    #     lossFile = open(os.path.join(MAIN_FOLDER,"loss"),'wb')
    #     pickle.dump({"loss_train":{}, "loss_val":{}},lossFile)
    #     lossFile.close()
    
    start_epoch = current_epoch + 1
    for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
        print("Epoch number {}".format(epoch))

        lossFile = open(os.path.join(MAIN_FOLDER,"loss"+str(epoch)),'wb')
        pickle.dump({"loss_train":{}, "loss_val":{}},lossFile)
        lossFile.close()

        #plotKeypointsOverOutputModel(0,evalset,net,IMAGES_FOLDER)#Displaying the result over the first element of the evalset
        running_loss = 0.0

        #For each epoch, we keep the loss under a dictionnary with epoch_nb as key and list of loss as value ;
        lossFile = open(os.path.join(MAIN_FOLDER,"loss"+str(epoch)),'rb')
        loss_dic = pickle.load(lossFile)
        lossFile.close()
        lossFile = open(os.path.join(MAIN_FOLDER,"loss"+str(epoch)),'wb')
        loss_dic['loss_train'][epoch] = []
        loss_dic['loss_val'][epoch] = []
        pickle.dump(loss_dic,lossFile)
        lossFile.close()

        for i, data in enumerate(trainloader, 0):
            print("Batch number {}".format(i))
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('Trainset loss[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            
            #Save the loss_train in disk for each batch
            lossFile = open(os.path.join(MAIN_FOLDER,"loss"+str(epoch)),'rb')
            loss_dic = pickle.load(lossFile)
            lossFile.close()
            lossFile = open(os.path.join(MAIN_FOLDER,"loss"+str(epoch)),'wb')
            loss_dic['loss_train'][epoch] += [loss.data[0]]
            pickle.dump(loss_dic,lossFile)
            lossFile.close()
        
        #Save the model
        net.cpu()
        state = {
            'epoch': epoch,
            'state_dict': net.state_dict()
        }
        torch.save(state, os.path.join(MAIN_FOLDER,"model_"+str(epoch))) #Save the torch model after each epoch
        
        net.cuda()
        running_loss_eval = 0.0
        print("Starting Eval for Epoch {}".format(epoch))
        for i, data in enumerate(evaloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # forward 
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # print statistics
            running_loss_eval += loss.data[0]

            #Save the loss_val in disk for each batch
            lossFile = open(os.path.join(MAIN_FOLDER,"loss"+str(epoch)),'rb')  
            loss_dic = pickle.load(lossFile)
            lossFile.close()
            lossFile = open(os.path.join(MAIN_FOLDER,"loss"+str(epoch)),'wb') 
            loss_dic['loss_val'][epoch] += [loss.data[0]]
            pickle.dump(loss_dic,lossFile)
            lossFile.close()

        print("Evalset Loss for Epoch {0} : {1}".format(epoch,running_loss_eval/evalset_length))
        #loss_val[epoch] += [running_loss_eval/evalset_length] #Stock the loss on evalset for each epoch
    
        

    print('Finished Training')

def launch_training(resuming=False, input_type=0, *args):
    """Function that configurates the model from init or a last model ; and then it trains the model"""
    epochs, trainloader, evaloader, optimizer, net, current_epoch, criterion, evalset_length, evalset = conf_training(resuming=resuming,input_type=input_type, *args)
    training(epochs, trainloader, evaloader, optimizer, net, current_epoch, criterion, evalset_length, evalset)

def launch_testing(model_epoch, input_type=0):
    """Function that launches a model over the test dataset"""
    testset = MSCOCO(IMAGES_FOLDER_TEST, ANNOTATION_FILE_TEST,input_type=input_type)

    #Load the training model
    checkpoint = torch.load(os.path.join(MAIN_FOLDER, model_epoch))
    net = Model(input_type=input_type)
    net.load_state_dict(checkpoint['state_dict'])

    # Loss
    criterion = nn.MSELoss()

    # Batch sizes
    batch_size_test = 1

    #TestLoader
    evaloader = torch.utils.data.DataLoader(testset,
                                            batch_size=batch_size_test,
                                            shuffle=True,
                                            num_workers=4
                                            )

    loss_test = 0.0
    for i, data in enumerate(evaloader):
        inputs, labels = data[0], data[1]
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = net(inputs)
        loss = criterion(y, outputs)
        loss_test += loss.data[0]
        if i % 500 ==0:
            print("Current loss over the test dataset: {0} after {1}ème iteration".format(loss_test/(i+1),i+1))

    loss_test = loss_test/len(testset)
    print("Average loss over the test dataset: {}".format(loss_test))
