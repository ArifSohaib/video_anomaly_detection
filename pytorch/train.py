"""
trains a given FCN model
"""
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import testFCN
from video_dataloader import H5VideoDataset
from torch.autograd import Variable
import numpy as np

def main():
    dataset = H5VideoDataset("../data/filtered_period1.h5")
    dataloader = DataLoader(dataset)


def train_model(model, criterion, optimizer, scheduler, dataloader, dataset_length, use_gpu, num_epochs=2):
    """
    Trains given model
    Args:
        model:
        criterion:
        optimizer:
        schedular:
        num_epochs:
    Returns:
        model: trained model
    """
    since = time.time()

    best_model_wts = model.state_dict()
    best_loss = 100000.0
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-"*10)
        running_loss = 0.0

        scheduler.step()

        #iterate over the data
        for i, data in enumerate(dataloader):
            if use_gpu:
                inputs = Variable(data.cuda())
            else:
                inputs = Variable(data)
            #zero the parameter gradients
            optimizer.zero_grad()
            
            #forward pass
            #get the output
            outputs = model(inputs)
            #calculate the loss
            loss = criterion(outputs, inputs)

            #backward pass
            #calculate gradients
            loss.backward()
            #update the weights
            optimizer.step()

            #get the statistics for display/debugging
            running_loss += loss.data[0]

            if(i%500==0):
                print(running_loss)
            
        epoch_loss = running_loss/dataset_length
        print('Epoch {} Loss: {:.4f}'.format(epoch, epoch_loss))

        #deep copy the model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = model.state_dict()
        
        print()
    time_elapsed = time.time() - since
    print("Training complete in {:.4f}m {:.4f}s".format(time_elapsed//60, time_elapsed%60))
    print("Best loss: {:.4f})".format(best_loss))

    model.load_state_dict(best_model_wts)
    return model








