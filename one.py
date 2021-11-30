#!/usr/bin/env python3
##
##
## Plz note that you do not need to reuse the tensor right here, it automatically reuse tensor/matrices in PyTorch 
##


import os
import random
from prompt_toolkit.layout import dimension
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.parallel
import torch.utils.data

import numpy as np 
#import matplotlib.pyplot as plt 
import plotly as plty #plotly instead of seaborn 
#import seaborn as sns ##deprecated 


#global variables that the student shouldn't need to set, or use 
appNumOfCpu = 4 #set this variable here 
batchSize = 64 #batch size 
numOfEpochs = 200 #numbee of iterations for training
learningRate = 0.0002 #learning rate lr
learningRateV2 = 1e-3 #this is a different learning rate to try in the simulation..
decayOfFirstOrderMom = 0.5 #first order b1
decayOfSecondOrderMom = 0.999 # second order b2
dimensionalityOfLatentSpace = 100 #name speaks for itself 
stdImageSize = 64 #squared size of image dimension 
numOfImageChannels = 1 #image channels 
samplerInterval = 400 #interval between sampling




#### End of global variables 

#the actual data was modified 
#grab the MINST data set within pytorch:
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose([
    transforms.Resize(stdImageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    ])
)
#grab training set and test set data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose([
    transforms.Resize(stdImageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    ])

)

#add this with the data loader method 
train_dataloader = DataLoader(training_data, batch_size=batchSize)
test_dataloader = DataLoader(test_data, batch_size=batchSize)


# Add the loss functions here and optimizers here 
loss_fn = nn.CrossEntropyLoss()
# Adversarial Loss function
adversarial_loss = torch.nn.BCELoss()

#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#I dunno what these do, I just grabbed them from the pytorch tutorial
def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

#generator 
g_cpu = torch.Generator()
g_cpu.seed() #this is generating a random seed for sampling data in neural network


## For the model you have to define a method right here
class generatorNetwork(nn.Module):
    def __init__(self):
        super(generatorNetwork, self).__init__()
        self.flatten = nn.Flatten() #don't think I need to flatten right here 
        self.main = nn.Sequential(
            nn.Linear(128, 784),
            nn.ReLU(),
            nn.Linear(784, 10),
            nn.Sigmoid()
            )

        
        # Inputs layer: 128
        # Hidden layer: 784
        
        
        # Output layer, 10 units - one for each digit

    def forward(self, x):
        #we are not doing anything special
        output = self.main(x)
        return output



## For the model you have to define a method right here
class descriminatorNetwork(nn.Module):
    def __init__(self):
        super(descriminatorNetwork, self).__init__()
        self.flatten = nn.Flatten() #don't think flattening is needed 
        self.main = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
            )

        # Inputs layer: 128

    def forward(self, x):
        output = self.main(x) #main forward action right here 
        return output


# The optimizers are here for the networks  
# Optimizers
#optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



###
#Training
torch.random.seed() #I guess a random seed again? #Maybe we need to delete 
def train_GAN():
    #setting up the data right here 
    dfe, dre, ge, = 0
    d_real_data, d_fake_data, d_real_data, d_fake_data = None, None, None, None

    #probably not needed right here but here are the activator functions right here 
    discriminator_activation_function = torch.sigmoid
    generator_activation_function = torch.tanh


    #main iteration
    for epochs in range(numOfEpochs):
        ##  

        print(epochs)




