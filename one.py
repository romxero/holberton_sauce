#!/usr/bin/env python3
##
##
## Plz note that you do not need to reuse the tensor right here, it automatically reuse tensor/matrices in PyTorch 
##


from prompt_toolkit.layout import dimension
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
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
stdImageSize = 28 #squared size of image dimension 
numOfImageChannels = 1 #image channels 
samplerInterval = 400 #interval between sampling


#### End of global variables 

#grab the MINST data set within pytorch:
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
#grab training set and test set data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


#add this with the data loader method 
train_dataloader = DataLoader(training_data, batch_size=batchSize)
test_dataloader = DataLoader(test_data, batch_size=batchSize)


#probably not needed right here 
discriminator_activation_function = torch.sigmoid
generator_activation_function = torch.tanh



# Add the loss functions here and optimizers here 
loss_fn = nn.CrossEntropyLoss()
# Adversarial Loss function
adversarial_loss = torch.nn.BCELoss()

#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)








#Remove below 
#lets define the batch size:


#set up the devices right here 
#dtype = torch.float
#device = torch.device("cpu") # This executes all calculations on the CPU

#create a tensor type
# DEL # x = torch.tensor([[1, 2, 3], [4, 5, 6]])



#generator 
g_cpu = torch.Generator()



## For the model you have to define a method right here
class generatorNetwork(nn.Module):
    def __init__(self):
        super(generatorNetwork, self).__init__()
        self.flatten = nn.Flatten() #don't think I need to flatten right here 
        
        # Inputs layer: 128
        # Hidden layer: 784
        self.hidden = nn.Linear(128, 784)
        self.layer_1 = nn.Linear(128, 784) 
        
        
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(784, 10)
        self.layer_2 = nn.Linear(784, 10)

    def forward(self, x):
        forwardOutputs = [0,0,0,0,0] #array slice here to test

        #right here apply the first layer 
        x = self.layer_1(x)
        nn.ReLU()
        #next activation function 

        #doing this layer 2 right here
        x = self.layer_2(x)
        nn.Sigmoid() #sig
        #logits = self.linear_relu_stack(x)
        return forwardOutputs



## For the model you have to define a method right here
class descriminatorNetwork(nn.Module):
    def __init__(self):
        super(descriminatorNetwork, self).__init__()
        self.flatten = nn.Flatten() #don't think flattening is needed 

        # Inputs layer: 128
        # Hidden layer: 784
        self.hidden = nn.Linear(128, 128)
        self.layer_1 = nn.Linear(128, 128) 
        
        
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(128, 1)
        self.layer_2 = nn.Linear(128, 1)

    def forward(self, x):
        forwardOutputs = [0,0,0,0,0] #array slice here to test
        x = self.layer_1(x)
        nn.ReLU()
        #next activation function 

        #doing this layer 2 right here
        x = self.layer_2(x)
        nn.Sigmoid()
        return forwardOutputs

# The optimizers are here for the networks  
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



###
#Training
torch.random.seed()

for epochs in range(numOfEpochs):
    print(epochs)


##


