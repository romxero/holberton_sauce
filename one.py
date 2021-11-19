#!/usr/bin/env python3
##
##
## Plz note that you do not need to reuse the tensor right here, it automatically reuse tensor/matrices in PyTorch 
##


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import numpy as np 
import matplotlib.pyplot as plt


#lets define the batch size:
batch_size = 64 

#set up the devices right here 
dtype = torch.float
device = torch.device("cpu") # This executes all calculations on the CPU

#create a tensor type
x = torch.tensor([[1, 2, 3], [4, 5, 6]])


#set the random seed state
torch.random.seed()

#generator 
g_cpu = torch.Generator()


#generator right here 
def generator(Z):
	#torch.tensor()
# take a look at some other methods, that are below, instead of instantiating a sequential layer 


    generatorModel = nn.Sequential(
            nn.Linear(128),
            nn.ReLU(),
            nn.Linear(784),
            nn.ReLU(),
            nn.Sigmoid()
          )
    
    # ~ self.model = nn.Sequential(
            # ~ *block(opt.latent_dim, 128, normalize=False),
            # ~ *block(128, 256),
            # ~ *block(256, 512),
            # ~ *block(512, 1024),
            # ~ #nn.Linear(1024, int(np.prod(shape=(Z[0])))),
            # ~ nn.ReLU()
        )


    return logits



def discriminator(self, x):
	x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits

def train_discriminator(self, x):
	return 0

def train_generator(self, x):
	return 0

def sample_Z(self, x):
	return 0

def train_GAN(self, x):
	return 0





# Add the loss functions here and optimizers here 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Adversarial Loss function
adversarial_loss = torch.nn.BCELoss()

##Make sure to add backwards propagation down here: 












#this is the main file to create a generator in 
##
##
##
##


#lets download the training and test data from an online repository 

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)



#	encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#	transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
#	src = torch.rand(10, 32, 512)
#	out = transformer_encoder(src)

	#this is the layer stuff below right here  
#	layer_1 = nn.layer(128, activation='relu') #middle layer with relu activation
#    layer_2 = nn.layer(784, activation='sigmoid') #last layer with sigmoid activation    
# ~ __self.model = nn.Sequential(layer_1,layer_2,nn.ReLU()) #I think this should be the right syntax- its just psuedo code anyways 
    
