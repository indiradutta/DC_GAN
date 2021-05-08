''' This file contains the DCGAN module. '''

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

class DCGAN(object):
    def __init__(self, batch_size = 128, image_size = 64, nc = 3, nz = 100, ngf = 64, ndf = 64, num_epochs = 5, lr = 0.0002, beta1 = 0.5, ngpu = 1):
        
        ''' 
        batch_size = the batch size used in training, according to the paper it is 128.

        image_size = the spatial size of the image used for training, according to the paper it is 64*64.

        nc = number of color channels in an image, we have used 3 channels(RGB).

        nz = length of the latent vector that is initially passed into the Generator, according to the paper it is 100.

        ngf =  denotes the depth of the feature maps passed through the Generator, according to the paper it is 64.

        ndf = denotes the depth of the feature maps passed through the Discriminator, according to the paper it is 64.

        num_epochs = number of epochs to run during training, according to the paper it is 5.

        lr = learning rate for training, according to the paper it is 0.0002.

        beta1 = hyperparameter for Adam Optimizer, according to the paper it is 0.5.

        ngpu = number of GPUs available for training. If no GPU is available, the model will train on CPU. Here, we have only 1 GPU available.

        '''

        self.batch_size = batch_size
        self.image_size = image_size
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.num_epochs = num_epochs
        self.lr = lr
        self.beta1 = beta1
        self.ngpu = ngpu


class Generator(nn.Module):

    ''' Generator Model '''

    def __init__(self,ngpu,nz,ngf,nc):

        ''' initialising the variables '''

        super(Generator,self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        '''
        Building the model - 

        We have 4 fractionally-strided convolutions to help the model learn it's own upsampling methods.

        Following the above are 4 batch normalization layers to stabilize the learning by normalizing the input to each unit to have zero mean and unit variance
        and to ease the geadient flow in deeper layers.

        Finally we have 4 ReLU activation layers which allows the model to learn more quickly and cover the complete spatial extent of an image.

        The final layer has a tanh activation function which limits the feature values between 0 and 1. 

        '''
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias = False), #sride=1, padding=0
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False), #sride=2, padding=1
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False), #sride=2, padding=1
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False), #sride=2, padding=1
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  #sride=2, padding=1
            nn.Tanh()
        )

    ''' Function to forward the input into the model '''

    def forward(self,input):
        return self.main(input)

class Discriminator(nn.Module):

    ''' Discriminator Model '''

    def __init__(self,ngpu,ndf,nc):

        ''' initialising the variables '''

        super(Discriminator,self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.nc = nc

        '''
        Building the model - 

        We have 4 convolution layers for downsampling.

        Following the above, we have 4 LeakyReLU activation layers which according to the paper gives better results on the discriminator specially for
        higher-resolution images.

        The final layer has a sigmoid activation function that outputs the probabilty of an image being fake or real. 
        '''

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), #sride=2, padding=1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), #sride=2, padding=1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), #sride=2, padding=1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False), #sride=2, padding=1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False), #sride=1, padding=0
            nn.Sigmoid()
        )
        
    ''' Function to forward the input into the model '''

    def forward(self,input):
        return self.main(input)
