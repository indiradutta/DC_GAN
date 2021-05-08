''' This file contains the DCGAN module. '''

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

import os
import random

from IPython.display import HTML

class Generator(nn.Module):

    ''' Generator Model '''

    def __init__(self,ngpu, nz, ngf, nc):

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
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias = False), #stride=1, padding=0
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False), #stride=2, padding=1
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False), #stride=2, padding=1
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False), #stride=2, padding=1
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  #stride=2, padding=1
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
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), #stride=2, padding=1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), #stride=2, padding=1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), #stride=2, padding=1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False), #stride=2, padding=1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False), #stride=1, padding=0
            nn.Sigmoid()
        )
        
    ''' Function to forward the input into the model '''

    def forward(self,input):
        return self.main(input)

class DCGAN(object):

    def __init__(self, data = 'data/lsun', batch_size = 128, image_size = 64, nc = 3, nz = 100, ngf = 64, ndf = 64, num_epochs = 5, lr = 0.0002, beta1 = 0.5, ngpu = 1):
        
        ''' 
        The constructor has the Parameters which are going to be used to generate the images

        Parameters:

        - data(dafault: 'data/lsun'): path to the dataset used for training, according to the paper we shall use the Large-scale Scene Understanding (LSUN) dataset.

        - batch_size(default: 128): the batch size used in training, according to the paper it is 128.

        - image_size(default: 64): the spatial size of the image used for training, according to the paper it is 64*64.

        - nc(default: 3): number of color channels in an image, we have used 3 channels(RGB).

        - nz(default: 100): length of the latent vector that is initially passed into the Generator, according to the paper it is 100.

        - ngf(default: 64):  denotes the depth of the feature maps passed through the Generator, according to the paper it is 64.

        - ndf(default: 64): denotes the depth of the feature maps passed through the Discriminator, according to the paper it is 64.

        - num_epochs(default: 5): number of epochs to run during training, according to the paper it is 5.

        - lr(default: 0.0002): learning rate for training, according to the paper it is 0.0002.

        - beta1(default: 0.5): hyperparameter for Adam Optimizer, according to the paper it is 0.5.

        - ngpu(default: 1): number of GPUs available for training. If no GPU is available, the model will train on CPU. Here, we have only 1 GPU available.
        '''

        if ngpu > 0 and not torch.cuda.is_available():
            raise ValueError('ngpu > 0 but cuda not available')

        self.dataroot = data
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
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    '''
    creating the dataset and dataloader
    '''

    def data_loader(self, dataroot):

        ''' Creating the dataset '''
        dataset = dset.ImageFolder(root = self.dataroot,
                                transform = transforms.Compose([
                                transforms.Resize(self.image_size),
                                transforms.CenterCrop(self.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

        ''' Creating the dataloader '''
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size,
                                                shuffle = True)

        return dataloader

    
    '''
    randomly initializing model weights from a Normal distribution with mean = 0, stdev = 0.02 as mentioned in the DCGAN paper
    '''

    def weights_init(self, init_model):

        '''
        input: an initialized model

        output: reinitialized convolutional, convolutional-transpose, and batch normalization layers 
        '''

        classname = init_model.__class__.__name__

        if classname.find('Conv') != -1:
            nn.init.normal_(init_model.weight.data, 0.0, 0.02)
            
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(init_model.weight.data, 1.0, 0.02)
            nn.init.constant_(init_model.bias.data, 0)
            
