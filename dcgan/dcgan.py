'''
This file contains the main DCGAN module.

The results may be obtained by running 'train()'
'''

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

    def data_loader(self):

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
            
    
    def train(self, path):

        ''' loading the data '''
        dataloader = self.data_loader()

        ''' Creating the generator '''
        netG = Generator(self.ngpu, self.nz, self.ngf, self.nc).to(self.device)

        if (self.device.type == 'cuda') and (self.ngpu > 1):
            netG = nn.DataParallel(netG, list(range(self.ngpu)))

        ''' Apply the weights_init function to randomly initialize all weights from a distribution with mean=0, stdev=0.2 '''
        netG.apply(self.weights_init)

        ''' Creating the discriminator '''
        netD = Discriminator(self.ngpu, self.ndf, self.nc).to(self.device)

        if (self.device.type == 'cuda') and (self.ngpu > 1):
            netD = nn.DataParallel(netG, list(range(self.ngpu)))

        ''' Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2 '''
        netD.apply(self.weights_init)


        ''' loss function: we shall use the binary cross entropy loss as mentioned in the paper '''
        adversarial_loss = torch.nn.BCELoss()  

        ''' Creating batch of latent vectors that we will use to visualize the progression of the generator '''
        fixed_noise = torch.randn(64, self.nz, 1, 1, device = self.device)

        ''' defining real label as 1 and the fake label as 0, to be used when calculating the losses of Discriminator and Generator '''
        real_label = 1.
        fake_label = 0.

        ''' setting up two separate Adam optimizers, for Discriminator and G as specified in the DCGAN papers, with learning rate 0.0002 and Beta1 = 0.5 '''
        optimizerD = optim.Adam(netD.parameters(), lr =self.lr, betas = (self.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr =self.lr, betas = (self.beta1, 0.999))

        '''
        Training:

        - construct different mini-batches for real and fake images, and adjust G’s objective function to maximize log(D(G(z)))

        - Discriminator Training: update the discriminator by ascending its stochastic gradient, maximize log(D(x))+log(1−D(G(z)))

        - Generator Training: train the Generator by minimizing log(1-D(G(z)))
        '''

        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print('Training...')

        for epoch in range(self.num_epochs):
            for i, data in enumerate(dataloader, 0):
                
                ''' Training the Discriminator with real samples '''
                ''' updating Discriminator network: maximize log(D(x)) + log(1 - D(G(z))) ''' 
                netD.zero_grad()

                ''' creating batches of real samples from the dataset '''
                batch = data[0].to(self.device)
                b_size = batch.size(0)

                ''' creating the target tensor '''
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)

                ''' passing the batch of real samples through the discriminator '''
                output = netD(batch).view(-1)

                ''' calculating the discriminator error for real samples '''
                errorD_real = adversarial_loss(output, label)

                ''' calculating the gradients through backprop '''
                errorD_real.backward()
                Dx = output.mean().item()

                ''' Training the Discriminator with fake samples '''
                ''' generating a fake batch from the generator '''
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                fake_batch = netG(noise)
                label.fill_(fake_label)

                ''' passing the batch of fake samples through the discriminator '''
                output = netD(fake_batch.detach()).view(-1)

                ''' calculating the discriminator error for fake samples '''
                errorD_fake = adversarial_loss(output, label)

                ''' calculating the gradients through backprop '''
                errorD_fake.backward()
                Dz = output.mean().item()
                
                ''' computing the final discriminator error '''
                errorD = errorD_fake + errorD_real

                ''' updating the discrimintor '''
                optimizerD.step()


                ''' Training the Generator '''
                netG.zero_grad()

                ''' creating the target tensor '''
                label.fill_(real_label)

                ''' passing the batch of fake samples through the discriminator '''
                output = netD(fake_batch).view(-1)

                ''' calculating the generator error '''
                errorG = adversarial_loss(output, label)

                ''' calculating the gradients through backprop '''
                errorG.backward()
                Gz = output.mean().item()

                ''' updating the generator '''
                optimizerG.step()

                ''' output training steps '''
                if i%50==0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch+1, self.num_epochs, i, len(dataloader), errorD.item(), errorG.item(), Dx, Dz, Gz))

                ''' saving the losses from the discriminator and generator '''
                G_losses.append(errorG.item())
                D_losses.append(errorD.item())

                if (iters % 500 == 0) or ((epoch == self.num_epochs-1) and (i==len(dataloader)-1)):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters+=1
        
        ''' saving the model weights and losses '''
        torch.save({
            'generator_state_dict': netG.state_dict(),
            'discriminator_state_dict': netD.state_dict(),
            'G_losses': G_losses,
            'D_losses': D_losses
            }, path) 
                
        return img_list, G_losses, D_losses
